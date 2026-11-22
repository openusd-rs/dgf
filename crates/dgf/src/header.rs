//! Semantic DGF1 block header representation, exponent types, and checked parsing.

use crate::bits::BLOCK_BYTES;
use crate::error::DecodeError;

/// Fixed DGF1 magic byte at header byte 0.
pub const MAGIC: u8 = 0x06;
/// Fixed header size in bytes (5 little-endian 32-bit words).
pub const HEADER_BYTES: usize = 20;
/// Fixed header size in bits.
pub const HEADER_BITS: usize = HEADER_BYTES * 8;
/// Maximum vertices per DGF1 block.
pub const MAX_VERTICES: u8 = 64;
/// Maximum triangles per DGF1 block.
pub const MAX_TRIANGLES: u8 = 64;
/// Minimum biased exponent stored in a DGF1 header (`2^-126`).
pub const MIN_BIASED_EXPONENT: u8 = 1;
/// Maximum biased exponent stored in a DGF1 header (`2^105`).
pub const MAX_BIASED_EXPONENT: u8 = 232;
/// Bias subtracted from stored exponent to obtain unbiased exponent.
pub const EXPONENT_BIAS: i16 = 127;
/// Minimum unbiased quantization exponent (`-126`).
pub const MIN_UNBIASED_EXPONENT: i16 = -126;
/// Maximum unbiased quantization exponent (`105`).
pub const MAX_UNBIASED_EXPONENT: i16 = 105;
/// Minimum signed 24-bit position anchor (`-8_388_608`).
pub const ANCHOR_MIN: i32 = -(1 << 23);
/// Maximum signed 24-bit position anchor (`8_388_607`).
pub const ANCHOR_MAX: i32 = (1 << 23) - 1;
/// Maximum 24-bit geometry ID (`16_777_215`).
pub const MAX_GEOM_ID: u32 = (1 << 24) - 1;
/// Maximum 9-bit geometry ID expressible in constant mode (`511`).
pub const MAX_CONSTANT_GEOM_ID: u32 = 511;
/// Maximum distinct `(geometry_id, opaque)` entries in palette mode.
pub const MAX_GEOM_PALETTE_ENTRIES: u8 = 32;
/// Maximum geometry palette common-prefix length in bits.
pub const MAX_GEOM_PREFIX_BITS: u8 = 25;
/// Maximum 29-bit primitive ID (`536_870_911`).
pub const MAX_PRIM_ID: u32 = (1 << 29) - 1;
/// Maximum OMM descriptor count per block (`7`).
pub const MAX_OMM_DESCRIPTORS: u8 = 7;
/// Minimum repeated-index bit width (`3`).
pub const MIN_BITS_PER_INDEX: u8 = 3;
/// Maximum repeated-index bit width (`6`).
pub const MAX_BITS_PER_INDEX: u8 = 6;

/// Validated biased DGF1 exponent in `1..=232` (representing scale `2^(E - 127)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BiasedExponent(u8);

impl BiasedExponent {
    /// Constructs a validated biased exponent from a raw header byte (`1..=232`).
    pub const fn new(raw: u8) -> Result<Self, DecodeError> {
        if raw < MIN_BIASED_EXPONENT || raw > MAX_BIASED_EXPONENT {
            Err(DecodeError::InvalidExponent { raw })
        } else {
            Ok(Self(raw))
        }
    }

    /// Returns the raw stored biased exponent (`1..=232`).
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// Converts to the corresponding unbiased [`QuantizationExponent`] (`-126..=105`).
    pub const fn unbiased(self) -> QuantizationExponent {
        QuantizationExponent(self.0 as i16 - EXPONENT_BIAS)
    }

    /// Returns the exact power-of-two scale `2^(E - 127)` as an `f32`.
    pub const fn scale_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 23)
    }

    /// Returns the exact power-of-two scale `2^(E - 127)` as an `f64`.
    pub const fn scale_f64(self) -> f64 {
        let f64_biased = ((self.0 as i32) - (EXPONENT_BIAS as i32) + 1023) as u64;
        f64::from_bits(f64_biased << 52)
    }
}

/// Validated unbiased quantization exponent in `-126..=105` (representing grid step `2^e`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QuantizationExponent(i16);

impl QuantizationExponent {
    /// Constructs a validated unbiased exponent (`-126..=105`).
    pub const fn new(exponent: i16) -> Option<Self> {
        if exponent < MIN_UNBIASED_EXPONENT || exponent > MAX_UNBIASED_EXPONENT {
            None
        } else {
            Some(Self(exponent))
        }
    }

    /// Returns the unbiased exponent (`-126..=105`).
    pub const fn value(self) -> i16 {
        self.0
    }

    /// Converts to the stored [`BiasedExponent`] (`1..=232`).
    pub const fn biased(self) -> BiasedExponent {
        BiasedExponent((self.0 + EXPONENT_BIAS) as u8)
    }

    /// Returns the exact grid step `2^e` as an `f64`.
    pub const fn step_f64(self) -> f64 {
        self.biased().scale_f64()
    }

    /// Returns the exact grid step `2^e` as an `f32`.
    pub const fn step_f32(self) -> f32 {
        self.biased().scale_f32()
    }
}

/// Geometry ID and opacity header mode decoded from `geom_id_mode` and `geom_id_meta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeomIdHeaderMode {
    /// All triangles in the block share a single 9-bit geometry ID and opaque flag.
    Constant {
        /// Constant geometry ID (`0..=511`).
        geometry_id: u16,
        /// Constant opaque flag.
        opaque: bool,
    },
    /// Per-triangle geometry ID and opaque flag stored in the geometry palette section.
    Palette {
        /// Shared high-bit prefix length (`0..=25`).
        prefix_bits: u8,
        /// Number of distinct palette entries (`1..=32`).
        entry_count: u8,
    },
}

/// Validated semantic fields of a DGF1 block header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Header {
    /// Bit width of each repeated vertex index (`3..=6`).
    pub bits_per_index: u8,
    /// Number of stored vertices in the block (`1..=64`).
    pub num_vertices: u8,
    /// Number of triangles in the block (`1..=64`).
    pub num_triangles: u8,
    /// Constant or palette geometry/opacity mode.
    pub geom_id_mode: GeomIdHeaderMode,
    /// Biased position quantization exponent (`1..=232`).
    pub exponent: BiasedExponent,
    /// Sign-extended 24-bit integer position anchor `[x, y, z]`.
    pub anchor: [i32; 3],
    /// Position offset bit widths `[x_bits, y_bits, z_bits]` (`1..=16`, sum multiple of 4).
    pub offset_bits: [u8; 3],
    /// Number of OMM descriptors (`0..=7`).
    pub omm_descriptor_count: u8,
    /// 29-bit primitive ID base (`prim_id_base + num_triangles - 1 <= MAX_PRIM_ID`).
    pub prim_id_base: u32,
    /// Whether a 4-byte user-data word follows the header.
    pub have_user_data: bool,
}

/// Raw diagnostic inspection of a 128-byte block header without assuming full validity.
///
/// Do not treat this struct as proof that a block is valid; use [`crate::validate_block`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawHeaderInspection {
    /// Raw magic byte (expected `0x06`).
    pub magic: u8,
    /// Decoded `bits_per_index` (`3..=6`).
    pub bits_per_index: u8,
    /// Decoded `num_vertices` (`1..=64`).
    pub num_vertices: u8,
    /// Decoded `num_triangles` (`1..=64`).
    pub num_triangles: u8,
    /// Raw 10-bit `geom_id_meta` field.
    pub geom_id_meta: u16,
    /// Raw `geom_id_mode` bit (`0` = constant, `1` = palette).
    pub geom_id_mode_bit: u8,
    /// Raw biased exponent byte.
    pub raw_exponent: u8,
    /// Sign-extended 24-bit anchor `[x, y, z]`.
    pub anchor: [i32; 3],
    /// Decoded offset bit widths `[x_bits, y_bits, z_bits]`.
    pub offset_bits: [u8; 3],
    /// Raw `omm_descriptor_count` (`0..=7`).
    pub omm_descriptor_count: u8,
    /// Raw 29-bit `prim_id_base`.
    pub prim_id_base: u32,
    /// Whether `have_user_data` bit is set.
    pub have_user_data: bool,
    /// Raw 2-bit reserved field (must be 0 in valid blocks).
    pub reserved: u8,
}

/// Sign-extends the low 24 bits of `raw` into a signed `i32` in `[-8_388_608, 8_388_607]`.
pub const fn sign_extend_24(raw: u32) -> i32 {
    ((raw << 8) as i32) >> 8
}

/// Packs a signed integer in `[-8_388_608, 8_388_607]` into its 24-bit two's-complement representation.
pub const fn pack_signed_24(value: i32) -> Option<u32> {
    if value < ANCHOR_MIN || value > ANCHOR_MAX {
        None
    } else {
        Some((value as u32) & 0x00FF_FFFF)
    }
}

/// Extracts raw header fields for diagnostic inspection without rejecting malformed blocks.
///
/// Palette prefix bits are **not** read here because their offset depends on user data,
/// vertex offsets, and OMM palette sizes.
pub fn inspect_header(bytes: &[u8; BLOCK_BYTES]) -> RawHeaderInspection {
    let w0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let w1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let w2 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let w3 = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let w4 = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);

    RawHeaderInspection {
        magic: (w0 & 0xFF) as u8,
        bits_per_index: (((w0 >> 8) & 0x3) as u8) + 3,
        num_vertices: (((w0 >> 10) & 0x3F) as u8) + 1,
        num_triangles: (((w0 >> 16) & 0x3F) as u8) + 1,
        geom_id_meta: ((w0 >> 22) & 0x3FF) as u16,
        geom_id_mode_bit: ((w3 >> 7) & 0x1) as u8,
        raw_exponent: (w1 & 0xFF) as u8,
        anchor: [
            sign_extend_24(w1 >> 8),
            sign_extend_24(w2 >> 8),
            sign_extend_24(w3 >> 8),
        ],
        offset_bits: [
            ((w2 & 0xF) as u8) + 1,
            (((w2 >> 4) & 0xF) as u8) + 1,
            ((w3 & 0xF) as u8) + 1,
        ],
        omm_descriptor_count: ((w3 >> 4) & 0x7) as u8,
        prim_id_base: w4 & MAX_PRIM_ID,
        have_user_data: ((w4 >> 29) & 0x1) != 0,
        reserved: ((w4 >> 30) & 0x3) as u8,
    }
}

impl Header {
    /// Parses and validates all header-level constraints from a 128-byte block.
    pub fn parse(bytes: &[u8; BLOCK_BYTES]) -> Result<Self, DecodeError> {
        let raw = inspect_header(bytes);

        if raw.magic != MAGIC {
            return Err(DecodeError::InvalidMagic { found: raw.magic });
        }
        if raw.reserved != 0 {
            return Err(DecodeError::ReservedBitsNonZero { bits: raw.reserved });
        }

        let exponent = BiasedExponent::new(raw.raw_exponent)?;

        let width_sum = raw.offset_bits[0] + raw.offset_bits[1] + raw.offset_bits[2];
        if !width_sum.is_multiple_of(4) {
            return Err(DecodeError::InvalidVertexWidthSum {
                x_bits: raw.offset_bits[0],
                y_bits: raw.offset_bits[1],
                z_bits: raw.offset_bits[2],
                sum: width_sum,
            });
        }

        let geom_id_mode = if raw.geom_id_mode_bit == 0 {
            GeomIdHeaderMode::Constant {
                geometry_id: raw.geom_id_meta >> 1,
                opaque: (raw.geom_id_meta & 1) != 0,
            }
        } else {
            let prefix_bits = (raw.geom_id_meta & 0x1F) as u8;
            if prefix_bits > MAX_GEOM_PREFIX_BITS {
                return Err(DecodeError::InvalidGeomPrefixLength { prefix_bits });
            }
            let entry_count = ((raw.geom_id_meta >> 5) as u8) + 1;
            GeomIdHeaderMode::Palette {
                prefix_bits,
                entry_count,
            }
        };

        let last_prim = u64::from(raw.prim_id_base) + u64::from(raw.num_triangles - 1);
        if last_prim > u64::from(MAX_PRIM_ID) {
            return Err(DecodeError::PrimitiveIdOverflow {
                prim_id_base: raw.prim_id_base,
                num_triangles: raw.num_triangles,
            });
        }

        Ok(Self {
            bits_per_index: raw.bits_per_index,
            num_vertices: raw.num_vertices,
            num_triangles: raw.num_triangles,
            geom_id_mode,
            exponent,
            anchor: raw.anchor,
            offset_bits: raw.offset_bits,
            omm_descriptor_count: raw.omm_descriptor_count,
            prim_id_base: raw.prim_id_base,
            have_user_data: raw.have_user_data,
        })
    }

    /// Serializes a validated header into the first 20 bytes of `bytes`.
    pub fn write(&self, bytes: &mut [u8; BLOCK_BYTES]) -> Result<(), DecodeError> {
        if self.bits_per_index < MIN_BITS_PER_INDEX || self.bits_per_index > MAX_BITS_PER_INDEX {
            return Err(DecodeError::BitOutOfBounds { start: 8, width: 2 });
        }
        if self.num_vertices == 0 || self.num_vertices > MAX_VERTICES {
            return Err(DecodeError::VertexIndexOutOfRange {
                strip_position: 0,
                index: 0,
                num_vertices: self.num_vertices,
            });
        }
        if self.num_triangles == 0 || self.num_triangles > MAX_TRIANGLES {
            return Err(DecodeError::ArithmeticOverflow {
                context: "num_triangles out of 1..=64",
            });
        }
        if self.offset_bits.iter().any(|&w| w == 0 || w > 16) {
            return Err(DecodeError::InvalidVertexWidthSum {
                x_bits: self.offset_bits[0],
                y_bits: self.offset_bits[1],
                z_bits: self.offset_bits[2],
                sum: self.offset_bits[0]
                    .saturating_add(self.offset_bits[1])
                    .saturating_add(self.offset_bits[2]),
            });
        }
        let width_sum = self.offset_bits[0] + self.offset_bits[1] + self.offset_bits[2];
        if !width_sum.is_multiple_of(4) {
            return Err(DecodeError::InvalidVertexWidthSum {
                x_bits: self.offset_bits[0],
                y_bits: self.offset_bits[1],
                z_bits: self.offset_bits[2],
                sum: width_sum,
            });
        }
        if self.omm_descriptor_count > MAX_OMM_DESCRIPTORS {
            return Err(DecodeError::OmmSelectorOutOfRange {
                triangle: 0,
                selector: self.omm_descriptor_count,
                descriptor_count: MAX_OMM_DESCRIPTORS,
            });
        }
        let last_prim = u64::from(self.prim_id_base) + u64::from(self.num_triangles - 1);
        if self.prim_id_base > MAX_PRIM_ID || last_prim > u64::from(MAX_PRIM_ID) {
            return Err(DecodeError::PrimitiveIdOverflow {
                prim_id_base: self.prim_id_base,
                num_triangles: self.num_triangles,
            });
        }

        let (geom_mode_bit, geom_meta): (u32, u32) = match self.geom_id_mode {
            GeomIdHeaderMode::Constant {
                geometry_id,
                opaque,
            } => {
                if u32::from(geometry_id) > MAX_CONSTANT_GEOM_ID {
                    return Err(DecodeError::ArithmeticOverflow {
                        context: "constant geometry_id exceeds 511",
                    });
                }
                (0, (u32::from(geometry_id) << 1) | u32::from(opaque))
            }
            GeomIdHeaderMode::Palette {
                prefix_bits,
                entry_count,
            } => {
                if prefix_bits > MAX_GEOM_PREFIX_BITS {
                    return Err(DecodeError::InvalidGeomPrefixLength { prefix_bits });
                }
                if entry_count == 0 || entry_count > MAX_GEOM_PALETTE_ENTRIES {
                    return Err(DecodeError::ArithmeticOverflow {
                        context: "palette entry_count out of 1..=32",
                    });
                }
                (
                    1,
                    u32::from(prefix_bits) | (u32::from(entry_count - 1) << 5),
                )
            }
        };

        let x_anc = pack_signed_24(self.anchor[0]).ok_or(DecodeError::ArithmeticOverflow {
            context: "x_anchor out of signed 24-bit range",
        })?;
        let y_anc = pack_signed_24(self.anchor[1]).ok_or(DecodeError::ArithmeticOverflow {
            context: "y_anchor out of signed 24-bit range",
        })?;
        let z_anc = pack_signed_24(self.anchor[2]).ok_or(DecodeError::ArithmeticOverflow {
            context: "z_anchor out of signed 24-bit range",
        })?;

        let w0: u32 = u32::from(MAGIC)
            | (u32::from(self.bits_per_index - 3) << 8)
            | (u32::from(self.num_vertices - 1) << 10)
            | (u32::from(self.num_triangles - 1) << 16)
            | (geom_meta << 22);
        let w1: u32 = u32::from(self.exponent.raw()) | (x_anc << 8);
        let w2: u32 = u32::from(self.offset_bits[0] - 1)
            | (u32::from(self.offset_bits[1] - 1) << 4)
            | (y_anc << 8);
        let w3: u32 = u32::from(self.offset_bits[2] - 1)
            | (u32::from(self.omm_descriptor_count) << 4)
            | (geom_mode_bit << 7)
            | (z_anc << 8);
        let w4: u32 = self.prim_id_base | (u32::from(self.have_user_data) << 29);

        bytes[0..4].copy_from_slice(&w0.to_le_bytes());
        bytes[4..8].copy_from_slice(&w1.to_le_bytes());
        bytes[8..12].copy_from_slice(&w2.to_le_bytes());
        bytes[12..16].copy_from_slice(&w3.to_le_bytes());
        bytes[16..20].copy_from_slice(&w4.to_le_bytes());
        Ok(())
    }

    /// Returns the checked primitive ID for local triangle `tri_index`.
    pub const fn prim_id(&self, tri_index: u8) -> Option<u32> {
        if tri_index >= self.num_triangles {
            None
        } else {
            Some(self.prim_id_base + tri_index as u32)
        }
    }

    /// Updates `prim_id_base` in an already-encoded block header after verifying the new 29-bit range.
    pub fn update_prim_id_base(
        bytes: &mut [u8; BLOCK_BYTES],
        new_prim_id_base: u32,
    ) -> Result<(), DecodeError> {
        let header = Self::parse(bytes)?;
        let last_prim = u64::from(new_prim_id_base) + u64::from(header.num_triangles - 1);
        if new_prim_id_base > MAX_PRIM_ID || last_prim > u64::from(MAX_PRIM_ID) {
            return Err(DecodeError::PrimitiveIdOverflow {
                prim_id_base: new_prim_id_base,
                num_triangles: header.num_triangles,
            });
        }
        let w4 = new_prim_id_base | (u32::from(header.have_user_data) << 29);
        bytes[16..20].copy_from_slice(&w4.to_le_bytes());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_extension_boundaries() {
        assert_eq!(sign_extend_24(0x0000_0000), 0);
        assert_eq!(sign_extend_24(0x007F_FFFF), 8_388_607);
        assert_eq!(sign_extend_24(0x0080_0000), -8_388_608);
        assert_eq!(sign_extend_24(0x00FF_FFFF), -1);
        assert_eq!(pack_signed_24(0), Some(0x0000_0000));
        assert_eq!(pack_signed_24(8_388_607), Some(0x007F_FFFF));
        assert_eq!(pack_signed_24(-8_388_608), Some(0x0080_0000));
        assert_eq!(pack_signed_24(-1), Some(0x00FF_FFFF));
        assert_eq!(pack_signed_24(8_388_608), None);
        assert_eq!(pack_signed_24(-8_388_609), None);
    }

    #[test]
    fn exponent_boundaries_and_scales() {
        assert!(BiasedExponent::new(0).is_err());
        assert!(BiasedExponent::new(233).is_err());
        assert!(BiasedExponent::new(255).is_err());

        let min_e = BiasedExponent::new(1).unwrap();
        assert_eq!(min_e.unbiased().value(), -126);
        assert_eq!(min_e.scale_f32(), f32::MIN_POSITIVE);
        assert_eq!(min_e.scale_f64(), 2.0f64.powi(-126));

        let zero_e = BiasedExponent::new(127).unwrap();
        assert_eq!(zero_e.unbiased().value(), 0);
        assert_eq!(zero_e.scale_f32(), 1.0);
        assert_eq!(zero_e.scale_f64(), 1.0);

        let max_e = BiasedExponent::new(232).unwrap();
        assert_eq!(max_e.unbiased().value(), 105);
        assert_eq!(max_e.scale_f32(), 2.0f32.powi(105));
        assert_eq!(max_e.scale_f64(), 2.0f64.powi(105));
        assert!(((ANCHOR_MAX as f32) * max_e.scale_f32()).is_finite());
        assert!(!((((ANCHOR_MAX + 1) as f32) * max_e.scale_f32()).is_finite()));
    }

    #[test]
    fn starter_vector_header_roundtrip() {
        let mut block = [0u8; BLOCK_BYTES];
        block[0..20].copy_from_slice(&[
            0x06, 0x08, 0x40, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);
        let h = Header::parse(&block).unwrap();
        assert_eq!(h.bits_per_index, 3);
        assert_eq!(h.num_vertices, 3);
        assert_eq!(h.num_triangles, 1);
        assert_eq!(
            h.geom_id_mode,
            GeomIdHeaderMode::Constant {
                geometry_id: 0,
                opaque: true
            }
        );
        assert_eq!(h.exponent.raw(), 127);
        assert_eq!(h.anchor, [0, 0, 0]);
        assert_eq!(h.offset_bits, [2, 1, 1]);
        assert_eq!(h.omm_descriptor_count, 0);
        assert_eq!(h.prim_id_base, 0);
        assert!(!h.have_user_data);

        let mut out = [0u8; BLOCK_BYTES];
        h.write(&mut out).unwrap();
        assert_eq!(&out[0..20], &block[0..20]);
    }

    #[test]
    fn rejects_invalid_header_fields() {
        let mut valid = [0u8; BLOCK_BYTES];
        valid[0..20].copy_from_slice(&[
            0x06, 0x08, 0x40, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ]);

        // Bad magic
        let mut b = valid;
        b[0] = 0x07;
        assert_eq!(
            Header::parse(&b),
            Err(DecodeError::InvalidMagic { found: 0x07 })
        );

        // Nonzero reserved bits
        let mut b = valid;
        b[19] = 0x40;
        assert_eq!(
            Header::parse(&b),
            Err(DecodeError::ReservedBitsNonZero { bits: 1 })
        );

        // Bad width sum: [1, 1, 1] -> sum = 3
        let mut b = valid;
        b[8] = 0x00;
        assert_eq!(
            Header::parse(&b),
            Err(DecodeError::InvalidVertexWidthSum {
                x_bits: 1,
                y_bits: 1,
                z_bits: 1,
                sum: 3
            })
        );

        // Bad geom prefix > 25 in palette mode
        let mut b = valid;
        b[12] |= 0x80; // geom_id_mode = 1
        // Set geom_id_meta (bits 22..31 of w0) to prefix_bits = 26
        let w0 = (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & !(0x3FF << 22)) | (26u32 << 22);
        b[0..4].copy_from_slice(&w0.to_le_bytes());
        assert_eq!(
            Header::parse(&b),
            Err(DecodeError::InvalidGeomPrefixLength { prefix_bits: 26 })
        );

        // Primitive ID overflow: base = MAX_PRIM_ID with 2 triangles
        let mut b = valid;
        let w0 = (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & !(0x3F << 16)) | (1u32 << 16);
        b[0..4].copy_from_slice(&w0.to_le_bytes());
        b[16..20].copy_from_slice(&MAX_PRIM_ID.to_le_bytes());
        assert_eq!(
            Header::parse(&b),
            Err(DecodeError::PrimitiveIdOverflow {
                prim_id_base: MAX_PRIM_ID,
                num_triangles: 2
            })
        );
    }
}
