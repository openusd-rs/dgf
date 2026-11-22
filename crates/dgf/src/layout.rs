//! Central checked DGF1 section layout calculation shared by decoding, encoding, and validation.

use crate::bits::BLOCK_BITS;
use crate::error::DecodeError;
use crate::header::{GeomIdHeaderMode, HEADER_BITS, Header};

/// Maximum size of the front buffer (`vertex + OMM + geometry palette`) in bits (96 bytes).
pub const MAX_FRONT_BUFFER_BITS: usize = 96 * 8;

/// Maximum size of the repeated-index buffer in bits (24 bytes).
pub const MAX_REPEAT_BUFFER_BITS: usize = 24 * 8;

/// Maximum strip index count (64 triangles * 3 indices on all restarts).
pub const MAX_STRIP_INDICES: usize = 64 * 3;

/// Rounds `bits` up to the next multiple of 8 using checked arithmetic.
pub const fn align8(bits: usize) -> Option<usize> {
    match bits.checked_add(7) {
        Some(v) => Some(v & !7),
        None => None,
    }
}

/// Computes `ceil(log2(n))` for positive `n` using integer bit math (`ceil_log2(1) == 0`).
pub const fn ceil_log2(n: usize) -> u8 {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as u8
    }
}

/// Exact bit positions and sizes of the forward-packed front sections derived from a [`Header`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontLayout {
    /// Bit offset of optional user data (always `160`).
    pub user_data_start: usize,
    /// Size of user-data section in bits (`0` or `32`).
    pub user_data_bits: usize,
    /// Bit offset where vertex offsets begin (`160 + user_data_bits`).
    pub vertex_start: usize,
    /// Unpadded bit length of vertex offsets (`V * (bx + by + bz)`).
    pub vertex_data_bits: usize,
    /// Byte-aligned bit length of the vertex offset section.
    pub vertex_bits: usize,
    /// Bit offset where the OMM palette begins (or equals `geom_start` when `D == 0`).
    pub omm_start: usize,
    /// Bit length of the OMM hot-patch region (`0` or `32 * (2 + D)`).
    pub omm_hot_patch_bits: usize,
    /// Bit offset where OMM per-triangle selectors begin.
    pub omm_selector_start: usize,
    /// Bit width of each OMM selector (`0` when `D <= 1`, else `ceil_log2(D)`).
    pub omm_selector_bits_per_tri: u8,
    /// Unpadded bit length of OMM selectors (`T * omm_selector_bits_per_tri`).
    pub omm_selector_data_bits: usize,
    /// Total byte-aligned bit length of the OMM palette section.
    pub omm_bits: usize,
    /// Bit offset where the geometry palette section begins.
    pub geom_start: usize,
    /// Prefix bit width in palette mode (`0` in constant mode).
    pub geom_prefix_bits: u8,
    /// Selector bit width per triangle in palette mode (`0` in constant mode or 1-entry palette).
    pub geom_selector_bits_per_tri: u8,
    /// Payload bit width per entry in palette mode (`0` in constant mode).
    pub geom_payload_bits_per_entry: u8,
    /// Unpadded bit length of geometry palette data.
    pub geom_data_bits: usize,
    /// Byte-aligned bit length of the geometry palette section (`0` in constant mode).
    pub geom_bits: usize,
    /// Total front buffer size (`vertex_bits + omm_bits + geom_bits`, checked `<= 768`).
    pub front_bits: usize,
    /// Bit offset where the forward repeated-index buffer begins (`160 + U + front_bits`).
    pub repeat_start: usize,
}

impl FrontLayout {
    /// Computes and validates the front-section layout for `header`.
    pub fn compute(header: &Header) -> Result<Self, DecodeError> {
        let v = usize::from(header.num_vertices);
        let t = usize::from(header.num_triangles);
        let user_data_start = HEADER_BITS;
        let user_data_bits = if header.have_user_data { 32 } else { 0 };
        let vertex_start =
            user_data_start
                .checked_add(user_data_bits)
                .ok_or(DecodeError::ArithmeticOverflow {
                    context: "vertex_start",
                })?;

        let bits_per_vert = usize::from(header.offset_bits[0])
            + usize::from(header.offset_bits[1])
            + usize::from(header.offset_bits[2]);
        let vertex_data_bits =
            v.checked_mul(bits_per_vert)
                .ok_or(DecodeError::ArithmeticOverflow {
                    context: "vertex_data_bits",
                })?;
        let vertex_bits = align8(vertex_data_bits).ok_or(DecodeError::ArithmeticOverflow {
            context: "vertex_bits",
        })?;

        let omm_start =
            vertex_start
                .checked_add(vertex_bits)
                .ok_or(DecodeError::ArithmeticOverflow {
                    context: "omm_start",
                })?;

        let d = usize::from(header.omm_descriptor_count);
        let (
            omm_hot_patch_bits,
            omm_selector_start,
            omm_selector_bits_per_tri,
            omm_selector_data_bits,
            omm_bits,
        ) =
            if d == 0 {
                (0, omm_start, 0, 0, 0)
            } else {
                let hot_patch_bits =
                    32usize
                        .checked_mul(2 + d)
                        .ok_or(DecodeError::ArithmeticOverflow {
                            context: "omm_hot_patch_bits",
                        })?;
                let sel_start = omm_start.checked_add(hot_patch_bits).ok_or(
                    DecodeError::ArithmeticOverflow {
                        context: "omm_selector_start",
                    },
                )?;
                let sel_width = ceil_log2(d);
                let sel_data_bits = t.checked_mul(usize::from(sel_width)).ok_or(
                    DecodeError::ArithmeticOverflow {
                        context: "omm_selector_data_bits",
                    },
                )?;
                let sel_aligned = align8(sel_data_bits).ok_or(DecodeError::ArithmeticOverflow {
                    context: "omm_selector_aligned",
                })?;
                let total_omm = hot_patch_bits.checked_add(sel_aligned).ok_or(
                    DecodeError::ArithmeticOverflow {
                        context: "omm_bits",
                    },
                )?;
                (
                    hot_patch_bits,
                    sel_start,
                    sel_width,
                    sel_data_bits,
                    total_omm,
                )
            };

        let geom_start =
            omm_start
                .checked_add(omm_bits)
                .ok_or(DecodeError::ArithmeticOverflow {
                    context: "geom_start",
                })?;

        let (
            geom_prefix_bits,
            geom_selector_bits_per_tri,
            geom_payload_bits_per_entry,
            geom_data_bits,
            geom_bits,
        ) = match header.geom_id_mode {
            GeomIdHeaderMode::Constant { .. } => (0, 0, 0, 0, 0),
            GeomIdHeaderMode::Palette {
                prefix_bits,
                entry_count,
            } => {
                let p = usize::from(entry_count);
                let l = usize::from(prefix_bits);
                let sel_width = ceil_log2(p);
                let payload_width = 25u8 - prefix_bits;
                let sel_total = t.checked_mul(usize::from(sel_width)).ok_or(
                    DecodeError::ArithmeticOverflow {
                        context: "geom_selector_bits",
                    },
                )?;
                let payload_total = p.checked_mul(usize::from(payload_width)).ok_or(
                    DecodeError::ArithmeticOverflow {
                        context: "geom_payload_bits",
                    },
                )?;
                let data_bits = l
                    .checked_add(sel_total)
                    .and_then(|x| x.checked_add(payload_total))
                    .ok_or(DecodeError::ArithmeticOverflow {
                        context: "geom_data_bits",
                    })?;
                let aligned = align8(data_bits).ok_or(DecodeError::ArithmeticOverflow {
                    context: "geom_bits",
                })?;
                (prefix_bits, sel_width, payload_width, data_bits, aligned)
            }
        };

        let front_bits = vertex_bits
            .checked_add(omm_bits)
            .and_then(|x| x.checked_add(geom_bits))
            .ok_or(DecodeError::ArithmeticOverflow {
                context: "front_bits",
            })?;

        if front_bits > MAX_FRONT_BUFFER_BITS {
            return Err(DecodeError::FrontBufferOverflow {
                front_bits,
                max_bits: MAX_FRONT_BUFFER_BITS,
            });
        }

        let repeat_start =
            vertex_start
                .checked_add(front_bits)
                .ok_or(DecodeError::ArithmeticOverflow {
                    context: "repeat_start",
                })?;

        Ok(Self {
            user_data_start,
            user_data_bits,
            vertex_start,
            vertex_data_bits,
            vertex_bits,
            omm_start,
            omm_hot_patch_bits,
            omm_selector_start,
            omm_selector_bits_per_tri,
            omm_selector_data_bits,
            omm_bits,
            geom_start,
            geom_prefix_bits,
            geom_selector_bits_per_tri,
            geom_payload_bits_per_entry,
            geom_data_bits,
            geom_bits,
            front_bits,
            repeat_start,
        })
    }
}

/// Complete checked bit layout of a DGF1 block, including backward topology and repeated indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    /// Forward front-section layout (header, user data, vertices, OMM, geometry palette).
    pub front: FrontLayout,
    /// Total entries in the strip index stream (`3..=192`).
    pub strip_index_count: usize,
    /// Bit length of the backward triangle control stream (`2 * (T - 1)`).
    pub control_bits: usize,
    /// Bit length of the backward first-use flag stream (`strip_index_count - 3`).
    pub first_use_bits: usize,
    /// Bit offset where the backward topology tail begins (`1024 - control_bits - first_use_bits`).
    pub tail_start: usize,
    /// Number of repeated vertex indices stored in the forward stream.
    pub repeated_index_count: usize,
    /// Bit length of the repeated-index stream (`repeated_index_count * bits_per_index`, `<= 192`).
    pub repeat_bits: usize,
    /// Bit offset where the repeated-index stream ends (`repeat_start + repeat_bits`, `<= tail_start`).
    pub repeat_end: usize,
}

impl BlockLayout {
    /// Finishes the layout calculation given the measured `strip_index_count` and `repeated_index_count`.
    pub fn compute(
        front: FrontLayout,
        header: &Header,
        strip_index_count: usize,
        repeated_index_count: usize,
    ) -> Result<Self, DecodeError> {
        if !(3..=MAX_STRIP_INDICES).contains(&strip_index_count) {
            return Err(DecodeError::ArithmeticOverflow {
                context: "strip_index_count out of 3..=192",
            });
        }
        let t = usize::from(header.num_triangles);
        let control_bits = 2usize
            .checked_mul(t - 1)
            .ok_or(DecodeError::ArithmeticOverflow {
                context: "control_bits",
            })?;
        let first_use_bits = strip_index_count - 3;
        let tail_bits =
            control_bits
                .checked_add(first_use_bits)
                .ok_or(DecodeError::ArithmeticOverflow {
                    context: "tail_bits",
                })?;
        let tail_start = BLOCK_BITS
            .checked_sub(tail_bits)
            .ok_or(DecodeError::SectionOverlap {
                repeat_end: front.repeat_start,
                tail_start: 0,
            })?;

        let repeat_bits = repeated_index_count
            .checked_mul(usize::from(header.bits_per_index))
            .ok_or(DecodeError::ArithmeticOverflow {
                context: "repeat_bits",
            })?;

        if repeat_bits > MAX_REPEAT_BUFFER_BITS {
            return Err(DecodeError::RepeatBufferOverflow {
                repeat_bits,
                max_bits: MAX_REPEAT_BUFFER_BITS,
            });
        }

        let repeat_end =
            front
                .repeat_start
                .checked_add(repeat_bits)
                .ok_or(DecodeError::ArithmeticOverflow {
                    context: "repeat_end",
                })?;

        if repeat_end > tail_start {
            return Err(DecodeError::SectionOverlap {
                repeat_end,
                tail_start,
            });
        }

        Ok(Self {
            front,
            strip_index_count,
            control_bits,
            first_use_bits,
            tail_start,
            repeated_index_count,
            repeat_bits,
            repeat_end,
        })
    }

    /// Returns the number of unused gap bits between `repeat_end` and `tail_start`.
    pub const fn unused_gap_bits(&self) -> usize {
        self.tail_start - self.repeat_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::BiasedExponent;

    #[test]
    fn ceil_log2_values() {
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(7), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(17), 5);
        assert_eq!(ceil_log2(32), 5);
    }

    #[test]
    fn align8_values() {
        assert_eq!(align8(0), Some(0));
        assert_eq!(align8(1), Some(8));
        assert_eq!(align8(7), Some(8));
        assert_eq!(align8(8), Some(8));
        assert_eq!(align8(9), Some(16));
        assert_eq!(align8(usize::MAX), None);
    }

    #[test]
    fn front_and_repeat_sublimits_enforced_exactly() {
        // 64 vertices * 12 bits = 768 bits -> exact front buffer limit (96 bytes)
        let mut h = Header {
            bits_per_index: 6,
            num_vertices: 64,
            num_triangles: 1,
            geom_id_mode: GeomIdHeaderMode::Constant {
                geometry_id: 0,
                opaque: true,
            },
            exponent: BiasedExponent::new(127).unwrap(),
            anchor: [0, 0, 0],
            offset_bits: [4, 4, 4],
            omm_descriptor_count: 0,
            prim_id_base: 0,
            have_user_data: false,
        };
        let fl = FrontLayout::compute(&h).unwrap();
        assert_eq!(fl.front_bits, 768);
        assert_eq!(fl.repeat_start, 160 + 768);

        // One extra byte (geom palette 1 entry, prefix 25 -> 25 bits -> align8 = 32 bits) overflows front buffer
        h.geom_id_mode = GeomIdHeaderMode::Palette {
            prefix_bits: 25,
            entry_count: 1,
        };
        assert_eq!(
            FrontLayout::compute(&h),
            Err(DecodeError::FrontBufferOverflow {
                front_bits: 800,
                max_bits: 768
            })
        );

        // Test repeat buffer exact limit 192 bits (32 * 6) vs 198 bits (33 * 6)
        h.num_vertices = 3;
        h.offset_bits = [2, 1, 1];
        h.geom_id_mode = GeomIdHeaderMode::Constant {
            geometry_id: 0,
            opaque: true,
        };
        let fl_small = FrontLayout::compute(&h).unwrap();
        assert!(BlockLayout::compute(fl_small, &h, 35, 32).is_ok());
        assert_eq!(
            BlockLayout::compute(fl_small, &h, 36, 33),
            Err(DecodeError::RepeatBufferOverflow {
                repeat_bits: 198,
                max_bits: 192
            })
        );
    }

    #[test]
    fn exact_omm_sizes_all_descriptor_counts() {
        for d in 0u8..=7 {
            let h = Header {
                bits_per_index: 3,
                num_vertices: 3,
                num_triangles: 5,
                geom_id_mode: GeomIdHeaderMode::Constant {
                    geometry_id: 0,
                    opaque: true,
                },
                exponent: BiasedExponent::new(127).unwrap(),
                anchor: [0, 0, 0],
                offset_bits: [2, 1, 1],
                omm_descriptor_count: d,
                prim_id_base: 0,
                have_user_data: true,
            };
            let fl = FrontLayout::compute(&h).unwrap();
            let expected = if d == 0 {
                0
            } else {
                32 * (2 + usize::from(d))
                    + align8(5 * usize::from(ceil_log2(usize::from(d)))).unwrap()
            };
            assert_eq!(fl.omm_bits, expected, "d={d}");
            assert_eq!(fl.geom_start, fl.omm_start + expected, "d={d}");
        }
    }
}
