//! Geometry ID / opacity palette and Opacity Micromap (OMM) palette decoding.

use crate::bits::{BLOCK_BYTES, bits_are_zero, read_bits};
use crate::block::ValidationMode;
use crate::error::{DecodeError, PaddingSection};
use crate::header::{GeomIdHeaderMode, Header, MAX_GEOM_PALETTE_ENTRIES, MAX_TRIANGLES};
use crate::layout::FrontLayout;

/// Maximum size of the OMM hot-patch region in bytes (`4 * (2 + 7) = 36`).
pub const MAX_OMM_HOT_PATCH_BYTES: usize = 4 * (2 + 7);

/// Decoded OMM section data for a single DGF1 block.
///
/// The hot-patch bytes are preserved verbatim as opaque data and are never interpreted
/// as host pointers or driver-specific descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedOmm {
    /// Number of OMM descriptors (`0..=7`).
    pub descriptor_count: u8,
    /// Number of valid bytes in [`Self::hot_patch_bytes`] (`0` when `descriptor_count == 0`,
    /// otherwise `4 * (2 + descriptor_count)`).
    pub hot_patch_len: u8,
    /// Opaque hot-patch bytes (`[..hot_patch_len]`).
    pub hot_patch_bytes: [u8; MAX_OMM_HOT_PATCH_BYTES],
    /// Per-triangle descriptor selector indices (`0..descriptor_count`), active up to `num_triangles`.
    pub triangle_descriptors: [u8; MAX_TRIANGLES as usize],
}

impl DecodedOmm {
    /// Returns an empty OMM record (`descriptor_count == 0`).
    pub const fn empty() -> Self {
        Self {
            descriptor_count: 0,
            hot_patch_len: 0,
            hot_patch_bytes: [0u8; MAX_OMM_HOT_PATCH_BYTES],
            triangle_descriptors: [0u8; MAX_TRIANGLES as usize],
        }
    }

    /// Returns the active slice of opaque hot-patch bytes.
    pub fn hot_patch(&self) -> &[u8] {
        &self.hot_patch_bytes[..usize::from(self.hot_patch_len)]
    }
}

/// Decodes and validates per-triangle geometry IDs and opacity flags.
///
/// Handles both constant mode and palette mode, including zero-width selectors (`entry_count == 1`)
/// and zero-width payloads (`prefix_bits == 25`).
pub fn decode_geom_ids(
    bytes: &[u8; BLOCK_BYTES],
    header: &Header,
    layout: &FrontLayout,
) -> Result<
    (
        [u32; MAX_TRIANGLES as usize],
        [bool; MAX_TRIANGLES as usize],
    ),
    DecodeError,
> {
    let mut geom_ids = [0u32; MAX_TRIANGLES as usize];
    let mut opaque = [false; MAX_TRIANGLES as usize];
    let t = usize::from(header.num_triangles);

    match header.geom_id_mode {
        GeomIdHeaderMode::Constant {
            geometry_id,
            opaque: is_opaque,
        } => {
            for i in 0..t {
                geom_ids[i] = u32::from(geometry_id);
                opaque[i] = is_opaque;
            }
            Ok((geom_ids, opaque))
        }
        GeomIdHeaderMode::Palette {
            prefix_bits,
            entry_count,
        } => {
            let pad_bits = layout.geom_bits - layout.geom_data_bits;
            let pad_start = layout.geom_start + layout.geom_data_bits;
            if pad_bits > 0 && !bits_are_zero(bytes, pad_start, pad_bits)? {
                return Err(DecodeError::PaddingNonZero {
                    section: PaddingSection::GeometryPalette,
                    bit_offset: pad_start,
                });
            }

            let prefix_w = usize::from(prefix_bits);
            let sel_w = usize::from(layout.geom_selector_bits_per_tri);
            let payload_w = usize::from(layout.geom_payload_bits_per_entry);
            let p = usize::from(entry_count);

            let prefix = read_bits(bytes, layout.geom_start, prefix_w)? as u32;
            let selectors_start = layout.geom_start + prefix_w;
            let payloads_start = selectors_start + t * sel_w;

            // Decode palette entries first.
            let mut entries = [0u32; MAX_GEOM_PALETTE_ENTRIES as usize];
            for (e, slot) in entries.iter_mut().enumerate().take(p) {
                let payload = read_bits(bytes, payloads_start + e * payload_w, payload_w)? as u32;
                *slot = if payload_w == 0 {
                    prefix
                } else {
                    (prefix << payload_w) | payload
                };
            }

            // Decode per-triangle selectors and resolve (geometry_id, opaque).
            for i in 0..t {
                let sel = read_bits(bytes, selectors_start + i * sel_w, sel_w)? as u8;
                if sel >= entry_count {
                    return Err(DecodeError::GeomPaletteIndexOutOfRange {
                        triangle: i as u8,
                        selector: sel,
                        entry_count,
                    });
                }
                let combined = entries[usize::from(sel)];
                geom_ids[i] = combined >> 1;
                opaque[i] = (combined & 1) != 0;
            }

            Ok((geom_ids, opaque))
        }
    }
}

/// Decodes and validates the OMM palette section, preserving opaque hot-patch bytes.
///
/// In [`ValidationMode::Asset`], rejects nonzero hot-patch bytes.
/// In [`ValidationMode::Runtime`], preserves any runtime-patched hot-patch bytes.
pub fn decode_omm(
    bytes: &[u8; BLOCK_BYTES],
    header: &Header,
    layout: &FrontLayout,
    mode: ValidationMode,
) -> Result<DecodedOmm, DecodeError> {
    let d = header.omm_descriptor_count;
    if d == 0 {
        return Ok(DecodedOmm::empty());
    }

    let mut out = DecodedOmm::empty();
    out.descriptor_count = d;

    let hot_patch_bytes_len = layout.omm_hot_patch_bits / 8;
    let omm_byte_start = layout.omm_start / 8;
    debug_assert_eq!(layout.omm_start % 8, 0);
    debug_assert_eq!(layout.omm_hot_patch_bits % 8, 0);

    out.hot_patch_len = hot_patch_bytes_len as u8;
    out.hot_patch_bytes[..hot_patch_bytes_len]
        .copy_from_slice(&bytes[omm_byte_start..omm_byte_start + hot_patch_bytes_len]);

    if mode == ValidationMode::Asset {
        for (idx, &b) in out.hot_patch_bytes[..hot_patch_bytes_len]
            .iter()
            .enumerate()
        {
            if b != 0 {
                return Err(DecodeError::OmmHotPatchNonZero {
                    byte_offset: idx,
                    value: b,
                });
            }
        }
    }

    let sel_total_aligned = layout.omm_bits - layout.omm_hot_patch_bits;
    let pad_bits = sel_total_aligned - layout.omm_selector_data_bits;
    let pad_start = layout.omm_selector_start + layout.omm_selector_data_bits;
    if pad_bits > 0 && !bits_are_zero(bytes, pad_start, pad_bits)? {
        return Err(DecodeError::PaddingNonZero {
            section: PaddingSection::OmmSelectors,
            bit_offset: pad_start,
        });
    }

    let sel_w = usize::from(layout.omm_selector_bits_per_tri);
    let t = usize::from(header.num_triangles);
    for i in 0..t {
        let sel = read_bits(bytes, layout.omm_selector_start + i * sel_w, sel_w)? as u8;
        if sel >= d {
            return Err(DecodeError::OmmSelectorOutOfRange {
                triangle: i as u8,
                selector: sel,
                descriptor_count: d,
            });
        }
        out.triangle_descriptors[i] = sel;
    }

    Ok(out)
}
