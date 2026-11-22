//! Allocation-free scalar block decoder and multi-block stream / mesh decoders.

#[cfg(feature = "std")]
use crate::bits::BLOCK_BYTES;
use crate::bits::read_bits;
use crate::block::ValidatedBlock;
use crate::error::DecodeError;
use crate::header::{BiasedExponent, Header, MAX_TRIANGLES, MAX_VERTICES};
use crate::layout::BlockLayout;
use crate::palette::{DecodedOmm, decode_geom_ids, decode_omm};
use crate::topology::{TriControl, decode_strip_indices, expand_strip_to_triangles};

#[cfg(feature = "std")]
use crate::block::{ValidationMode, validate_block_with_mode};
#[cfg(feature = "std")]
use crate::error::StreamDecodeError;

/// Fully decoded single DGF1 block stored in fixed-size arrays with zero heap allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedBlock {
    /// Validated header metadata.
    pub header: Header,
    /// Validated bit layout.
    pub layout: BlockLayout,
    /// Active vertex count (`3..=64`).
    pub num_vertices: u8,
    /// Active triangle count (`1..=64`).
    pub num_triangles: u8,
    /// Signed 24-bit integer position anchor `[x, y, z]`.
    pub anchor: [i32; 3],
    /// Biased position quantization exponent (`1..=232`).
    pub exponent: BiasedExponent,
    /// Unsigned per-vertex integer offsets `[x, y, z]` (`[..num_vertices]`).
    pub offsets: [[u16; 3]; MAX_VERTICES as usize],
    /// Exact integer quantized positions `anchor + offset` (`[..num_vertices]`).
    pub quantized_positions: [[i32; 3]; MAX_VERTICES as usize],
    /// Decoded `f32` vertex positions `quantized * 2^(E - 127)` (`[..num_vertices]`).
    pub positions: [[f32; 3]; MAX_VERTICES as usize],
    /// Decoded local triangle indices (`[..num_triangles]`).
    pub triangles: [[u8; 3]; MAX_TRIANGLES as usize],
    /// Decoded triangle control sequence (`[..num_triangles]`).
    pub controls: [TriControl; MAX_TRIANGLES as usize],
    /// Per-triangle 24-bit geometry IDs (`[..num_triangles]`).
    pub geometry_ids: [u32; MAX_TRIANGLES as usize],
    /// Per-triangle opaque flags (`[..num_triangles]`).
    pub opaque: [bool; MAX_TRIANGLES as usize],
    /// 29-bit primitive ID base.
    pub prim_id_base: u32,
    /// Optional 4-byte user-data word (never interpreted as a pointer).
    pub user_data: Option<u32>,
    /// Decoded OMM descriptor count, per-triangle selectors, and opaque hot-patch bytes.
    pub omm: DecodedOmm,
}

impl DecodedBlock {
    /// Returns the active decoded `f32` positions slice.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions[..usize::from(self.num_vertices)]
    }

    /// Returns the active exact integer quantized positions slice.
    pub fn quantized_positions(&self) -> &[[i32; 3]] {
        &self.quantized_positions[..usize::from(self.num_vertices)]
    }

    /// Returns the active unsigned per-vertex integer offsets slice.
    pub fn offsets(&self) -> &[[u16; 3]] {
        &self.offsets[..usize::from(self.num_vertices)]
    }

    /// Returns the active local triangle index slice.
    pub fn triangles(&self) -> &[[u8; 3]] {
        &self.triangles[..usize::from(self.num_triangles)]
    }

    /// Returns the active triangle control sequence slice.
    pub fn controls(&self) -> &[TriControl] {
        &self.controls[..usize::from(self.num_triangles)]
    }

    /// Returns the active per-triangle geometry ID slice.
    pub fn geometry_ids(&self) -> &[u32] {
        &self.geometry_ids[..usize::from(self.num_triangles)]
    }

    /// Returns the active per-triangle opaque flag slice.
    pub fn opaque_flags(&self) -> &[bool] {
        &self.opaque[..usize::from(self.num_triangles)]
    }

    /// Returns the active per-triangle OMM descriptor index slice (`0..omm.descriptor_count`).
    ///
    /// Empty when `self.omm.descriptor_count == 0`.
    pub fn omm_triangle_descriptors(&self) -> &[u8] {
        if self.omm.descriptor_count == 0 {
            &[]
        } else {
            &self.omm.triangle_descriptors[..usize::from(self.num_triangles)]
        }
    }

    /// Returns the checked 29-bit primitive ID for local triangle `tri_index`.
    pub const fn prim_id(&self, tri_index: u8) -> Option<u32> {
        self.header.prim_id(tri_index)
    }
}

/// Decodes a [`ValidatedBlock`] into an allocation-free [`DecodedBlock`].
pub fn decode_block(block: &ValidatedBlock<'_>) -> Result<DecodedBlock, DecodeError> {
    let bytes = block.bytes();
    let header = *block.header();
    let layout = *block.layout();
    let controls = *block.controls();

    let user_data = if header.have_user_data {
        Some(u32::from_le_bytes([
            bytes[20], bytes[21], bytes[22], bytes[23],
        ]))
    } else {
        None
    };

    let mut offsets = [[0u16; 3]; MAX_VERTICES as usize];
    let mut quantized_positions = [[0i32; 3]; MAX_VERTICES as usize];
    let mut positions = [[0.0f32; 3]; MAX_VERTICES as usize];

    let scale = header.exponent.scale_f32();
    let bx = usize::from(header.offset_bits[0]);
    let by = usize::from(header.offset_bits[1]);
    let bz = usize::from(header.offset_bits[2]);
    let stride = bx + by + bz;

    for v in 0..usize::from(header.num_vertices) {
        let base = layout.front.vertex_start + v * stride;
        let ox = read_bits(bytes, base, bx)? as u16;
        let oy = read_bits(bytes, base + bx, by)? as u16;
        let oz = read_bits(bytes, base + bx + by, bz)? as u16;
        offsets[v] = [ox, oy, oz];

        let qx = header.anchor[0] + i32::from(ox);
        let qy = header.anchor[1] + i32::from(oy);
        let qz = header.anchor[2] + i32::from(oz);
        quantized_positions[v] = [qx, qy, qz];

        let px = (qx as f32) * scale;
        let py = (qy as f32) * scale;
        let pz = (qz as f32) * scale;
        for (axis, (&p, &q)) in [px, py, pz].iter().zip([qx, qy, qz].iter()).enumerate() {
            if !p.is_finite() {
                return Err(DecodeError::NonFinitePosition {
                    vertex: v as u8,
                    axis: axis as u8,
                    quantized: q,
                    exponent: header.exponent.raw(),
                });
            }
        }
        positions[v] = [px, py, pz];
    }

    let strip_indices = decode_strip_indices(bytes, &header, &layout, block.is_first())?;
    let triangles = expand_strip_to_triangles(
        &controls,
        header.num_triangles,
        &strip_indices,
        layout.strip_index_count,
    )?;

    let (geometry_ids, opaque) = decode_geom_ids(bytes, &header, &layout.front)?;
    let omm = decode_omm(bytes, &header, &layout.front, block.validation_mode())?;

    Ok(DecodedBlock {
        header,
        layout,
        num_vertices: header.num_vertices,
        num_triangles: header.num_triangles,
        anchor: header.anchor,
        exponent: header.exponent,
        offsets,
        quantized_positions,
        positions,
        triangles,
        controls,
        geometry_ids,
        opaque,
        prim_id_base: header.prim_id_base,
        user_data,
        omm,
    })
}

/// Output capacity limits for multi-block stream and mesh decoding.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// Maximum number of 128-byte blocks accepted.
    pub max_blocks: usize,
    /// Maximum cumulative decoded vertices across all blocks.
    pub max_vertices: usize,
    /// Maximum cumulative decoded triangles across all blocks.
    pub max_triangles: usize,
    /// Validation mode applied to each block (`Runtime` by default).
    pub validation_mode: ValidationMode,
}

#[cfg(feature = "std")]
impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_blocks: 16_777_216,
            max_vertices: 1_073_741_824,
            max_triangles: 1_073_741_824,
            validation_mode: ValidationMode::Runtime,
        }
    }
}

#[cfg(feature = "std")]
impl DecodeLimits {
    /// Returns limits with no practical cap beyond `usize::MAX`.
    pub const fn unlimited() -> Self {
        Self {
            max_blocks: usize::MAX,
            max_vertices: usize::MAX,
            max_triangles: usize::MAX,
            validation_mode: ValidationMode::Runtime,
        }
    }
}

/// Concatenated triangle mesh produced by [`decode_mesh`].
///
/// Preserves block-local vertex separation without silently welding vertices whose
/// decoded positions happen to match.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedMesh {
    /// Concatenated vertex positions from each block in stream order.
    pub positions: Vec<[f32; 3]>,
    /// Concatenated triangles with indices adjusted by each block's vertex offset.
    pub triangles: Vec<[u32; 3]>,
    /// Per-triangle 24-bit geometry IDs.
    pub geometry_ids: Vec<u32>,
    /// Per-triangle opaque flags.
    pub opaque: Vec<bool>,
    /// Per-triangle 29-bit primitive IDs (`prim_id_base + local_triangle_index`).
    pub prim_ids: Vec<u32>,
    /// Per-block optional 4-byte user-data words.
    pub block_user_data: Vec<Option<u32>>,
    /// Starting index in [`Self::positions`] for each block.
    pub block_vertex_offsets: Vec<usize>,
    /// Starting index in [`Self::triangles`] for each block.
    pub block_triangle_offsets: Vec<usize>,
}

/// Lazily validates and decodes a raw stream of 128-byte DGF1 blocks.
///
/// Rejects a partial final block immediately with [`DecodeError::TruncatedStream`].
/// An empty byte slice yields an empty iterator (valid empty mesh).
#[cfg(feature = "std")]
pub fn decode_blocks(
    bytes: &[u8],
) -> Result<impl Iterator<Item = Result<DecodedBlock, StreamDecodeError>> + '_, StreamDecodeError> {
    decode_blocks_with_mode(bytes, ValidationMode::Runtime)
}

/// Lazily validates and decodes a raw stream of 128-byte DGF1 blocks with an explicit [`ValidationMode`].
#[cfg(feature = "std")]
pub fn decode_blocks_with_mode(
    bytes: &[u8],
    mode: ValidationMode,
) -> Result<impl Iterator<Item = Result<DecodedBlock, StreamDecodeError>> + '_, StreamDecodeError> {
    if !bytes.len().is_multiple_of(BLOCK_BYTES) {
        return Err(StreamDecodeError::stream(DecodeError::TruncatedStream {
            byte_len: bytes.len(),
        }));
    }

    Ok(bytes
        .chunks_exact(BLOCK_BYTES)
        .enumerate()
        .map(move |(idx, chunk)| {
            let arr: &[u8; BLOCK_BYTES] = chunk
                .try_into()
                .expect("chunks_exact(128) always yields 128-byte slices");
            let validated = validate_block_with_mode(arr, mode)
                .map_err(|e| StreamDecodeError::at_block(idx, e))?;
            decode_block(&validated).map_err(|e| StreamDecodeError::at_block(idx, e))
        }))
}

/// Decodes a raw stream of 128-byte DGF1 blocks into a `Vec<DecodedBlock>`, enforcing [`DecodeLimits`].
#[cfg(feature = "std")]
pub fn decode_stream(
    bytes: &[u8],
    limits: &DecodeLimits,
) -> Result<Vec<DecodedBlock>, StreamDecodeError> {
    if !bytes.len().is_multiple_of(BLOCK_BYTES) {
        return Err(StreamDecodeError::stream(DecodeError::TruncatedStream {
            byte_len: bytes.len(),
        }));
    }
    let num_blocks = bytes.len() / BLOCK_BYTES;
    if num_blocks > limits.max_blocks {
        return Err(StreamDecodeError::stream(DecodeError::BlockLimitExceeded {
            count: num_blocks,
            limit: limits.max_blocks,
        }));
    }

    let mut out = Vec::with_capacity(num_blocks);
    let mut total_verts = 0usize;
    let mut total_tris = 0usize;

    for (idx, item) in decode_blocks_with_mode(bytes, limits.validation_mode)?.enumerate() {
        let block = item?;
        total_verts = total_verts
            .checked_add(usize::from(block.num_vertices))
            .ok_or_else(|| {
                StreamDecodeError::at_block(
                    idx,
                    DecodeError::ArithmeticOverflow {
                        context: "cumulative decoded vertices",
                    },
                )
            })?;
        if total_verts > limits.max_vertices {
            return Err(StreamDecodeError::at_block(
                idx,
                DecodeError::VertexLimitExceeded {
                    count: total_verts,
                    limit: limits.max_vertices,
                },
            ));
        }

        total_tris = total_tris
            .checked_add(usize::from(block.num_triangles))
            .ok_or_else(|| {
                StreamDecodeError::at_block(
                    idx,
                    DecodeError::ArithmeticOverflow {
                        context: "cumulative decoded triangles",
                    },
                )
            })?;
        if total_tris > limits.max_triangles {
            return Err(StreamDecodeError::at_block(
                idx,
                DecodeError::TriangleLimitExceeded {
                    count: total_tris,
                    limit: limits.max_triangles,
                },
            ));
        }

        out.push(block);
    }

    Ok(out)
}

/// Decodes a raw stream of 128-byte DGF1 blocks into a concatenated [`DecodedMesh`].
///
/// Preserves per-block vertex separation and adjusts triangle indices by each block's
/// starting vertex offset. Never merges vertices by position equality.
#[cfg(feature = "std")]
pub fn decode_mesh(bytes: &[u8], limits: &DecodeLimits) -> Result<DecodedMesh, StreamDecodeError> {
    if !bytes.len().is_multiple_of(BLOCK_BYTES) {
        return Err(StreamDecodeError::stream(DecodeError::TruncatedStream {
            byte_len: bytes.len(),
        }));
    }
    let num_blocks = bytes.len() / BLOCK_BYTES;
    if num_blocks > limits.max_blocks {
        return Err(StreamDecodeError::stream(DecodeError::BlockLimitExceeded {
            count: num_blocks,
            limit: limits.max_blocks,
        }));
    }

    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    let mut geometry_ids = Vec::new();
    let mut opaque = Vec::new();
    let mut prim_ids = Vec::new();
    let mut block_user_data = Vec::with_capacity(num_blocks);
    let mut block_vertex_offsets = Vec::with_capacity(num_blocks);
    let mut block_triangle_offsets = Vec::with_capacity(num_blocks);

    for (idx, item) in decode_blocks_with_mode(bytes, limits.validation_mode)?.enumerate() {
        let block = item?;
        let vert_offset = positions.len();
        let tri_offset = triangles.len();

        let new_verts = vert_offset
            .checked_add(usize::from(block.num_vertices))
            .ok_or_else(|| {
                StreamDecodeError::at_block(
                    idx,
                    DecodeError::ArithmeticOverflow {
                        context: "cumulative mesh vertices",
                    },
                )
            })?;
        if new_verts > limits.max_vertices || new_verts > (u32::MAX as usize) {
            return Err(StreamDecodeError::at_block(
                idx,
                DecodeError::VertexLimitExceeded {
                    count: new_verts,
                    limit: limits.max_vertices.min(u32::MAX as usize),
                },
            ));
        }

        let new_tris = tri_offset
            .checked_add(usize::from(block.num_triangles))
            .ok_or_else(|| {
                StreamDecodeError::at_block(
                    idx,
                    DecodeError::ArithmeticOverflow {
                        context: "cumulative mesh triangles",
                    },
                )
            })?;
        if new_tris > limits.max_triangles {
            return Err(StreamDecodeError::at_block(
                idx,
                DecodeError::TriangleLimitExceeded {
                    count: new_tris,
                    limit: limits.max_triangles,
                },
            ));
        }

        block_vertex_offsets.push(vert_offset);
        block_triangle_offsets.push(tri_offset);
        block_user_data.push(block.user_data);

        positions.extend_from_slice(block.positions());
        let base = vert_offset as u32;
        for (local_t, &[a, b, c]) in block.triangles().iter().enumerate() {
            triangles.push([
                base + u32::from(a),
                base + u32::from(b),
                base + u32::from(c),
            ]);
            geometry_ids.push(block.geometry_ids[local_t]);
            opaque.push(block.opaque[local_t]);
            prim_ids.push(block.prim_id_base + local_t as u32);
        }
    }

    Ok(DecodedMesh {
        positions,
        triangles,
        geometry_ids,
        opaque,
        prim_ids,
        block_user_data,
        block_vertex_offsets,
        block_triangle_offsets,
    })
}
