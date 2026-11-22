//! Low-level DGF1 prepared-block planner and deterministic 128-byte serializer.

use crate::bits::{BLOCK_BITS, write_bits};
use crate::block::Block;
use crate::error::{EncodeError, FitFailure};
use crate::header::{
    ANCHOR_MAX, ANCHOR_MIN, BiasedExponent, GeomIdHeaderMode, Header, MAX_BIASED_EXPONENT,
    MAX_BITS_PER_INDEX, MAX_CONSTANT_GEOM_ID, MAX_GEOM_ID, MAX_GEOM_PALETTE_ENTRIES,
    MAX_OMM_DESCRIPTORS, MAX_PRIM_ID, MAX_TRIANGLES, MAX_VERTICES, MIN_BITS_PER_INDEX,
};
use crate::layout::{
    BlockLayout, FrontLayout, MAX_FRONT_BUFFER_BITS, MAX_REPEAT_BUFFER_BITS, MAX_STRIP_INDICES,
};
use crate::topology::TriControl;

/// Maximum unsigned 16-bit vertex position offset (`65535`).
pub const MAX_VERTEX_OFFSET: u64 = u16::MAX as u64;

/// Validated first-use-ordered topology ready for block planning and serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedTopology {
    /// Per-triangle controls (`1..=64`, `controls[0] == TriControl::Restart`).
    pub controls: Vec<TriControl>,
    /// Strip vertex indices in first-use local vertex order (`3..=192`).
    pub strip_indices: Vec<u8>,
    /// Number of unique local vertices referenced (`3..=64`).
    pub num_vertices: u8,
}

impl PreparedTopology {
    /// Validates internal strip-topology invariants and returns `(is_first_flags, repeated_indices, max_repeated)`.
    pub fn analyze(&self) -> Result<(Vec<bool>, Vec<u8>, u8), EncodeError> {
        let num_tris = self.controls.len();
        if num_tris == 0 || self.num_vertices < 3 {
            return Err(EncodeError::EmptyBlock);
        }
        if num_tris > usize::from(MAX_TRIANGLES) {
            return Err(EncodeError::DoesNotFit(FitFailure::TooManyTriangles {
                count: num_tris,
                max: usize::from(MAX_TRIANGLES),
            }));
        }
        if self.num_vertices > MAX_VERTICES {
            return Err(EncodeError::DoesNotFit(FitFailure::TooManyVertices {
                count: usize::from(self.num_vertices),
                max: usize::from(MAX_VERTICES),
            }));
        }
        if self.controls[0] != TriControl::Restart {
            return Err(EncodeError::InvalidTopology(
                "triangle 0 control must be RESTART",
            ));
        }

        let mut expected_strip_len = 3usize;
        for i in 1..num_tris {
            let c = self.controls[i];
            if c == TriControl::Backtrack
                && (i < 2 || !matches!(self.controls[i - 1], TriControl::Edge1 | TriControl::Edge2))
            {
                return Err(EncodeError::InvalidTopology(
                    "illegal BACKTRACK not preceded by EDGE1 or EDGE2",
                ));
            }
            expected_strip_len += c.index_count();
        }
        if expected_strip_len > MAX_STRIP_INDICES {
            return Err(EncodeError::InvalidTopology(
                "strip index count exceeds 192",
            ));
        }
        if self.strip_indices.len() != expected_strip_len {
            return Err(EncodeError::InvalidTopology(
                "strip_indices length does not match controls",
            ));
        }
        if self.strip_indices[0] != 0 || self.strip_indices[1] != 1 || self.strip_indices[2] != 2 {
            return Err(EncodeError::InvalidTopology(
                "first three strip indices must be [0, 1, 2] in first-use order",
            ));
        }

        let mut is_first = vec![false; expected_strip_len];
        is_first[0] = true;
        is_first[1] = true;
        is_first[2] = true;

        let mut next_new = 3u8;
        let mut repeated = Vec::new();
        let mut max_repeated = 0u8;

        for (j, &idx) in self.strip_indices.iter().enumerate().skip(3) {
            if idx == next_new {
                if next_new >= self.num_vertices {
                    return Err(EncodeError::InvalidTopology(
                        "strip introduces more vertices than num_vertices",
                    ));
                }
                is_first[j] = true;
                next_new += 1;
            } else if idx < next_new {
                repeated.push(idx);
                if idx > max_repeated {
                    max_repeated = idx;
                }
            } else {
                return Err(EncodeError::InvalidTopology(
                    "strip index is not in first-use order (skips next_new vertex)",
                ));
            }
        }

        Ok((is_first, repeated, max_repeated))
    }
}

/// Optional OMM palette payload for a [`PreparedBlock`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedOmm {
    /// Number of distinct OMM descriptors (`0..=7`).
    pub descriptor_count: u8,
    /// Per-triangle local descriptor selector (`0..descriptor_count`). Empty when `descriptor_count == 0`.
    pub triangle_descriptors: Vec<u8>,
    /// Optional explicit hot-patch bytes (`4 * (2 + descriptor_count)` bytes).
    /// When `None`, the encoder zero-initializes the hot-patch region for fresh assets.
    pub hot_patch_bytes: Option<Vec<u8>>,
}

/// Complete quantized and topology-ordered input to the low-level block planner and serializer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedBlock {
    /// Quantized integer positions in first-use local vertex order (`len == topology.num_vertices`).
    pub quantized_positions: Vec<[i32; 3]>,
    /// Biased quantization exponent (`1..=232`).
    pub exponent: BiasedExponent,
    /// First-use-ordered strip topology.
    pub topology: PreparedTopology,
    /// Per-triangle 24-bit geometry IDs (`len == topology.controls.len()`).
    pub geometry_ids: Vec<u32>,
    /// Per-triangle opaque flags (`len == topology.controls.len()`).
    pub opaque: Vec<bool>,
    /// Starting 29-bit primitive ID for triangle 0.
    pub prim_id_base: u32,
    /// Optional 4-byte user-data word.
    pub user_data: Option<u32>,
    /// Optional OMM descriptor assignments.
    pub omm: Option<PreparedOmm>,
    /// Whether to apply lossless power-of-two trailing-zero exponent adjustment to coordinates.
    pub allow_exponent_adjustment: bool,
}

/// Validated, exact-sized plan produced by [`plan_block`] prior to writing any bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedBlock {
    /// Computed semantic DGF1 header.
    pub header: Header,
    /// Computed exact bit layout.
    pub layout: BlockLayout,
    /// Adjusted integer offsets `[x, y, z]` relative to `header.anchor`.
    pub offsets: Vec<[u16; 3]>,
    /// First-use flags (`len == layout.strip_index_count`).
    pub is_first: Vec<bool>,
    /// Forward repeated local vertex indices (`len == layout.repeated_index_count`).
    pub repeated_indices: Vec<u8>,
    /// Geometry palette prefix (0 in constant mode).
    pub geom_prefix: u32,
    /// Per-triangle geometry palette selectors (empty in constant mode).
    pub geom_selectors: Vec<u8>,
    /// Geometry palette entry payloads (empty in constant mode).
    pub geom_payloads: Vec<u32>,
    /// Per-triangle OMM selectors (empty when `descriptor_count == 0`).
    pub omm_selectors: Vec<u8>,
    /// Optional explicit hot-patch bytes (`None` -> zero-initialized).
    pub omm_hot_patch: Option<Vec<u8>>,
    /// Optional 4-byte user-data word.
    pub user_data: Option<u32>,
}

/// Computes the minimum unsigned bit width needed to represent `v` (`0 -> 0`).
pub const fn unsigned_bit_length(v: u64) -> u8 {
    if v == 0 {
        0
    } else {
        (u64::BITS - v.leading_zeros()) as u8
    }
}

/// Validates `input` and computes its exact DGF1 header, palettes, widths, and section budgets.
///
/// Returns [`EncodeError::DoesNotFit`] for valid candidates that exceed a DGF1 block budget,
/// and other [`EncodeError`] variants for invalid inputs.
pub fn plan_block(input: &PreparedBlock) -> Result<PlannedBlock, EncodeError> {
    let num_tris = input.topology.controls.len();
    let num_verts = usize::from(input.topology.num_vertices);

    if num_tris == 0 || num_verts < 3 {
        return Err(EncodeError::EmptyBlock);
    }
    if input.quantized_positions.len() != num_verts {
        return Err(EncodeError::InvalidTopology(
            "quantized_positions length does not match topology.num_vertices",
        ));
    }
    if input.geometry_ids.len() != num_tris || input.opaque.len() != num_tris {
        return Err(EncodeError::InvalidTopology(
            "geometry_ids or opaque length does not match triangle count",
        ));
    }

    let last_prim = u64::from(input.prim_id_base) + (num_tris as u64 - 1);
    if input.prim_id_base > MAX_PRIM_ID || last_prim > u64::from(MAX_PRIM_ID) {
        return Err(EncodeError::PrimitiveIdRangeExceeded {
            base: input.prim_id_base,
            total_triangles: num_tris,
        });
    }

    for (t, &gid) in input.geometry_ids.iter().enumerate() {
        if gid > MAX_GEOM_ID {
            return Err(EncodeError::InvalidGeometryId {
                triangle: t,
                geometry_id: gid,
            });
        }
    }

    // Validate topology first (also checks triangle/vertex limits).
    let (is_first, repeated_indices, max_repeated) = input.topology.analyze()?;

    // Determine repeated-index bit width from actually repeated indices only.
    let bits_per_index = if repeated_indices.is_empty() {
        MIN_BITS_PER_INDEX
    } else {
        unsigned_bit_length(u64::from(max_repeated)).clamp(MIN_BITS_PER_INDEX, MAX_BITS_PER_INDEX)
    };

    // Optional lossless trailing-zero exponent adjustment.
    let (exponent, adjusted_positions) = if input.allow_exponent_adjustment {
        adjust_trailing_zeros(input.exponent, &input.quantized_positions)
    } else {
        (input.exponent, input.quantized_positions.clone())
    };

    // Compute componentwise min anchor and max offset per axis.
    let mut min_q = [i32::MAX; 3];
    let mut max_q = [i32::MIN; 3];
    for pos in &adjusted_positions {
        for a in 0..3 {
            if pos[a] < min_q[a] {
                min_q[a] = pos[a];
            }
            if pos[a] > max_q[a] {
                max_q[a] = pos[a];
            }
        }
    }

    for (axis, &anc) in min_q.iter().enumerate() {
        if !(ANCHOR_MIN..=ANCHOR_MAX).contains(&anc) {
            return Err(EncodeError::DoesNotFit(FitFailure::AnchorOutOfRange {
                axis: axis as u8,
                anchor: anc,
            }));
        }
    }

    let mut offset_bits = [1u8; 3];
    for a in 0..3 {
        let span = (i64::from(max_q[a]) - i64::from(min_q[a])) as u64;
        if span > MAX_VERTEX_OFFSET {
            return Err(EncodeError::DoesNotFit(FitFailure::PositionSpanTooLarge {
                axis: a as u8,
                span,
                max: MAX_VERTEX_OFFSET,
            }));
        }
        offset_bits[a] = unsigned_bit_length(span).max(1);
    }

    // Deterministically round width sum up to a multiple of 4 (axis order: X -> Y -> Z).
    pad_offset_widths_to_multiple_of_4(&mut offset_bits);

    // Build unsigned offsets and verify decoded f32 finiteness (catches E=232 overflow).
    let scale = exponent.scale_f32();
    let mut offsets = Vec::with_capacity(num_verts);
    for (v, pos) in adjusted_positions.iter().enumerate() {
        let ox = (i64::from(pos[0]) - i64::from(min_q[0])) as u16;
        let oy = (i64::from(pos[1]) - i64::from(min_q[1])) as u16;
        let oz = (i64::from(pos[2]) - i64::from(min_q[2])) as u16;
        for (axis, &q) in pos.iter().enumerate() {
            if !((q as f32) * scale).is_finite() {
                return Err(EncodeError::QuantizedNonFinite {
                    vertex: v,
                    axis: axis as u8,
                    exponent: exponent.unbiased().value(),
                });
            }
        }
        offsets.push([ox, oy, oz]);
    }

    // Build geometry/opacity mode and palette (deduplicating (geometry_id, opaque) pairs).
    let (geom_id_mode, geom_prefix, geom_selectors, geom_payloads) =
        build_geom_palette(&input.geometry_ids, &input.opaque)?;

    // Validate OMM descriptors if present.
    let (omm_descriptor_count, omm_selectors, omm_hot_patch) = match &input.omm {
        None => (0u8, Vec::new(), None),
        Some(omm) => {
            if omm.descriptor_count == 0 {
                if !omm.triangle_descriptors.is_empty() {
                    return Err(EncodeError::InvalidTopology(
                        "OMM triangle_descriptors must be empty when descriptor_count == 0",
                    ));
                }
                if let Some(hp) = &omm.hot_patch_bytes
                    && !hp.is_empty()
                {
                    return Err(EncodeError::InvalidTopology(
                        "OMM hot_patch_bytes must be empty when descriptor_count == 0",
                    ));
                }
                (0u8, Vec::new(), None)
            } else {
                if omm.descriptor_count > MAX_OMM_DESCRIPTORS {
                    return Err(EncodeError::DoesNotFit(FitFailure::TooManyOmmDescriptors {
                        count: usize::from(omm.descriptor_count),
                        max: usize::from(MAX_OMM_DESCRIPTORS),
                    }));
                }
                if omm.triangle_descriptors.len() != num_tris {
                    return Err(EncodeError::InvalidTopology(
                        "OMM triangle_descriptors length does not match triangle count",
                    ));
                }
                for &sel in &omm.triangle_descriptors {
                    if sel >= omm.descriptor_count {
                        return Err(EncodeError::InvalidTopology(
                            "OMM selector exceeds descriptor_count",
                        ));
                    }
                }
                let expected_hp_len = 4 * (2 + usize::from(omm.descriptor_count));
                if let Some(hp) = &omm.hot_patch_bytes
                    && hp.len() != expected_hp_len
                {
                    return Err(EncodeError::InvalidTopology(
                        "OMM hot_patch_bytes length does not equal 4 * (2 + descriptor_count)",
                    ));
                }
                (
                    omm.descriptor_count,
                    omm.triangle_descriptors.clone(),
                    omm.hot_patch_bytes.clone(),
                )
            }
        }
    };

    let header = Header {
        bits_per_index,
        num_vertices: input.topology.num_vertices,
        num_triangles: num_tris as u8,
        geom_id_mode,
        exponent,
        anchor: min_q,
        offset_bits,
        omm_descriptor_count,
        prim_id_base: input.prim_id_base,
        have_user_data: input.user_data.is_some(),
    };

    let front = FrontLayout::compute(&header).map_err(|e| match e {
        crate::error::DecodeError::FrontBufferOverflow { front_bits, .. } => {
            EncodeError::DoesNotFit(FitFailure::FrontBufferOverflow {
                bits: front_bits,
                max_bits: MAX_FRONT_BUFFER_BITS,
            })
        }
        other => EncodeError::InternalValidation(other),
    })?;

    let layout = BlockLayout::compute(
        front,
        &header,
        input.topology.strip_indices.len(),
        repeated_indices.len(),
    )
    .map_err(|e| match e {
        crate::error::DecodeError::RepeatBufferOverflow { repeat_bits, .. } => {
            EncodeError::DoesNotFit(FitFailure::RepeatBufferOverflow {
                bits: repeat_bits,
                max_bits: MAX_REPEAT_BUFFER_BITS,
            })
        }
        crate::error::DecodeError::SectionOverlap {
            repeat_end,
            tail_start,
        } => EncodeError::DoesNotFit(FitFailure::BlockCapacityOverflow {
            repeat_end,
            tail_start,
        }),
        other => EncodeError::InternalValidation(other),
    })?;

    Ok(PlannedBlock {
        header,
        layout,
        offsets,
        is_first,
        repeated_indices,
        geom_prefix,
        geom_selectors,
        geom_payloads,
        omm_selectors,
        omm_hot_patch,
        user_data: input.user_data,
    })
}

/// Serializes a validated [`PreparedBlock`] into a deterministic zero-padded 128-byte [`Block`].
pub fn encode_block(input: &PreparedBlock) -> Result<Block, EncodeError> {
    let plan = plan_block(input)?;
    serialize_planned_block(&plan, &input.topology.controls)
}

/// Serializes an already-planned [`PlannedBlock`] into a deterministic 128-byte [`Block`].
pub fn serialize_planned_block(
    plan: &PlannedBlock,
    controls: &[TriControl],
) -> Result<Block, EncodeError> {
    let mut block = Block::zeroed();
    let bytes = block.as_bytes_mut();

    // 1. Header (bytes 0..20)
    plan.header
        .write(bytes)
        .map_err(EncodeError::InternalValidation)?;

    // 2. Optional 4-byte user data (bytes 20..24)
    if let Some(ud) = plan.user_data {
        bytes[20..24].copy_from_slice(&ud.to_le_bytes());
    }

    // 3. Vertex offsets
    let bx = usize::from(plan.header.offset_bits[0]);
    let by = usize::from(plan.header.offset_bits[1]);
    let bz = usize::from(plan.header.offset_bits[2]);
    let stride = bx + by + bz;
    for (v, &[ox, oy, oz]) in plan.offsets.iter().enumerate() {
        let base = plan.layout.front.vertex_start + v * stride;
        write_bits(bytes, base, bx, u64::from(ox)).map_err(|_| {
            EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                start: base,
                width: bx,
            })
        })?;
        write_bits(bytes, base + bx, by, u64::from(oy)).map_err(|_| {
            EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                start: base + bx,
                width: by,
            })
        })?;
        write_bits(bytes, base + bx + by, bz, u64::from(oz)).map_err(|_| {
            EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                start: base + bx + by,
                width: bz,
            })
        })?;
    }

    // 4. Optional OMM palette
    if plan.header.omm_descriptor_count > 0 {
        if let Some(hp) = &plan.omm_hot_patch {
            let start_byte = plan.layout.front.omm_start / 8;
            bytes[start_byte..start_byte + hp.len()].copy_from_slice(hp);
        }
        let sel_w = usize::from(plan.layout.front.omm_selector_bits_per_tri);
        if sel_w > 0 {
            for (t, &sel) in plan.omm_selectors.iter().enumerate() {
                let pos = plan.layout.front.omm_selector_start + t * sel_w;
                write_bits(bytes, pos, sel_w, u64::from(sel)).map_err(|_| {
                    EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                        start: pos,
                        width: sel_w,
                    })
                })?;
            }
        }
    }

    // 5. Optional geometry palette
    if let GeomIdHeaderMode::Palette { prefix_bits, .. } = plan.header.geom_id_mode {
        let prefix_w = usize::from(prefix_bits);
        let sel_w = usize::from(plan.layout.front.geom_selector_bits_per_tri);
        let payload_w = usize::from(plan.layout.front.geom_payload_bits_per_entry);
        let start = plan.layout.front.geom_start;

        write_bits(bytes, start, prefix_w, u64::from(plan.geom_prefix)).map_err(|_| {
            EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                start,
                width: prefix_w,
            })
        })?;

        let sel_start = start + prefix_w;
        for (t, &sel) in plan.geom_selectors.iter().enumerate() {
            let pos = sel_start + t * sel_w;
            write_bits(bytes, pos, sel_w, u64::from(sel)).map_err(|_| {
                EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                    start: pos,
                    width: sel_w,
                })
            })?;
        }

        let payload_start = sel_start + plan.geom_selectors.len() * sel_w;
        for (e, &payload) in plan.geom_payloads.iter().enumerate() {
            let pos = payload_start + e * payload_w;
            write_bits(bytes, pos, payload_w, u64::from(payload)).map_err(|_| {
                EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                    start: pos,
                    width: payload_w,
                })
            })?;
        }
    }

    // 6. Forward repeated vertex indices
    let bpi = usize::from(plan.header.bits_per_index);
    for (idx, &v) in plan.repeated_indices.iter().enumerate() {
        let pos = plan.layout.front.repeat_start + idx * bpi;
        write_bits(bytes, pos, bpi, u64::from(v)).map_err(|_| {
            EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                start: pos,
                width: bpi,
            })
        })?;
    }

    // 7. Backward triangle controls (1..T)
    let num_tris = controls.len();
    for (i, &ctrl) in controls.iter().enumerate().skip(1) {
        let pos = BLOCK_BITS - 2 * i;
        write_bits(bytes, pos, 2, u64::from(ctrl as u8)).map_err(|_| {
            EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                start: pos,
                width: 2,
            })
        })?;
    }

    // 8. Backward first-use flags (3..strip_index_count)
    let control_bits = 2 * (num_tris - 1);
    for (j, &flag) in plan.is_first.iter().enumerate().skip(3) {
        let pos = BLOCK_BITS - control_bits - 1 - (j - 3);
        write_bits(bytes, pos, 1, u64::from(flag)).map_err(|_| {
            EncodeError::InternalValidation(crate::error::DecodeError::BitOutOfBounds {
                start: pos,
                width: 1,
            })
        })?;
    }

    Ok(block)
}

/// Deterministically pads `[x_bits, y_bits, z_bits]` upward until their sum is a multiple of 4,
/// respecting the 16-bit per-axis ceiling (order: X -> Y -> Z).
fn pad_offset_widths_to_multiple_of_4(widths: &mut [u8; 3]) {
    let rem = (widths[0] + widths[1] + widths[2]) % 4;
    if rem == 0 {
        return;
    }
    let mut need = 4 - rem;
    let mut axis = 0usize;
    while need > 0 {
        if widths[axis] < 16 {
            widths[axis] += 1;
            need -= 1;
        }
        axis = (axis + 1) % 3;
    }
}

/// Losslessly shifts common power-of-two trailing zeros from absolute quantized coordinates
/// into the block exponent, preserving every decoded `f32` position bit-for-bit.
fn adjust_trailing_zeros(
    exponent: BiasedExponent,
    positions: &[[i32; 3]],
) -> (BiasedExponent, Vec<[i32; 3]>) {
    let mut combined_bits = 0u32;
    for pos in positions {
        for &q in pos {
            combined_bits |= q.unsigned_abs();
        }
    }
    // Explicitly handle all-zero coordinates so trailing_zeros(0) never shifts by word size.
    if combined_bits == 0 {
        return (exponent, positions.to_vec());
    }

    let max_shift_from_bits = combined_bits.trailing_zeros();
    let max_shift_from_exp = u32::from(MAX_BIASED_EXPONENT - exponent.raw());
    let shift = max_shift_from_bits.min(max_shift_from_exp);
    if shift == 0 {
        return (exponent, positions.to_vec());
    }

    let new_exp = BiasedExponent::new(exponent.raw() + shift as u8).unwrap_or(exponent);
    let old_scale = exponent.scale_f32();
    let new_scale = new_exp.scale_f32();

    let mut candidate = Vec::with_capacity(positions.len());
    for pos in positions {
        let mut shifted = [0i32; 3];
        for a in 0..3 {
            // Exact arithmetic right shift because low `shift` bits of |q| are zero.
            let sq = pos[a] >> shift;
            if !(ANCHOR_MIN..=ANCHOR_MAX).contains(&sq) {
                return (exponent, positions.to_vec());
            }
            let old_f = (pos[a] as f32) * old_scale;
            let new_f = (sq as f32) * new_scale;
            if !new_f.is_finite() || old_f.to_bits() != new_f.to_bits() {
                return (exponent, positions.to_vec());
            }
            shifted[a] = sq;
        }
        candidate.push(shifted);
    }

    (new_exp, candidate)
}

#[allow(clippy::type_complexity)]
fn build_geom_palette(
    geometry_ids: &[u32],
    opaque: &[bool],
) -> Result<(GeomIdHeaderMode, u32, Vec<u8>, Vec<u32>), EncodeError> {
    let mut entries: Vec<u32> = Vec::new();
    let mut selectors: Vec<u8> = Vec::with_capacity(geometry_ids.len());

    for (&gid, &op) in geometry_ids.iter().zip(opaque.iter()) {
        let combined = (gid << 1) | u32::from(op);
        if let Some(pos) = entries.iter().position(|&e| e == combined) {
            selectors.push(pos as u8);
        } else {
            let pos = entries.len();
            if pos >= usize::from(MAX_GEOM_PALETTE_ENTRIES) {
                return Err(EncodeError::DoesNotFit(
                    FitFailure::TooManyGeomPaletteEntries {
                        count: pos + 1,
                        max: usize::from(MAX_GEOM_PALETTE_ENTRIES),
                    },
                ));
            }
            entries.push(combined);
            selectors.push(pos as u8);
        }
    }

    if entries.len() == 1 {
        let single = entries[0];
        let gid = single >> 1;
        let op = (single & 1) != 0;
        if gid <= MAX_CONSTANT_GEOM_ID {
            return Ok((
                GeomIdHeaderMode::Constant {
                    geometry_id: gid as u16,
                    opaque: op,
                },
                0,
                Vec::new(),
                Vec::new(),
            ));
        }
        // Single entry with geometry_id >= 512 uses 1-entry palette with 25-bit prefix (0 selector & payload bits).
        return Ok((
            GeomIdHeaderMode::Palette {
                prefix_bits: 25,
                entry_count: 1,
            },
            single,
            vec![0u8; geometry_ids.len()],
            vec![0u32; 1],
        ));
    }

    // Multiple distinct (geometry_id, opaque) pairs: compute common high-bit prefix over 25 bits.
    let first = entries[0];
    let mut diff = 0u32;
    for &e in &entries[1..] {
        diff |= e ^ first;
    }
    diff &= (1u32 << 25) - 1;

    let payload_bits: u8 = if diff == 0 {
        0
    } else {
        (32 - diff.leading_zeros()) as u8
    };
    let prefix_bits = 25 - payload_bits;
    let prefix = if payload_bits == 25 {
        0
    } else {
        first >> payload_bits
    };
    let payload_mask = if payload_bits == 0 {
        0
    } else {
        (1u32 << payload_bits) - 1
    };
    let payloads: Vec<u32> = entries.iter().map(|&e| e & payload_mask).collect();

    Ok((
        GeomIdHeaderMode::Palette {
            prefix_bits,
            entry_count: entries.len() as u8,
        },
        prefix,
        selectors,
        payloads,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{ValidationMode, validate_block_with_mode};
    use crate::decode::decode_block;

    #[test]
    fn starter_vector_exact_128_byte_match() {
        let prepared = PreparedBlock {
            quantized_positions: vec![[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            exponent: BiasedExponent::new(127).unwrap(),
            topology: PreparedTopology {
                controls: vec![TriControl::Restart],
                strip_indices: vec![0, 1, 2],
                num_vertices: 3,
            },
            geometry_ids: vec![0],
            opaque: vec![true],
            prim_id_base: 0,
            user_data: None,
            omm: None,
            allow_exponent_adjustment: false,
        };

        let block = encode_block(&prepared).unwrap();
        let mut expected = [0u8; 128];
        expected[0..22].copy_from_slice(&[
            0x06, 0x08, 0x40, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x04,
        ]);
        assert_eq!(block.as_bytes(), &expected);

        let validated = validate_block_with_mode(block.as_bytes(), ValidationMode::Asset).unwrap();
        let decoded = decode_block(&validated).unwrap();
        assert_eq!(
            decoded.quantized_positions(),
            &[[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        );
        assert_eq!(decoded.triangles(), &[[0, 1, 2]]);
        assert_eq!(decoded.geometry_ids(), &[0]);
        assert_eq!(decoded.opaque_flags(), &[true]);
    }

    #[test]
    fn trailing_zero_adjustment_handles_all_zeros_safely() {
        let exp = BiasedExponent::new(127).unwrap();
        let zeros = vec![[0, 0, 0], [0, 0, 0], [0, 0, 0]];
        let (new_exp, new_pos) = adjust_trailing_zeros(exp, &zeros);
        assert_eq!(new_exp, exp);
        assert_eq!(new_pos, zeros);

        let even = vec![[0, 0, 0], [8, 0, 0], [0, 16, 0]];
        let (new_exp2, new_pos2) = adjust_trailing_zeros(exp, &even);
        assert_eq!(new_exp2.raw(), 130); // shifted by 3 (dividing by 8)
        assert_eq!(new_pos2, vec![[0, 0, 0], [1, 0, 0], [0, 2, 0]]);
    }

    #[test]
    fn palette_handles_32_entries_and_rejects_33() {
        let gids32: Vec<u32> = (0..32).collect();
        let op32 = vec![true; 32];
        let (mode, _, sels, payloads) = build_geom_palette(&gids32, &op32).unwrap();
        assert!(matches!(
            mode,
            GeomIdHeaderMode::Palette {
                entry_count: 32,
                ..
            }
        ));
        assert_eq!(sels.len(), 32);
        assert_eq!(payloads.len(), 32);

        let gids33: Vec<u32> = (0..33).collect();
        let op33 = vec![true; 33];
        assert_eq!(
            build_geom_palette(&gids33, &op33),
            Err(EncodeError::DoesNotFit(
                FitFailure::TooManyGeomPaletteEntries { count: 33, max: 32 }
            ))
        );
    }

    #[test]
    fn same_geom_id_different_opaque_uses_two_palette_entries() {
        let (mode, prefix_bits_val, sels, payloads) =
            build_geom_palette(&[5, 5], &[false, true]).unwrap();
        assert_eq!(
            mode,
            GeomIdHeaderMode::Palette {
                prefix_bits: 24,
                entry_count: 2
            }
        );
        assert_eq!(prefix_bits_val, 5);
        assert_eq!(sels, vec![0, 1]);
        assert_eq!(payloads, vec![0, 1]);
    }
}
