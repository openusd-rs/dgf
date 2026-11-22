//! Exact-fit SAH subdivision block packer with explicit work stack and full layout verification.

use crate::block::Block;
use crate::encode::block::{
    PlannedBlock, PreparedBlock, PreparedOmm, plan_block, serialize_planned_block,
};
use crate::encode::cluster::{Aabb64, TriSpatial};
use crate::encode::mesh::{FitFailureCounts, TriangleMetadata};
use crate::encode::strip::{
    BuiltStrip, CandidateTriangle, build_optimized_strip, build_restart_only_strip,
};
use crate::error::{EncodeError, FitFailure};
use crate::header::{BiasedExponent, MAX_OMM_DESCRIPTORS, MAX_TRIANGLES};
use crate::remap::{BlockOmmMap, BlockRange, TriangleRemap};
use crate::topology::TriControl;

/// Per-retained-triangle descriptor passed into block packers.
#[derive(Debug, Clone, Copy)]
pub struct RetainedTriangle {
    /// Original source triangle index.
    pub input_triangle: u32,
    /// Source vertex indices `[v0, v1, v2]`.
    pub vertices: [u32; 3],
    /// Geometry ID and opacity flag.
    pub metadata: TriangleMetadata,
    /// Optional caller OMM identity (`None` when triangle has no OMM).
    pub omm_id: Option<u32>,
}

/// Packing strategy parameters for a single cluster.
#[derive(Debug, Clone, Copy)]
pub struct ClusterPackParams {
    /// Mesh-wide biased exponent.
    pub exponent: BiasedExponent,
    /// Optional constant user data to reserve/write on each block.
    pub user_data: Option<u32>,
    /// Whether to apply lossless trailing-zero exponent adjustment per block.
    pub enable_exponent_adjustment: bool,
    /// Whether to use the optimized strip builder (vs `RESTART`-only).
    pub use_strips: bool,
    /// Optional cluster ID recorded on [`BlockRange::input_cluster`].
    pub input_cluster: Option<usize>,
}

/// Statistics accumulated from packing one or more blocks.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockPackStats {
    /// Count of `RESTART` triangle controls.
    pub total_restarts: usize,
    /// Count of `EDGE1` triangle controls.
    pub total_edge1: usize,
    /// Count of `EDGE2` triangle controls.
    pub total_edge2: usize,
    /// Count of `BACKTRACK` triangle controls.
    pub total_backtracks: usize,
    /// Total repeated vertex indices emitted.
    pub total_repeated_indices: usize,
    /// Cumulative front-buffer bits used.
    pub total_front_bits: usize,
    /// Cumulative repeated-index buffer bits used.
    pub total_repeat_bits: usize,
    /// Cumulative backward topology tail bits used.
    pub total_tail_bits: usize,
    /// Sum of decoded block AABB surface areas.
    pub total_block_surface_area: f64,
    /// Candidate fit failure counts.
    pub fit_failures: FitFailureCounts,
}

impl BlockPackStats {
    /// Accumulates counters from `other` into `self`.
    pub fn merge_from(&mut self, other: &Self) {
        self.total_restarts += other.total_restarts;
        self.total_edge1 += other.total_edge1;
        self.total_edge2 += other.total_edge2;
        self.total_backtracks += other.total_backtracks;
        self.total_repeated_indices += other.total_repeated_indices;
        self.total_front_bits += other.total_front_bits;
        self.total_repeat_bits += other.total_repeat_bits;
        self.total_tail_bits += other.total_tail_bits;
        self.total_block_surface_area += other.total_block_surface_area;
        self.fit_failures.merge_from(&other.fit_failures);
    }
}

/// Output produced by packing a single independent cluster.
#[derive(Debug, Clone)]
pub struct ClusterPackOutput {
    /// Encoded 128-byte blocks for this cluster.
    pub blocks: Vec<Block>,
    /// Per-output-triangle remaps for this cluster.
    pub triangle_remap: Vec<TriangleRemap>,
    /// Per-block-vertex source vertex IDs for this cluster.
    pub vertex_remap: Vec<u32>,
    /// Per-block offset ranges (cluster-local until merged).
    pub block_ranges: Vec<BlockRange>,
    /// Per-block sideband OMM descriptor maps.
    pub block_omm_maps: Vec<BlockOmmMap>,
    /// Packing statistics for this cluster.
    pub stats: BlockPackStats,
}

/// Fully planned candidate ready to emit as a [`Block`].
pub struct PlannedCandidate {
    /// Built strip topology and mappings.
    pub strip: BuiltStrip,
    /// Exact validated block layout plan.
    pub plan: PlannedBlock,
    /// Sideband OMM descriptor identity map.
    pub omm_map: BlockOmmMap,
    /// Decoded block AABB in `f64`.
    pub aabb: Aabb64,
}

/// Attempts to plan `candidate_indices` (indices into `retained`) as a single DGF1 block.
///
/// Runs the real block planner including vertex count, geometry palette, OMM descriptors,
/// position spans, front buffer (`<= 768` bits), repeated-index buffer (`<= 192` bits),
/// and total capacity (`repeat_end <= tail_start`).
pub fn try_plan_candidate(
    candidate_indices: &[usize],
    retained: &[RetainedTriangle],
    quantized_positions: &[[i32; 3]],
    reconstructed_positions: &[[f32; 3]],
    params: &ClusterPackParams,
) -> Result<PlannedCandidate, EncodeError> {
    if candidate_indices.is_empty() {
        return Err(EncodeError::EmptyBlock);
    }
    if candidate_indices.len() > usize::from(MAX_TRIANGLES) {
        return Err(EncodeError::DoesNotFit(FitFailure::TooManyTriangles {
            count: candidate_indices.len(),
            max: usize::from(MAX_TRIANGLES),
        }));
    }

    // Build CandidateTriangle slice.
    let mut cands = Vec::with_capacity(candidate_indices.len());
    for (local_i, &ret_idx) in candidate_indices.iter().enumerate() {
        let rt = retained[ret_idx];
        cands.push(CandidateTriangle {
            input_triangle: rt.input_triangle,
            candidate_index: local_i,
            vertices: rt.vertices,
        });
    }

    let strip = if params.use_strips {
        build_optimized_strip(&cands)
    } else {
        build_restart_only_strip(&cands)
    }
    .map_err(EncodeError::DoesNotFit)?;

    let mut block_q_positions = Vec::with_capacity(strip.local_to_source_vertex.len());
    let mut aabb = Aabb64::EMPTY;
    for &src_v in &strip.local_to_source_vertex {
        block_q_positions.push(quantized_positions[src_v as usize]);
        let rp = reconstructed_positions[src_v as usize];
        aabb.grow_point([f64::from(rp[0]), f64::from(rp[1]), f64::from(rp[2])]);
    }

    let mut geom_ids = Vec::with_capacity(candidate_indices.len());
    let mut opaque = Vec::with_capacity(candidate_indices.len());
    let mut omm_descriptors: Vec<u32> = Vec::new();
    let mut omm_selectors: Vec<u8> = Vec::new();
    let mut any_omm = false;
    let mut any_non_omm = false;

    for &cand_i in &strip.emitted_candidate_indices {
        let rt = retained[candidate_indices[cand_i]];
        geom_ids.push(rt.metadata.geometry_id);
        opaque.push(rt.metadata.opaque);
        match rt.omm_id {
            Some(id) => {
                any_omm = true;
                if let Some(pos) = omm_descriptors.iter().position(|&d| d == id) {
                    omm_selectors.push(pos as u8);
                } else {
                    let pos = omm_descriptors.len();
                    if pos >= usize::from(MAX_OMM_DESCRIPTORS) {
                        return Err(EncodeError::DoesNotFit(FitFailure::TooManyOmmDescriptors {
                            count: pos + 1,
                            max: usize::from(MAX_OMM_DESCRIPTORS),
                        }));
                    }
                    omm_descriptors.push(id);
                    omm_selectors.push(pos as u8);
                }
            }
            None => {
                any_non_omm = true;
            }
        }
    }

    // Never silently mix OMM and non-OMM triangles in the same block.
    if any_omm && any_non_omm {
        return Err(EncodeError::DoesNotFit(FitFailure::TooManyOmmDescriptors {
            count: omm_descriptors.len() + 1,
            max: usize::from(MAX_OMM_DESCRIPTORS),
        }));
    }

    let prepared_omm = if any_omm {
        Some(PreparedOmm {
            descriptor_count: omm_descriptors.len() as u8,
            triangle_descriptors: omm_selectors,
            hot_patch_bytes: None,
        })
    } else {
        None
    };

    let prepared = PreparedBlock {
        quantized_positions: block_q_positions,
        exponent: params.exponent,
        topology: strip.topology.clone(),
        geometry_ids: geom_ids,
        opaque,
        prim_id_base: 0, // assigned during deterministic merge
        user_data: params.user_data,
        omm: prepared_omm,
        allow_exponent_adjustment: params.enable_exponent_adjustment,
    };

    let plan = plan_block(&prepared)?;

    Ok(PlannedCandidate {
        strip,
        plan,
        omm_map: BlockOmmMap {
            descriptors: omm_descriptors,
        },
        aabb,
    })
}

/// Emits a [`PlannedCandidate`] into `out`, updating remap tables and statistics.
pub fn emit_planned_candidate(
    planned: PlannedCandidate,
    input_cluster: Option<usize>,
    out: &mut ClusterPackOutput,
) -> Result<(), EncodeError> {
    let block = serialize_planned_block(&planned.plan, &planned.strip.topology.controls)?;

    let vertex_offset = out.vertex_remap.len();
    let triangle_offset = out.triangle_remap.len();
    let vertex_count = planned.strip.local_to_source_vertex.len();
    let triangle_count = planned.strip.triangle_remap.len();

    for &ctrl in &planned.strip.topology.controls {
        match ctrl {
            TriControl::Restart => out.stats.total_restarts += 1,
            TriControl::Edge1 => out.stats.total_edge1 += 1,
            TriControl::Edge2 => out.stats.total_edge2 += 1,
            TriControl::Backtrack => out.stats.total_backtracks += 1,
        }
    }
    out.stats.total_repeated_indices += planned.plan.layout.repeated_index_count;
    out.stats.total_front_bits += planned.plan.layout.front.front_bits;
    out.stats.total_repeat_bits += planned.plan.layout.repeat_bits;
    out.stats.total_tail_bits +=
        planned.plan.layout.control_bits + planned.plan.layout.first_use_bits;
    out.stats.total_block_surface_area += planned.aabb.surface_area();

    out.blocks.push(block);
    out.vertex_remap
        .extend_from_slice(&planned.strip.local_to_source_vertex);
    out.triangle_remap
        .extend_from_slice(&planned.strip.triangle_remap);
    out.block_ranges.push(BlockRange {
        vertex_offset,
        vertex_count,
        triangle_offset,
        triangle_count,
        input_cluster,
        aabb: Some(planned.aabb.to_f32_pair()),
    });
    out.block_omm_maps.push(planned.omm_map);
    Ok(())
}

/// Greedy source-order packer used for [`crate::encode::PackingPolicy::BaselineRestartOnly`]
/// and [`crate::encode::PackingPolicy::BaselineWithStrips`].
pub fn pack_cluster_greedy(
    cluster: &[usize],
    retained: &[RetainedTriangle],
    quantized_positions: &[[i32; 3]],
    reconstructed_positions: &[[f32; 3]],
    params: &ClusterPackParams,
) -> Result<ClusterPackOutput, EncodeError> {
    let mut out = ClusterPackOutput {
        blocks: Vec::new(),
        triangle_remap: Vec::new(),
        vertex_remap: Vec::new(),
        block_ranges: Vec::new(),
        block_omm_maps: Vec::new(),
        stats: BlockPackStats::default(),
    };

    if cluster.is_empty() {
        return Ok(out);
    }

    let mut current: Vec<usize> = Vec::with_capacity(usize::from(MAX_TRIANGLES));
    let mut last_good: Option<PlannedCandidate> = None;

    for &ret_idx in cluster {
        current.push(ret_idx);
        match try_plan_candidate(
            &current,
            retained,
            quantized_positions,
            reconstructed_positions,
            params,
        ) {
            Ok(planned) => {
                last_good = Some(planned);
            }
            Err(EncodeError::DoesNotFit(reason)) => {
                out.stats.fit_failures.record(reason);
                if let Some(prev_planned) = last_good.take() {
                    emit_planned_candidate(prev_planned, params.input_cluster, &mut out)?;
                    current.clear();
                    current.push(ret_idx);
                    match try_plan_candidate(
                        &current,
                        retained,
                        quantized_positions,
                        reconstructed_positions,
                        params,
                    ) {
                        Ok(single) => last_good = Some(single),
                        Err(EncodeError::DoesNotFit(single_reason)) => {
                            out.stats.fit_failures.record(single_reason);
                            return Err(EncodeError::SingleTriangleDoesNotFit {
                                input_triangle: retained[ret_idx].input_triangle,
                                reason: single_reason,
                            });
                        }
                        Err(other) => return Err(other),
                    }
                } else {
                    return Err(EncodeError::SingleTriangleDoesNotFit {
                        input_triangle: retained[ret_idx].input_triangle,
                        reason,
                    });
                }
            }
            Err(other) => return Err(other),
        }
    }

    if let Some(final_planned) = last_good {
        emit_planned_candidate(final_planned, params.input_cluster, &mut out)?;
    }

    Ok(out)
}

/// Exact-fit full-sweep SAH subdivision packer over a bounded cluster using an explicit work stack.
pub fn pack_cluster_sah(
    cluster: Vec<usize>,
    retained: &[RetainedTriangle],
    spatials: &[TriSpatial],
    quantized_positions: &[[i32; 3]],
    reconstructed_positions: &[[f32; 3]],
    params: &ClusterPackParams,
) -> Result<ClusterPackOutput, EncodeError> {
    let mut out = ClusterPackOutput {
        blocks: Vec::new(),
        triangle_remap: Vec::new(),
        vertex_remap: Vec::new(),
        block_ranges: Vec::new(),
        block_omm_maps: Vec::new(),
        stats: BlockPackStats::default(),
    };

    if cluster.is_empty() {
        return Ok(out);
    }

    let mut stack: Vec<Vec<usize>> = Vec::new();
    stack.push(cluster);

    // Scratch buffers reused across SAH splits in this cluster.
    let mut axis_orders: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut prefix_area: Vec<(f64, f64)> = Vec::new();
    let mut suffix_area: Vec<(f64, f64)> = Vec::new();

    while let Some(candidate) = stack.pop() {
        match try_plan_candidate(
            &candidate,
            retained,
            quantized_positions,
            reconstructed_positions,
            params,
        ) {
            Ok(planned) => {
                emit_planned_candidate(planned, params.input_cluster, &mut out)?;
            }
            Err(EncodeError::DoesNotFit(reason)) => {
                out.stats.fit_failures.record(reason);
                if candidate.len() == 1 {
                    return Err(EncodeError::SingleTriangleDoesNotFit {
                        input_triangle: retained[candidate[0]].input_triangle,
                        reason,
                    });
                }
                let (left, right) = split_full_sweep_sah(
                    &candidate,
                    spatials,
                    &mut axis_orders,
                    &mut prefix_area,
                    &mut suffix_area,
                );
                // Push right first so left is emitted first (deterministic order).
                stack.push(right);
                stack.push(left);
            }
            Err(other) => return Err(other),
        }
    }

    Ok(out)
}

/// Full-sweep `f64` SAH partition with deterministic tie breaking
/// (surface-area cost -> perimeter cost -> balance -> axis -> source ID)
/// and zero-area / equal-centroid median fallback.
fn split_full_sweep_sah(
    candidate: &[usize],
    spatials: &[TriSpatial],
    axis_orders: &mut [Vec<usize>; 3],
    prefix_metrics: &mut Vec<(f64, f64)>,
    suffix_metrics: &mut Vec<(f64, f64)>,
) -> (Vec<usize>, Vec<usize>) {
    let n = candidate.len();
    debug_assert!(n >= 2);

    let mut best_cost = f64::INFINITY;
    let mut best_perim = f64::INFINITY;
    let mut best_balance = usize::MAX;
    let mut best_axis = 0usize;
    let mut best_split = n / 2;
    let mut found_nonzero_extent = false;

    for (axis, order) in axis_orders.iter_mut().enumerate() {
        order.clear();
        order.extend_from_slice(candidate);
        order.sort_unstable_by(|&a, &b| {
            spatials[a].centroid[axis]
                .total_cmp(&spatials[b].centroid[axis])
                .then_with(|| spatials[a].input_triangle.cmp(&spatials[b].input_triangle))
        });

        let c_min = spatials[order[0]].centroid[axis];
        let c_max = spatials[order[n - 1]].centroid[axis];
        if (c_max - c_min) <= 0.0 || !(c_max - c_min).is_finite() {
            continue;
        }
        found_nonzero_extent = true;

        prefix_metrics.clear();
        prefix_metrics.resize(n, (0.0, 0.0));
        let mut p_box = Aabb64::EMPTY;
        for (i, &idx) in order.iter().enumerate() {
            p_box.union(&spatials[idx].aabb);
            prefix_metrics[i] = (p_box.surface_area(), p_box.half_perimeter());
        }

        suffix_metrics.clear();
        suffix_metrics.resize(n, (0.0, 0.0));
        let mut s_box = Aabb64::EMPTY;
        for i in (0..n).rev() {
            s_box.union(&spatials[order[i]].aabb);
            suffix_metrics[i] = (s_box.surface_area(), s_box.half_perimeter());
        }

        // Evaluate every nonempty split `left = 0..split`, `right = split..n`.
        for split in 1..n {
            let left_count = split;
            let right_count = n - split;
            let cost = prefix_metrics[split - 1].0 * (left_count as f64)
                + suffix_metrics[split].0 * (right_count as f64);
            if !cost.is_finite() {
                continue;
            }
            let perim = prefix_metrics[split - 1].1 * (left_count as f64)
                + suffix_metrics[split].1 * (right_count as f64);
            let balance = left_count.abs_diff(right_count);
            let better = match cost.total_cmp(&best_cost) {
                core::cmp::Ordering::Less => true,
                core::cmp::Ordering::Equal => match perim.total_cmp(&best_perim) {
                    core::cmp::Ordering::Less => true,
                    core::cmp::Ordering::Equal => {
                        balance < best_balance || (balance == best_balance && axis < best_axis)
                    }
                    core::cmp::Ordering::Greater => false,
                },
                core::cmp::Ordering::Greater => false,
            };
            if better {
                best_cost = cost;
                best_perim = perim;
                best_balance = balance;
                best_axis = axis;
                best_split = split;
            }
        }
    }

    if found_nonzero_extent {
        let best_order = &axis_orders[best_axis];
        let left = best_order[..best_split].to_vec();
        let right = best_order[best_split..].to_vec();
        return (left, right);
    }

    // Deterministic median fallback when all centroids are identical across all axes.
    let mut sorted = candidate.to_vec();
    sorted.sort_unstable_by_key(|&idx| spatials[idx].input_triangle);
    let mid = sorted.len() / 2;
    let right = sorted.split_off(mid);
    (sorted, right)
}
