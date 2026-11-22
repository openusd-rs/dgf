//! Whole-mesh DGF1 baking orchestration, validation, remaps, statistics, and deterministic parallelism.

use crate::block::Block;
use crate::encode::cluster::{ClusterConfig, TriSpatial, build_coarse_clusters};
use crate::encode::pack::{
    BlockPackStats, ClusterPackOutput, ClusterPackParams, RetainedTriangle, pack_cluster_greedy,
    pack_cluster_sah,
};
use crate::encode::quantize::{QuantizationMode, QuantizationReport, quantize_mesh};
use crate::error::{EncodeError, FitFailure};
use crate::header::{Header, MAX_GEOM_ID, MAX_PRIM_ID};
use crate::remap::{BlockOmmMap, BlockRange, TriangleRemap};

/// Per-triangle geometry ID and opacity metadata supplied to [`encode_mesh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriangleMetadata {
    /// 24-bit geometry ID (`0..=16_777_215`).
    pub geometry_id: u32,
    /// Opacity flag (`true` = opaque, `false` = non-opaque).
    pub opaque: bool,
}

impl Default for TriangleMetadata {
    fn default() -> Self {
        Self {
            geometry_id: 0,
            opaque: true,
        }
    }
}

/// Borrowed static triangle mesh input for [`encode_mesh`].
#[derive(Debug, Clone, Copy)]
pub struct MeshView<'a> {
    /// Vertex positions as `[x, y, z]`.
    pub positions: &'a [[f32; 3]],
    /// Triangle vertex indices `[v0, v1, v2]`.
    pub triangles: &'a [[u32; 3]],
    /// Optional per-triangle geometry ID and opacity metadata (defaults to `(0, true)` when `None`).
    pub triangle_metadata: Option<&'a [TriangleMetadata]>,
    /// Optional per-triangle OMM identity (`Some(id)` associates an OMM descriptor, `None` has no OMM).
    pub triangle_omm_ids: Option<&'a [Option<u32>]>,
}

/// Policy for handling input triangles with repeated vertex indices (e.g., `[4, 4, 8]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DegeneratePolicy {
    /// Return [`EncodeError::DegenerateTriangle`] on any repeated-index input triangle.
    #[default]
    Reject,
    /// Omit repeated-index triangles and list their original indices in [`EncodedMesh::dropped_triangles`].
    DropAndReport,
}

/// Block packing strategy for [`encode_mesh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PackingPolicy {
    /// Source-order greedy packing using `RESTART` for every triangle (correctness baseline).
    BaselineRestartOnly,
    /// Source-order greedy packing using the deterministic strip builder.
    BaselineWithStrips,
    /// Coarse spatial clustering + exact-fit SAH subdivision with strip building (default).
    #[default]
    Sah,
}

/// Configuration options controlling quantization, packing, metadata, limits, and parallelism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncodeOptions {
    /// Position quantization policy (explicit exponent, max coordinate error, or target bit width).
    pub quantization: QuantizationMode,
    /// Block packing policy.
    pub packing: PackingPolicy,
    /// How repeated-index input triangles are handled.
    pub degenerate_policy: DegeneratePolicy,
    /// If `true`, return [`EncodeError::QuantizationCollapsedTriangles`] when quantization collapses any triangle.
    pub reject_quantization_collapses: bool,
    /// Starting 29-bit primitive ID assigned to output triangle 0.
    pub prim_id_base: u32,
    /// Optional constant 4-byte user-data word written to every block.
    pub user_data: Option<u32>,
    /// Whether to apply lossless per-block trailing-zero exponent adjustment.
    pub enable_exponent_adjustment: bool,
    /// Coarse clustering configuration used when `packing == PackingPolicy::Sah`.
    pub cluster_config: ClusterConfig,
    /// Maximum allowed output block count.
    pub max_blocks: usize,
    /// Enable deterministic multi-threaded cluster packing (requires the `parallel` Cargo feature).
    pub parallel: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            quantization: QuantizationMode::MaxCoordError(1e-4),
            packing: PackingPolicy::Sah,
            degenerate_policy: DegeneratePolicy::Reject,
            reject_quantization_collapses: false,
            prim_id_base: 0,
            user_data: None,
            enable_exponent_adjustment: false,
            cluster_config: ClusterConfig::default(),
            max_blocks: 16_777_216,
            parallel: cfg!(feature = "parallel"),
        }
    }
}

/// Breakdown of candidate fit failures recorded during block packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FitFailureCounts {
    /// Candidates exceeding 64 triangles.
    pub too_many_triangles: usize,
    /// Candidates exceeding 64 unique vertices.
    pub too_many_vertices: usize,
    /// Candidates exceeding 32 distinct `(geometry_id, opaque)` pairs.
    pub too_many_geom_palette_entries: usize,
    /// Candidates exceeding 7 distinct OMM descriptors.
    pub too_many_omm_descriptors: usize,
    /// Candidates exceeding 16-bit coordinate offset span (`65535`).
    pub position_span_too_large: usize,
    /// Candidates whose anchor fell outside signed 24-bit bounds.
    pub anchor_out_of_range: usize,
    /// Candidates exceeding the 96-byte (768-bit) front-buffer limit.
    pub front_buffer_overflow: usize,
    /// Candidates exceeding the 24-byte (192-bit) repeated-index buffer limit.
    pub repeat_buffer_overflow: usize,
    /// Candidates whose forward stream overlapped the backward topology tail.
    pub block_capacity_overflow: usize,
}

impl FitFailureCounts {
    /// Increments the counter corresponding to `reason`.
    pub fn record(&mut self, reason: FitFailure) {
        match reason {
            FitFailure::TooManyTriangles { .. } => self.too_many_triangles += 1,
            FitFailure::TooManyVertices { .. } => self.too_many_vertices += 1,
            FitFailure::TooManyGeomPaletteEntries { .. } => self.too_many_geom_palette_entries += 1,
            FitFailure::TooManyOmmDescriptors { .. } => self.too_many_omm_descriptors += 1,
            FitFailure::PositionSpanTooLarge { .. } => self.position_span_too_large += 1,
            FitFailure::AnchorOutOfRange { .. } => self.anchor_out_of_range += 1,
            FitFailure::FrontBufferOverflow { .. } => self.front_buffer_overflow += 1,
            FitFailure::RepeatBufferOverflow { .. } => self.repeat_buffer_overflow += 1,
            FitFailure::BlockCapacityOverflow { .. } => self.block_capacity_overflow += 1,
        }
    }

    /// Adds all counters from `other` into `self`.
    pub fn merge_from(&mut self, other: &Self) {
        self.too_many_triangles += other.too_many_triangles;
        self.too_many_vertices += other.too_many_vertices;
        self.too_many_geom_palette_entries += other.too_many_geom_palette_entries;
        self.too_many_omm_descriptors += other.too_many_omm_descriptors;
        self.position_span_too_large += other.position_span_too_large;
        self.anchor_out_of_range += other.anchor_out_of_range;
        self.front_buffer_overflow += other.front_buffer_overflow;
        self.repeat_buffer_overflow += other.repeat_buffer_overflow;
        self.block_capacity_overflow += other.block_capacity_overflow;
    }
}

/// Detailed compression, topology, utilization, and spatial-bounds metrics for an encoded mesh.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EncodeStatistics {
    /// Number of supplied input vertices.
    pub input_vertices: usize,
    /// Number of supplied input triangles.
    pub input_triangles: usize,
    /// Number of retained triangles encoded into blocks.
    pub retained_triangles: usize,
    /// Number of degenerate input triangles dropped under [`DegeneratePolicy::DropAndReport`].
    pub dropped_triangles: usize,
    /// Number of 128-byte DGF1 blocks produced.
    pub output_blocks: usize,
    /// Total stored block vertices across all blocks (`vertex_remap.len()`).
    pub total_block_vertices: usize,
    /// Total `RESTART` triangle controls emitted across all blocks.
    pub total_restarts: usize,
    /// Total `EDGE1` triangle controls emitted across all blocks.
    pub total_edge1: usize,
    /// Total `EDGE2` triangle controls emitted across all blocks.
    pub total_edge2: usize,
    /// Total `BACKTRACK` triangle controls emitted across all blocks.
    pub total_backtracks: usize,
    /// Total repeated vertex indices emitted in forward reuse buffers.
    pub total_repeated_indices: usize,
    /// Sum of front-buffer bits across all blocks.
    pub total_front_bits: usize,
    /// Sum of repeated-index buffer bits across all blocks.
    pub total_repeat_bits: usize,
    /// Sum of backward topology tail bits across all blocks.
    pub total_tail_bits: usize,
    /// Ratio of stored block vertices to active unique source vertices (`0.0` if empty).
    pub vertex_duplication_factor: f64,
    /// `128 * output_blocks / retained_triangles` (`0.0` if empty).
    pub bytes_per_triangle: f64,
    /// Mean triangles per emitted block (`0.0` if empty).
    pub mean_triangles_per_block: f64,
    /// Mean vertices per emitted block (`0.0` if empty).
    pub mean_vertices_per_block: f64,
    /// Sum of decoded block AABB surface areas (spatial coherence metric).
    pub total_block_surface_area: f64,
    /// Counts of candidate fit failures encountered during packing.
    pub fit_failures: FitFailureCounts,
}

/// Complete result of encoding a mesh into DGF1 blocks, including mandatory provenance maps.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedMesh {
    /// Emitted 128-byte DGF1 blocks in output order.
    pub blocks: Vec<Block>,
    /// Per-output-triangle mapping back to input triangle and cyclic corner permutation.
    pub triangle_remap: Vec<TriangleRemap>,
    /// Per-stored-block-vertex mapping back to original source vertex index.
    pub vertex_remap: Vec<u32>,
    /// Per-block offsets, counts, cluster IDs, and decoded AABBs.
    pub block_ranges: Vec<BlockRange>,
    /// Per-block sideband mapping from local OMM descriptors `0..D` to caller OMM identities.
    pub block_omm_maps: Vec<BlockOmmMap>,
    /// Original source triangle indices dropped under [`DegeneratePolicy::DropAndReport`].
    pub dropped_triangles: Vec<u32>,
    /// Quantization accuracy and collapse report.
    pub quantization: QuantizationReport,
    /// Compression, topology, and spatial packing metrics.
    pub statistics: EncodeStatistics,
}

impl EncodedMesh {
    /// Serializes all blocks into a contiguous raw DGF1 byte stream (`128 * blocks.len()` bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.blocks.len() * 128);
        for b in &self.blocks {
            out.extend_from_slice(b.as_bytes());
        }
        out
    }
}

/// Encodes a static triangle mesh into DGF1 blocks with full vertex and triangle/corner mappings.
pub fn encode_mesh(
    mesh: MeshView<'_>,
    options: &EncodeOptions,
) -> Result<EncodedMesh, EncodeError> {
    encode_mesh_internal(mesh, None, options, None::<fn(usize, &BlockRange) -> u32>)
}

/// Encodes a static triangle mesh while reserving the 4-byte user-data slot on every block and
/// populating each block's user-data word after block planning via `user_data_fn`.
pub fn encode_mesh_with_user_data<F>(
    mesh: MeshView<'_>,
    options: &EncodeOptions,
    user_data_fn: F,
) -> Result<EncodedMesh, EncodeError>
where
    F: Fn(usize, &BlockRange) -> u32,
{
    let mut opts = *options;
    opts.user_data = Some(0);
    encode_mesh_internal(mesh, None, &opts, Some(user_data_fn))
}

/// Encodes a static triangle mesh using caller-supplied preclustered triangle partitions.
///
/// Each slice in `clusters` contains original source triangle indices. Every retained input
/// triangle must appear in exactly one cluster.
pub fn encode_preclustered_mesh(
    mesh: MeshView<'_>,
    clusters: &[&[u32]],
    options: &EncodeOptions,
) -> Result<EncodedMesh, EncodeError> {
    encode_mesh_internal(
        mesh,
        Some(clusters),
        options,
        None::<fn(usize, &BlockRange) -> u32>,
    )
}

fn encode_mesh_internal<F>(
    mesh: MeshView<'_>,
    preclusters: Option<&[&[u32]]>,
    options: &EncodeOptions,
    user_data_fn: Option<F>,
) -> Result<EncodedMesh, EncodeError>
where
    F: Fn(usize, &BlockRange) -> u32,
{
    // 1. Validate mesh dimensions and optional slices.
    if mesh.positions.len() > (u32::MAX as usize) {
        return Err(EncodeError::TooManyInputVertices {
            count: mesh.positions.len(),
        });
    }
    if mesh.triangles.len() > (u32::MAX as usize) {
        return Err(EncodeError::TooManyInputTriangles {
            count: mesh.triangles.len(),
        });
    }
    if let Some(meta) = mesh.triangle_metadata
        && meta.len() != mesh.triangles.len()
    {
        return Err(EncodeError::MetadataLengthMismatch {
            expected: mesh.triangles.len(),
            found: meta.len(),
        });
    }
    if let Some(omm) = mesh.triangle_omm_ids
        && omm.len() != mesh.triangles.len()
    {
        return Err(EncodeError::OmmLengthMismatch {
            expected: mesh.triangles.len(),
            found: omm.len(),
        });
    }

    // Validate vertex indices, metadata ranges, and apply degenerate policy.
    let mut retained: Vec<RetainedTriangle> = Vec::with_capacity(mesh.triangles.len());
    let mut retained_raw_tris: Vec<[u32; 3]> = Vec::with_capacity(mesh.triangles.len());
    let mut dropped_triangles: Vec<u32> = Vec::new();
    let mut source_to_retained: Vec<Option<usize>> = vec![None; mesh.triangles.len()];

    for (t, &tri) in mesh.triangles.iter().enumerate() {
        for (c, &idx) in tri.iter().enumerate() {
            if (idx as usize) >= mesh.positions.len() {
                return Err(EncodeError::VertexIndexOutOfBounds {
                    triangle: t,
                    corner: c as u8,
                    index: idx,
                    num_vertices: mesh.positions.len(),
                });
            }
        }

        let meta = mesh
            .triangle_metadata
            .map_or_else(TriangleMetadata::default, |m| m[t]);
        if meta.geometry_id > MAX_GEOM_ID {
            return Err(EncodeError::InvalidGeometryId {
                triangle: t,
                geometry_id: meta.geometry_id,
            });
        }

        let is_degenerate = tri[0] == tri[1] || tri[1] == tri[2] || tri[0] == tri[2];
        if is_degenerate {
            match options.degenerate_policy {
                DegeneratePolicy::Reject => {
                    return Err(EncodeError::DegenerateTriangle {
                        triangle: t,
                        indices: tri,
                    });
                }
                DegeneratePolicy::DropAndReport => {
                    dropped_triangles.push(t as u32);
                    continue;
                }
            }
        }

        let ret_idx = retained.len();
        source_to_retained[t] = Some(ret_idx);
        retained.push(RetainedTriangle {
            input_triangle: t as u32,
            vertices: tri,
            metadata: meta,
            omm_id: mesh.triangle_omm_ids.and_then(|s| s[t]),
        });
        retained_raw_tris.push(tri);
    }

    // Check aggregate 29-bit primitive ID range upfront.
    if !retained.is_empty() {
        let last_prim = u64::from(options.prim_id_base) + (retained.len() as u64 - 1);
        if options.prim_id_base > MAX_PRIM_ID || last_prim > u64::from(MAX_PRIM_ID) {
            return Err(EncodeError::PrimitiveIdRangeExceeded {
                base: options.prim_id_base,
                total_triangles: retained.len(),
            });
        }
    } else if options.prim_id_base > MAX_PRIM_ID {
        return Err(EncodeError::PrimitiveIdRangeExceeded {
            base: options.prim_id_base,
            total_triangles: 0,
        });
    }

    // 2. Quantize all source vertices once (guarantees shared vertices agree across all blocks).
    let quantized = quantize_mesh(
        mesh.positions,
        &retained_raw_tris,
        options.quantization,
        options.reject_quantization_collapses,
    )?;

    // Empty or all-dropped mesh produces a valid empty result with zero blocks.
    if retained.is_empty() {
        if let Some(clusters) = preclusters {
            // Still validate no out-of-bounds indices in preclusters.
            for (ci, cluster) in clusters.iter().enumerate() {
                for &src_t in *cluster {
                    if (src_t as usize) >= mesh.triangles.len() {
                        return Err(EncodeError::InvalidClusterTriangle {
                            cluster: ci,
                            triangle: src_t,
                            num_triangles: mesh.triangles.len(),
                        });
                    }
                }
            }
        }
        return Ok(EncodedMesh {
            blocks: Vec::new(),
            triangle_remap: Vec::new(),
            vertex_remap: Vec::new(),
            block_ranges: Vec::new(),
            block_omm_maps: Vec::new(),
            dropped_triangles,
            quantization: quantized.report,
            statistics: EncodeStatistics {
                input_vertices: mesh.positions.len(),
                input_triangles: mesh.triangles.len(),
                retained_triangles: 0,
                dropped_triangles: mesh.triangles.len(),
                ..EncodeStatistics::default()
            },
        });
    }

    // 3. Build per-triangle spatial descriptors in f64 from quantized reconstructed positions.
    let spatials: Vec<TriSpatial> = retained
        .iter()
        .map(|rt| {
            TriSpatial::new(
                rt.input_triangle,
                quantized.reconstructed[rt.vertices[0] as usize],
                quantized.reconstructed[rt.vertices[1] as usize],
                quantized.reconstructed[rt.vertices[2] as usize],
            )
        })
        .collect();

    // 4. Construct cluster work units, partitioning OMM vs non-OMM triangles cleanly.
    let work_units: Vec<(Option<usize>, Vec<usize>)> = if let Some(clusters) = preclusters {
        build_preclustered_work_units(
            clusters,
            mesh.triangles.len(),
            &source_to_retained,
            &retained,
        )?
    } else {
        build_auto_work_units(&retained, &retained_raw_tris, &spatials, options)
    };

    // 5. Pack clusters (serially or deterministically in parallel).
    let cluster_outputs = pack_work_units(
        work_units,
        &retained,
        &spatials,
        &quantized.positions,
        &quantized.reconstructed,
        quantized.exponent.biased(),
        options,
    )?;

    // 6. Deterministically merge ordered cluster slots, assign primitive bases, and check limits.
    let mut blocks = Vec::new();
    let mut triangle_remap = Vec::with_capacity(retained.len());
    let mut vertex_remap = Vec::new();
    let mut block_ranges = Vec::new();
    let mut block_omm_maps = Vec::new();
    let mut pack_stats = BlockPackStats::default();

    let mut prim_cursor = options.prim_id_base;

    for mut c_out in cluster_outputs {
        let base_vert_off = vertex_remap.len();
        let base_tri_off = triangle_remap.len();

        for (mut block, mut range) in c_out.blocks.drain(..).zip(c_out.block_ranges.drain(..)) {
            if blocks.len() >= options.max_blocks {
                return Err(EncodeError::MaxBlocksExceeded {
                    count: blocks.len() + 1,
                    limit: options.max_blocks,
                });
            }

            // Assign contiguous 29-bit primitive ID base via validated header update helper.
            Header::update_prim_id_base(block.as_bytes_mut(), prim_cursor)
                .map_err(EncodeError::InternalValidation)?;
            prim_cursor += range.triangle_count as u32;

            range.vertex_offset += base_vert_off;
            range.triangle_offset += base_tri_off;

            // Populate per-block user data callback if requested.
            if let Some(ref ud_fn) = user_data_fn {
                let block_idx = blocks.len();
                let word = ud_fn(block_idx, &range);
                block.as_bytes_mut()[20..24].copy_from_slice(&word.to_le_bytes());
            }

            blocks.push(block);
            block_ranges.push(range);
        }

        vertex_remap.extend_from_slice(&c_out.vertex_remap);
        triangle_remap.extend_from_slice(&c_out.triangle_remap);
        block_omm_maps.extend(c_out.block_omm_maps);
        pack_stats.merge_from(&c_out.stats);
    }

    // 7. Compute aggregate statistics.
    let active_unique_verts = mesh.positions.len() - quantized.report.unused_vertices;
    let output_blocks = blocks.len();
    let retained_count = retained.len();
    let total_block_vertices = vertex_remap.len();

    let statistics = EncodeStatistics {
        input_vertices: mesh.positions.len(),
        input_triangles: mesh.triangles.len(),
        retained_triangles: retained_count,
        dropped_triangles: dropped_triangles.len(),
        output_blocks,
        total_block_vertices,
        total_restarts: pack_stats.total_restarts,
        total_edge1: pack_stats.total_edge1,
        total_edge2: pack_stats.total_edge2,
        total_backtracks: pack_stats.total_backtracks,
        total_repeated_indices: pack_stats.total_repeated_indices,
        total_front_bits: pack_stats.total_front_bits,
        total_repeat_bits: pack_stats.total_repeat_bits,
        total_tail_bits: pack_stats.total_tail_bits,
        vertex_duplication_factor: if active_unique_verts > 0 {
            (total_block_vertices as f64) / (active_unique_verts as f64)
        } else {
            0.0
        },
        bytes_per_triangle: if retained_count > 0 {
            ((output_blocks * 128) as f64) / (retained_count as f64)
        } else {
            0.0
        },
        mean_triangles_per_block: if output_blocks > 0 {
            (retained_count as f64) / (output_blocks as f64)
        } else {
            0.0
        },
        mean_vertices_per_block: if output_blocks > 0 {
            (total_block_vertices as f64) / (output_blocks as f64)
        } else {
            0.0
        },
        total_block_surface_area: pack_stats.total_block_surface_area,
        fit_failures: pack_stats.fit_failures,
    };

    Ok(EncodedMesh {
        blocks,
        triangle_remap,
        vertex_remap,
        block_ranges,
        block_omm_maps,
        dropped_triangles,
        quantization: quantized.report,
        statistics,
    })
}

type WorkUnit = (Option<usize>, Vec<usize>);

fn build_preclustered_work_units(
    clusters: &[&[u32]],
    num_input_triangles: usize,
    source_to_retained: &[Option<usize>],
    retained: &[RetainedTriangle],
) -> Result<Vec<WorkUnit>, EncodeError> {
    let mut seen_retained = vec![false; retained.len()];
    let mut units = Vec::new();

    for (ci, cluster) in clusters.iter().enumerate() {
        let mut non_omm = Vec::new();
        let mut with_omm = Vec::new();
        for &src_t in *cluster {
            if (src_t as usize) >= num_input_triangles {
                return Err(EncodeError::InvalidClusterTriangle {
                    cluster: ci,
                    triangle: src_t,
                    num_triangles: num_input_triangles,
                });
            }
            if let Some(ret_idx) = source_to_retained[src_t as usize] {
                if seen_retained[ret_idx] {
                    return Err(EncodeError::DuplicateClusterTriangle { triangle: src_t });
                }
                seen_retained[ret_idx] = true;
                if retained[ret_idx].omm_id.is_some() {
                    with_omm.push(ret_idx);
                } else {
                    non_omm.push(ret_idx);
                }
            }
        }
        if !non_omm.is_empty() {
            units.push((Some(ci), non_omm));
        }
        if !with_omm.is_empty() {
            units.push((Some(ci), with_omm));
        }
    }

    let missing = seen_retained.iter().filter(|&&s| !s).count();
    if missing > 0 {
        return Err(EncodeError::MissingClusterTriangles { missing });
    }

    Ok(units)
}

fn build_auto_work_units(
    retained: &[RetainedTriangle],
    retained_raw_tris: &[[u32; 3]],
    spatials: &[TriSpatial],
    options: &EncodeOptions,
) -> Vec<(Option<usize>, Vec<usize>)> {
    // Partition triangles without OMM separately from triangles with OMM associations.
    let mut non_omm = Vec::new();
    let mut with_omm = Vec::new();
    for (idx, rt) in retained.iter().enumerate() {
        if rt.omm_id.is_some() {
            with_omm.push(idx);
        } else {
            non_omm.push(idx);
        }
    }

    let mut groups = Vec::new();
    if !non_omm.is_empty() {
        groups.push(non_omm);
    }
    if !with_omm.is_empty() {
        groups.push(with_omm);
    }

    match options.packing {
        PackingPolicy::BaselineRestartOnly | PackingPolicy::BaselineWithStrips => {
            groups.into_iter().map(|g| (None, g)).collect()
        }
        PackingPolicy::Sah => {
            let mut units = Vec::new();
            let mut cluster_id = 0usize;
            for group in groups {
                let coarse = build_coarse_clusters(
                    group,
                    retained_raw_tris,
                    spatials,
                    options.cluster_config,
                );
                for c in coarse {
                    units.push((Some(cluster_id), c));
                    cluster_id += 1;
                }
            }
            units
        }
    }
}

fn pack_work_units(
    work_units: Vec<(Option<usize>, Vec<usize>)>,
    retained: &[RetainedTriangle],
    spatials: &[TriSpatial],
    quantized_positions: &[[i32; 3]],
    reconstructed_positions: &[[f32; 3]],
    exponent: crate::header::BiasedExponent,
    options: &EncodeOptions,
) -> Result<Vec<ClusterPackOutput>, EncodeError> {
    let pack_one = |(input_cluster, cluster): (Option<usize>, Vec<usize>)| -> Result<ClusterPackOutput, EncodeError> {
        let params = ClusterPackParams {
            exponent,
            user_data: options.user_data,
            enable_exponent_adjustment: options.enable_exponent_adjustment,
            use_strips: !matches!(options.packing, PackingPolicy::BaselineRestartOnly),
            input_cluster,
        };
        match options.packing {
            PackingPolicy::BaselineRestartOnly | PackingPolicy::BaselineWithStrips => {
                pack_cluster_greedy(
                    &cluster,
                    retained,
                    quantized_positions,
                    reconstructed_positions,
                    &params,
                )
            }
            PackingPolicy::Sah => pack_cluster_sah(
                cluster,
                retained,
                spatials,
                quantized_positions,
                reconstructed_positions,
                &params,
            ),
        }
    };

    #[cfg(feature = "parallel")]
    if options.parallel && work_units.len() > 1 {
        use rayon::prelude::*;
        return work_units.into_par_iter().map(pack_one).collect();
    }

    let mut out = Vec::with_capacity(work_units.len());
    for unit in work_units {
        out.push(pack_one(unit)?);
    }
    Ok(out)
}
