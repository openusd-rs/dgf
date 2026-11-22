//! Deterministic DGF1 strip builders (`RESTART`-only baseline and low-valence strip with `BACKTRACK`).
//!
//! Only cyclic corner rotations `[0, 1, 2]`, `[1, 2, 0]`, and `[2, 0, 1]` are ever emitted,
//! strictly preserving triangle winding and recording exact `input_corners` remaps.

use crate::encode::adjacency::AdjacencyGraph;
use crate::encode::block::PreparedTopology;
use crate::error::FitFailure;
use crate::header::{MAX_TRIANGLES, MAX_VERTICES};
use crate::remap::TriangleRemap;
use crate::topology::TriControl;

/// Cyclic corner rotations preserving triangle winding.
const CYCLIC_ROTATIONS: [[u8; 3]; 3] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];

/// A candidate triangle tagged with its original source triangle index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateTriangle {
    /// Original triangle index in the input mesh.
    pub input_triangle: u32,
    /// Index into the caller's candidate slice (used to reorder per-triangle metadata / OMM).
    pub candidate_index: usize,
    /// Source vertex IDs `[v0, v1, v2]`.
    pub vertices: [u32; 3],
}

/// Complete strip topology plus synchronized vertex and triangle/corner mappings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltStrip {
    /// First-use-ordered DGF1 topology ready for [`crate::encode::PreparedBlock`].
    pub topology: PreparedTopology,
    /// Maps each local vertex `0..num_vertices` to its original source vertex ID.
    pub local_to_source_vertex: Vec<u32>,
    /// Per-output-triangle mapping back to input triangle and cyclic corner permutation.
    pub triangle_remap: Vec<TriangleRemap>,
    /// Maps each output triangle `0..T` back to `CandidateTriangle::candidate_index`.
    pub emitted_candidate_indices: Vec<usize>,
}

/// Builds a simple source-order `RESTART`-only strip (correctness baseline).
pub fn build_restart_only_strip(
    candidates: &[CandidateTriangle],
) -> Result<BuiltStrip, FitFailure> {
    if candidates.len() > usize::from(MAX_TRIANGLES) {
        return Err(FitFailure::TooManyTriangles {
            count: candidates.len(),
            max: usize::from(MAX_TRIANGLES),
        });
    }

    let mut controls = Vec::with_capacity(candidates.len());
    let mut strip_source_verts = Vec::with_capacity(candidates.len() * 3);
    let mut triangle_remap = Vec::with_capacity(candidates.len());
    let mut emitted_candidate_indices = Vec::with_capacity(candidates.len());

    for cand in candidates {
        controls.push(TriControl::Restart);
        strip_source_verts.extend_from_slice(&cand.vertices);
        triangle_remap.push(TriangleRemap {
            input_triangle: cand.input_triangle,
            input_corners: [0, 1, 2],
        });
        emitted_candidate_indices.push(cand.candidate_index);
    }

    finalize_first_use_strip(
        controls,
        &strip_source_verts,
        triangle_remap,
        emitted_candidate_indices,
    )
}

/// Builds a deterministic low-valence strip using `RESTART`, `EDGE1`, `EDGE2`, and `BACKTRACK`.
///
/// Preserves winding strictly via cyclic rotations and gracefully falls back to `RESTART` on
/// non-manifold edges, mixed winding, or disconnected components.
pub fn build_optimized_strip(candidates: &[CandidateTriangle]) -> Result<BuiltStrip, FitFailure> {
    let n = candidates.len();
    if n > usize::from(MAX_TRIANGLES) {
        return Err(FitFailure::TooManyTriangles {
            count: n,
            max: usize::from(MAX_TRIANGLES),
        });
    }

    // Fast unique-vertex check before building adjacency.
    let mut unique_verts: Vec<u32> = Vec::with_capacity(n * 3);
    for c in candidates {
        unique_verts.extend_from_slice(&c.vertices);
    }
    unique_verts.sort_unstable();
    unique_verts.dedup();
    if unique_verts.len() > usize::from(MAX_VERTICES) {
        return Err(FitFailure::TooManyVertices {
            count: unique_verts.len(),
            max: usize::from(MAX_VERTICES),
        });
    }

    if n == 1 {
        return build_restart_only_strip(candidates);
    }

    let raw_tris: Vec<[u32; 3]> = candidates.iter().map(|c| c.vertices).collect();
    let graph = AdjacencyGraph::build(&raw_tris);

    let mut visited = vec![false; n];
    let mut remaining = n;

    let mut controls = Vec::with_capacity(n);
    let mut strip_source_verts = Vec::with_capacity(n * 3);
    let mut triangle_remap = Vec::with_capacity(n);
    let mut emitted_candidate_indices = Vec::with_capacity(n);

    // Track emitted source triangles for Edge1 / Edge2 / Backtrack transitions.
    let mut emitted_tris: Vec<[u32; 3]> = Vec::with_capacity(n);

    while remaining > 0 {
        let emitted_len = emitted_tris.len();

        // 1. Try continuing from the previous triangle via EDGE1 or EDGE2.
        let mut continued = false;
        if emitted_len > 0 {
            let prev = emitted_tris[emitted_len - 1];
            let edge1_target = (prev[2], prev[1]); // reversed Edge1 (b, c) -> (c, b)
            let edge2_target = (prev[0], prev[2]); // reversed Edge2 (c, a) -> (a, c)

            let best_e1 = pick_best_neighbor(&graph, &raw_tris, &visited, edge1_target);
            let best_e2 = pick_best_neighbor(&graph, &raw_tris, &visited, edge2_target);

            let chosen = match (best_e1, best_e2) {
                (Some((t1, rot1, val1)), Some((t2, rot2, val2))) => {
                    if val1 <= val2 {
                        Some((TriControl::Edge1, t1, rot1))
                    } else {
                        Some((TriControl::Edge2, t2, rot2))
                    }
                }
                (Some((t1, rot1, _)), None) => Some((TriControl::Edge1, t1, rot1)),
                (None, Some((t2, rot2, _))) => Some((TriControl::Edge2, t2, rot2)),
                (None, None) => None,
            };

            if let Some((ctrl, next_tri, rot)) = chosen {
                let rotated = apply_rotation(raw_tris[next_tri], rot);
                visited[next_tri] = true;
                remaining -= 1;
                controls.push(ctrl);
                strip_source_verts.push(rotated[2]);
                emitted_tris.push(rotated);
                triangle_remap.push(TriangleRemap {
                    input_triangle: candidates[next_tri].input_triangle,
                    input_corners: rot,
                });
                emitted_candidate_indices.push(candidates[next_tri].candidate_index);
                continued = true;
            }
        }

        if continued {
            continue;
        }

        // 2. Try legal BACKTRACK when previous control was EDGE1 or EDGE2.
        if emitted_len >= 2 {
            let prev_ctrl = controls[emitted_len - 1];
            let prev_prev = emitted_tris[emitted_len - 2];
            let bt_target = match prev_ctrl {
                TriControl::Edge1 => Some((prev_prev[0], prev_prev[2])), // [d, f, n]
                TriControl::Edge2 => Some((prev_prev[2], prev_prev[1])), // [f, e, n]
                TriControl::Restart | TriControl::Backtrack => None,
            };

            if let Some(target) = bt_target
                && let Some((next_tri, rot, _)) =
                    pick_best_neighbor(&graph, &raw_tris, &visited, target)
            {
                let rotated = apply_rotation(raw_tris[next_tri], rot);
                visited[next_tri] = true;
                remaining -= 1;
                controls.push(TriControl::Backtrack);
                strip_source_verts.push(rotated[2]);
                emitted_tris.push(rotated);
                triangle_remap.push(TriangleRemap {
                    input_triangle: candidates[next_tri].input_triangle,
                    input_corners: rot,
                });
                emitted_candidate_indices.push(candidates[next_tri].candidate_index);
                continue;
            }
        }

        // 3. Start a new strip with RESTART at the lowest-valence unvisited triangle.
        let mut seed_tri = 0usize;
        let mut seed_val = usize::MAX;
        for i in 0..n {
            if !visited[i] {
                let val = graph.unvisited_valence(&raw_tris, &visited, i);
                if val < seed_val {
                    seed_val = val;
                    seed_tri = i;
                    if val == 0 {
                        break;
                    }
                }
            }
        }

        // Choose cyclic rotation that maximizes continuable edges (Edge1 and Edge2).
        let rot = choose_best_restart_rotation(&graph, &raw_tris, &visited, seed_tri);
        let rotated = apply_rotation(raw_tris[seed_tri], rot);
        visited[seed_tri] = true;
        remaining -= 1;
        controls.push(TriControl::Restart);
        strip_source_verts.extend_from_slice(&rotated);
        emitted_tris.push(rotated);
        triangle_remap.push(TriangleRemap {
            input_triangle: candidates[seed_tri].input_triangle,
            input_corners: rot,
        });
        emitted_candidate_indices.push(candidates[seed_tri].candidate_index);
    }

    finalize_first_use_strip(
        controls,
        &strip_source_verts,
        triangle_remap,
        emitted_candidate_indices,
    )
}

fn apply_rotation(tri: [u32; 3], rot: [u8; 3]) -> [u32; 3] {
    [
        tri[usize::from(rot[0])],
        tri[usize::from(rot[1])],
        tri[usize::from(rot[2])],
    ]
}

/// Finds the lowest-valence unvisited triangle containing directed edge `(target.0, target.1)`,
/// returning `(tri_index, cyclic_rotation_putting_target_on_edge0, valence)`.
fn pick_best_neighbor(
    graph: &AdjacencyGraph,
    triangles: &[[u32; 3]],
    visited: &[bool],
    target: (u32, u32),
) -> Option<(usize, [u8; 3], usize)> {
    let mut best: Option<(usize, [u8; 3], usize)> = None;
    for (nbr, edge_idx) in graph.triangles_with_directed_edge(target.0, target.1) {
        if visited[nbr] {
            continue;
        }
        let rot = CYCLIC_ROTATIONS[usize::from(edge_idx)];
        let rotated = apply_rotation(triangles[nbr], rot);
        debug_assert_eq!((rotated[0], rotated[1]), target);
        let val = graph.unvisited_valence(triangles, visited, nbr);
        match best {
            None => best = Some((nbr, rot, val)),
            Some((best_t, _, best_val)) => {
                if val < best_val || (val == best_val && nbr < best_t) {
                    best = Some((nbr, rot, val));
                }
            }
        }
    }
    best
}

/// Chooses the cyclic rotation `[0,1,2]`, `[1,2,0]`, or `[2,0,1]` for a `RESTART` seed triangle
/// that places as many continuable unvisited edges on Edge1 and Edge2 as possible.
fn choose_best_restart_rotation(
    graph: &AdjacencyGraph,
    triangles: &[[u32; 3]],
    visited: &[bool],
    tri: usize,
) -> [u8; 3] {
    let mut best_rot = CYCLIC_ROTATIONS[0];
    let mut best_score = 0u8;

    for rot in CYCLIC_ROTATIONS {
        let r = apply_rotation(triangles[tri], rot);
        let has_e1 = graph
            .triangles_with_directed_edge(r[2], r[1])
            .any(|(nbr, _)| nbr != tri && !visited[nbr]);
        let has_e2 = graph
            .triangles_with_directed_edge(r[0], r[2])
            .any(|(nbr, _)| nbr != tri && !visited[nbr]);
        let score = u8::from(has_e1) * 2 + u8::from(has_e2);
        if score > best_score {
            best_score = score;
            best_rot = rot;
        }
    }
    best_rot
}

/// Maps source vertex IDs in `strip_source_verts` into first-use local vertex IDs `0..V`.
fn finalize_first_use_strip(
    controls: Vec<TriControl>,
    strip_source_verts: &[u32],
    triangle_remap: Vec<TriangleRemap>,
    emitted_candidate_indices: Vec<usize>,
) -> Result<BuiltStrip, FitFailure> {
    let mut local_to_source: Vec<u32> = Vec::with_capacity(usize::from(MAX_VERTICES));
    let mut strip_indices: Vec<u8> = Vec::with_capacity(strip_source_verts.len());

    for &src_v in strip_source_verts {
        if let Some(local) = local_to_source.iter().position(|&v| v == src_v) {
            strip_indices.push(local as u8);
        } else {
            let local = local_to_source.len();
            if local >= usize::from(MAX_VERTICES) {
                return Err(FitFailure::TooManyVertices {
                    count: local + 1,
                    max: usize::from(MAX_VERTICES),
                });
            }
            local_to_source.push(src_v);
            strip_indices.push(local as u8);
        }
    }

    Ok(BuiltStrip {
        topology: PreparedTopology {
            controls,
            strip_indices,
            num_vertices: local_to_source.len() as u8,
        },
        local_to_source_vertex: local_to_source,
        triangle_remap,
        emitted_candidate_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::expand_strip_to_triangles;

    fn verify_strip_preserves_winding(candidates: &[CandidateTriangle], built: &BuiltStrip) {
        let num_tris = built.topology.controls.len();
        assert_eq!(num_tris, candidates.len());
        let expanded = expand_strip_to_triangles(
            &built.topology.controls,
            num_tris as u8,
            &built.topology.strip_indices,
            built.topology.strip_indices.len(),
        )
        .unwrap();

        let mut seen = vec![false; candidates.len()];
        for (i, &tri) in expanded.iter().enumerate().take(num_tris) {
            let cand_idx = built.emitted_candidate_indices[i];
            assert!(!seen[cand_idx], "duplicate candidate {cand_idx}");
            seen[cand_idx] = true;

            let remap = built.triangle_remap[i];
            assert_eq!(remap.input_triangle, candidates[cand_idx].input_triangle);
            assert!(
                CYCLIC_ROTATIONS.contains(&remap.input_corners),
                "non-cyclic corner remap {:?}",
                remap.input_corners
            );

            let decoded_src = [
                built.local_to_source_vertex[usize::from(tri[0])],
                built.local_to_source_vertex[usize::from(tri[1])],
                built.local_to_source_vertex[usize::from(tri[2])],
            ];
            let expected_src = apply_rotation(candidates[cand_idx].vertices, remap.input_corners);
            assert_eq!(decoded_src, expected_src);
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn strip_builder_uses_edge_continuations_and_backtrack_on_fan_and_grid() {
        // Center triangle [0, 1, 2] surrounded on all 3 edges -> forces BACKTRACK!
        let tris = [
            [0, 1, 2],
            [2, 1, 3], // across Edge1 of tri 0
            [0, 2, 4], // across Edge2 of tri 0
            [1, 0, 5], // across Edge0 of tri 0
        ];
        let cands: Vec<_> = tris
            .iter()
            .enumerate()
            .map(|(i, &v)| CandidateTriangle {
                input_triangle: i as u32,
                candidate_index: i,
                vertices: v,
            })
            .collect();

        let built = build_optimized_strip(&cands).unwrap();
        verify_strip_preserves_winding(&cands, &built);
        // Should need only 1 RESTART (or at most 2) and use EDGE1/EDGE2/BACKTRACK!
        let restarts = built
            .topology
            .controls
            .iter()
            .filter(|&&c| c == TriControl::Restart)
            .count();
        assert!(
            restarts <= 2,
            "expected strip reuse, got {:?}",
            built.topology.controls
        );
        assert!(
            built
                .topology
                .controls
                .iter()
                .any(|&c| matches!(c, TriControl::Edge1 | TriControl::Edge2))
        );
    }

    #[test]
    fn strip_builder_handles_mixed_winding_and_non_manifold_safely() {
        let tris = [
            [0, 1, 2],
            [1, 2, 3], // reversed winding across shared edge {1,2} -> must restart
            [2, 1, 4], // compatible across {1,2}
            [2, 1, 5], // non-manifold second wing across {1,2}
            [0, 1, 2], // duplicate face!
        ];
        let cands: Vec<_> = tris
            .iter()
            .enumerate()
            .map(|(i, &v)| CandidateTriangle {
                input_triangle: i as u32,
                candidate_index: i,
                vertices: v,
            })
            .collect();

        let built = build_optimized_strip(&cands).unwrap();
        verify_strip_preserves_winding(&cands, &built);
    }
}
