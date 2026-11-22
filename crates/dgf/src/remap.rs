//! Explicit triangle, corner, and vertex remap metadata plus attribute/mesh reconstruction helpers.

use crate::decode::DecodedMesh;
use crate::error::RemapError;

/// Maps an emitted output triangle back to its source triangle and cyclic corner permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TriangleRemap {
    /// Original source triangle index in the input mesh.
    pub input_triangle: u32,
    /// For each output corner `c` in `0..3`, the corresponding source corner index (`0`, `1`, or `2`).
    ///
    /// Winding-preserving encoders only emit cyclic permutations: `[0, 1, 2]`, `[1, 2, 0]`, or `[2, 0, 1]`.
    pub input_corners: [u8; 3],
}

/// Describes where a single encoded block's vertices and triangles sit in the concatenated remap arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockRange {
    /// Starting index in `vertex_remap` (and concatenated decoded vertex buffer).
    pub vertex_offset: usize,
    /// Number of vertices stored in this block.
    pub vertex_count: usize,
    /// Starting index in `triangle_remap` (and concatenated decoded triangle buffer).
    pub triangle_offset: usize,
    /// Number of triangles stored in this block.
    pub triangle_count: usize,
    /// Optional input cluster ID if encoded via coarse clustering or preclustered input.
    pub input_cluster: Option<usize>,
    /// Optional axis-aligned bounding box `(min, max)` of this block's quantized decoded positions.
    pub aabb: Option<([f32; 3], [f32; 3])>,
}

/// Sideband mapping from a block's local OMM descriptor indices (`0..D`) to caller OMM identities.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockOmmMap {
    /// Caller-supplied OMM identities indexed by local descriptor index (`len <= 7`).
    pub descriptors: Vec<u32>,
}

/// Globally indexed mesh reconstructed from a [`DecodedMesh`] and verified remap tables.
#[derive(Debug, Clone, PartialEq)]
pub struct ReconstructedMesh {
    /// Reconstructed positions indexed by original source vertex ID (`None` for unused source vertices).
    pub positions: Vec<Option<[f32; 3]>>,
    /// Triangles re-indexed into source vertex IDs, in decoded output triangle order.
    pub triangles: Vec<[u32; 3]>,
    /// Per-triangle geometry IDs in decoded output triangle order.
    pub geometry_ids: Vec<u32>,
    /// Per-triangle opaque flags in decoded output triangle order.
    pub opaque: Vec<bool>,
}

/// Remaps per-vertex attributes (e.g., vertex normals or skin weights) from source vertex order
/// to concatenated decoded block-vertex order using `vertex_remap`.
pub fn remap_vertex_attributes<T: Clone>(
    source: &[T],
    vertex_remap: &[u32],
) -> Result<Vec<T>, RemapError> {
    let mut out = Vec::with_capacity(vertex_remap.len());
    for (decoded_vertex, &source_vertex) in vertex_remap.iter().enumerate() {
        let idx = source_vertex as usize;
        let val = source.get(idx).ok_or(RemapError::SourceVertexOutOfBounds {
            decoded_vertex,
            source_vertex,
            source_len: source.len(),
        })?;
        out.push(val.clone());
    }
    Ok(out)
}

/// Remaps per-triangle / face attributes (e.g., material IDs or face normals) from source triangle
/// order to decoded output triangle order using `triangle_remap`.
pub fn remap_face_attributes<T: Clone>(
    source: &[T],
    triangle_remap: &[TriangleRemap],
) -> Result<Vec<T>, RemapError> {
    let mut out = Vec::with_capacity(triangle_remap.len());
    for (decoded_triangle, entry) in triangle_remap.iter().enumerate() {
        let idx = entry.input_triangle as usize;
        let val = source
            .get(idx)
            .ok_or(RemapError::SourceTriangleOutOfBounds {
                decoded_triangle,
                source_triangle: entry.input_triangle,
                source_len: source.len(),
            })?;
        out.push(val.clone());
    }
    Ok(out)
}

/// Remaps per-corner / face-varying attributes (such as USD face-varying UVs) stored as `[T; 3]`
/// per source triangle into decoded triangle and rotated corner order.
pub fn remap_corner_attributes<T: Clone>(
    source_corners: &[[T; 3]],
    triangle_remap: &[TriangleRemap],
) -> Result<Vec<[T; 3]>, RemapError> {
    let mut out = Vec::with_capacity(triangle_remap.len());
    for (decoded_triangle, entry) in triangle_remap.iter().enumerate() {
        validate_corners(decoded_triangle, entry.input_corners)?;
        let idx = entry.input_triangle as usize;
        let tri_attrs = source_corners
            .get(idx)
            .ok_or(RemapError::SourceTriangleOutOfBounds {
                decoded_triangle,
                source_triangle: entry.input_triangle,
                source_len: source_corners.len(),
            })?;
        out.push([
            tri_attrs[usize::from(entry.input_corners[0])].clone(),
            tri_attrs[usize::from(entry.input_corners[1])].clone(),
            tri_attrs[usize::from(entry.input_corners[2])].clone(),
        ]);
    }
    Ok(out)
}

/// Remaps flat per-corner attributes (`3 * num_source_triangles` elements) into flat decoded corner order.
pub fn remap_flat_corner_attributes<T: Clone>(
    source_flat_corners: &[T],
    triangle_remap: &[TriangleRemap],
) -> Result<Vec<T>, RemapError> {
    let mut out = Vec::with_capacity(triangle_remap.len() * 3);
    for (decoded_triangle, entry) in triangle_remap.iter().enumerate() {
        validate_corners(decoded_triangle, entry.input_corners)?;
        let base = (entry.input_triangle as usize).saturating_mul(3);
        if base + 3 > source_flat_corners.len() {
            return Err(RemapError::SourceTriangleOutOfBounds {
                decoded_triangle,
                source_triangle: entry.input_triangle,
                source_len: source_flat_corners.len() / 3,
            });
        }
        for &c in &entry.input_corners {
            out.push(source_flat_corners[base + usize::from(c)].clone());
        }
    }
    Ok(out)
}

/// Reconstructs a globally indexed mesh from a [`DecodedMesh`] using verified sideband remap arrays.
///
/// Explicitly verifies that every block-local copy mapped to the same source vertex decoded to the
/// exact same floating-point coordinates (bit-for-bit). Does **not** merge distinct source vertices
/// that happen to share coordinates.
pub fn reconstruct_with_remap(
    decoded: &DecodedMesh,
    vertex_remap: &[u32],
    triangle_remap: &[TriangleRemap],
    num_source_vertices: usize,
) -> Result<ReconstructedMesh, RemapError> {
    if decoded.positions.len() != vertex_remap.len() {
        return Err(RemapError::VertexCountMismatch {
            decoded: decoded.positions.len(),
            remap: vertex_remap.len(),
        });
    }
    if decoded.triangles.len() != triangle_remap.len() {
        return Err(RemapError::TriangleCountMismatch {
            decoded: decoded.triangles.len(),
            remap: triangle_remap.len(),
        });
    }

    let mut positions: Vec<Option<[f32; 3]>> = vec![None; num_source_vertices];
    for (decoded_vertex, (&pos, &source_vertex)) in decoded
        .positions
        .iter()
        .zip(vertex_remap.iter())
        .enumerate()
    {
        let slot = positions.get_mut(source_vertex as usize).ok_or(
            RemapError::SourceVertexOutOfBounds {
                decoded_vertex,
                source_vertex,
                source_len: num_source_vertices,
            },
        )?;
        match *slot {
            None => *slot = Some(pos),
            Some(existing) => {
                if existing[0].to_bits() != pos[0].to_bits()
                    || existing[1].to_bits() != pos[1].to_bits()
                    || existing[2].to_bits() != pos[2].to_bits()
                {
                    return Err(RemapError::InconsistentVertexPosition {
                        source_vertex,
                        first: existing,
                        conflicting: pos,
                    });
                }
            }
        }
    }

    let mut triangles = Vec::with_capacity(decoded.triangles.len());
    for (tri_idx, (&tri, remap_entry)) in decoded
        .triangles
        .iter()
        .zip(triangle_remap.iter())
        .enumerate()
    {
        validate_corners(tri_idx, remap_entry.input_corners)?;
        let mut src_tri = [0u32; 3];
        for (c, &dv) in tri.iter().enumerate() {
            let src_v =
                *vertex_remap
                    .get(dv as usize)
                    .ok_or(RemapError::DecodedIndexOutOfBounds {
                        triangle: tri_idx,
                        vertex: dv,
                        num_vertices: decoded.positions.len(),
                    })?;
            src_tri[c] = src_v;
        }
        triangles.push(src_tri);
    }

    Ok(ReconstructedMesh {
        positions,
        triangles,
        geometry_ids: decoded.geometry_ids.clone(),
        opaque: decoded.opaque.clone(),
    })
}

fn validate_corners(decoded_triangle: usize, corners: [u8; 3]) -> Result<(), RemapError> {
    let valid_mask =
        (1u8 << corners[0].min(3)) | (1u8 << corners[1].min(3)) | (1u8 << corners[2].min(3));
    if corners.iter().any(|&c| c >= 3) || valid_mask != 0b111 {
        return Err(RemapError::InvalidCornerMap {
            decoded_triangle,
            corners,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_detects_inconsistent_shared_vertex() {
        let decoded = DecodedMesh {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            triangles: vec![[0, 1, 2]],
            geometry_ids: vec![0],
            opaque: vec![true],
            prim_ids: vec![0],
            block_user_data: vec![None],
            block_vertex_offsets: vec![0],
            block_triangle_offsets: vec![0],
        };
        let vertex_remap = vec![0, 1, 2, 0]; // vertex 3 maps to source 0 with different position!
        let tri_remap = vec![TriangleRemap {
            input_triangle: 0,
            input_corners: [0, 1, 2],
        }];
        let err = reconstruct_with_remap(&decoded, &vertex_remap, &tri_remap, 3).unwrap_err();
        assert_eq!(
            err,
            RemapError::InconsistentVertexPosition {
                source_vertex: 0,
                first: [0.0, 0.0, 0.0],
                conflicting: [0.5, 0.0, 0.0],
            }
        );
    }

    #[test]
    fn corner_remap_preserves_face_varying_seams() {
        let uvs = vec![[[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]]];
        let remap = vec![TriangleRemap {
            input_triangle: 0,
            input_corners: [1, 2, 0],
        }];
        let out = remap_corner_attributes(&uvs, &remap).unwrap();
        assert_eq!(out, vec![[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]);
    }
}
