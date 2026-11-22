//! Baking extracted OpenUSD meshes into DGF1 blocks with remapped sideband attributes and provenance.

use std::collections::BTreeMap;
use std::sync::Arc;

use dgf::{
    DecodeLimits, DecodedMesh, EncodeOptions, EncodedMesh, MeshView, TriangleRemap, decode_mesh,
    encode_mesh, remap_face_attributes, remap_vertex_attributes,
};
use openusd_schemas::geom::Interpolation;

use crate::cache::{ContentHash, compute_content_hash};
use crate::error::UsdDgfError;
use crate::extract::{
    ExtractedAttribute, ExtractedUsdMesh, UsdAttributeValues, UsdExtractOptions, UsdMaterialSubset,
    UsdSceneMetadata,
};

/// Combined options for extracting a USD mesh and encoding it to DGF1.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UsdBakeOptions {
    /// USD extraction and validation options.
    pub extract: UsdExtractOptions,
    /// Underlying DGF1 encoder options.
    pub encode: EncodeOptions,
}

/// Composed provenance linking an emitted DGF1 output triangle back to both the extracted input
/// triangle and the original USD face and `faceVertexIndices` corner slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BakedTriangleProvenance {
    /// Original USD face index.
    pub usd_face: u32,
    /// Original indices into USD `faceVertexIndices` for each output corner `[0, 1, 2]`.
    pub usd_face_corners: [u32; 3],
    /// Extracted triangle index passed to `dgf::encode_mesh`.
    pub input_triangle: u32,
    /// Cyclic corner permutation applied by the DGF1 strip builder.
    pub input_corners: [u8; 3],
}

/// Sideband attribute data remapped to match DGF1 output block/vertex/triangle/corner ordering.
#[derive(Debug, Clone, PartialEq)]
pub enum RemappedAttributeValues {
    /// Single mesh-wide constant value.
    Constant(UsdAttributeValues),
    /// One value per emitted DGF1 triangle (in output block/triangle order).
    PerTriangle(UsdAttributeValues),
    /// One value per stored DGF1 block vertex (in concatenated block-vertex order).
    PerBlockVertex(UsdAttributeValues),
    /// Flat per-corner values (`3 * num_output_triangles`, preserving face-varying UV seams).
    PerTriangleCorner(UsdAttributeValues),
}

/// A USD normal or primvar preserved in sideband storage aligned to DGF1 output order.
#[derive(Debug, Clone, PartialEq)]
pub struct RemappedUsdAttribute {
    /// Attribute or primvar name.
    pub name: String,
    /// Original USD interpolation before remapping.
    pub source_interpolation: Interpolation,
    /// Remapped typed values.
    pub values: RemappedAttributeValues,
}

/// Baked DGF1 geometry plus remapped USD sideband attributes, subset bindings, and scene metadata.
///
/// Independent of world transform so multiple instances can share one `Arc<BakedUsdMesh>`.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedUsdMesh {
    /// Content hash of resolved geometry, attributes, subsets, and encoding options.
    pub content_hash: ContentHash,
    /// Encoded DGF1 blocks, remaps, quantization report, and packing statistics.
    pub encoded: EncodedMesh,
    /// Composed USD face/corner provenance for every emitted DGF1 triangle.
    pub triangle_provenance: Vec<BakedTriangleProvenance>,
    /// Material subsets corresponding to DGF1 `geometry_id` `1..=N`.
    pub material_subsets: Vec<UsdMaterialSubset>,
    /// Material bound directly on the owning mesh prim (for `geometry_id == 0`).
    pub default_material_path: Option<String>,
    /// Remapped normals attribute, if authored and extracted.
    pub normals: Option<RemappedUsdAttribute>,
    /// Remapped primvars (`st`, `displayColor`, etc.) keyed by primvar name.
    pub primvars: BTreeMap<String, RemappedUsdAttribute>,
    /// Orientation, double-sided flag, stage up-axis, and stage meters-per-unit.
    pub scene_metadata: UsdSceneMetadata,
}

impl BakedUsdMesh {
    /// Decodes the embedded DGF1 blocks into a concatenated [`DecodedMesh`].
    pub fn decode_mesh(&self, limits: &DecodeLimits) -> Result<DecodedMesh, UsdDgfError> {
        let bytes = self.encoded.to_bytes();
        Ok(decode_mesh(&bytes, limits)?)
    }

    /// Resolves the USD material path for a decoded DGF1 `geometry_id`.
    ///
    /// Returns the matching subset's bound material (or the mesh's default material when
    /// `geometry_id == 0` or the subset has no override binding).
    pub fn material_for_geometry_id(&self, geometry_id: u32) -> Option<&str> {
        if geometry_id == 0 {
            return self.default_material_path.as_deref();
        }
        self.material_subsets
            .iter()
            .find(|s| s.geometry_id == geometry_id)
            .and_then(|s| s.material_path.as_deref())
            .or(self.default_material_path.as_deref())
    }
}

/// A placed USD mesh instance referencing shared baked DGF1 geometry plus its world transform.
#[derive(Debug, Clone, PartialEq)]
pub struct UsdMeshInstance {
    /// USD prim path of this mesh instance.
    pub prim_path: String,
    /// Row-major local-to-world 4x4 transform matrix (`gf::Mat4d.0`).
    pub world_transform: [f64; 16],
    /// Shared baked DGF1 geometry (reused across identical resolved meshes / instances).
    pub geometry: Arc<BakedUsdMesh>,
}

/// Encodes an [`ExtractedUsdMesh`] into a [`BakedUsdMesh`], composing USD face/corner maps with
/// DGF1 triangle/vertex remaps and remapping all extracted normals and primvars.
pub fn bake_extracted_mesh(
    extracted: &ExtractedUsdMesh,
    encode_options: &EncodeOptions,
) -> Result<BakedUsdMesh, UsdDgfError> {
    let content_hash = compute_content_hash(extracted, encode_options);
    let view = MeshView {
        positions: &extracted.positions,
        triangles: &extracted.triangles,
        triangle_metadata: Some(&extracted.triangle_metadata),
        triangle_omm_ids: None,
    };

    let encoded = encode_mesh(view, encode_options)?;

    // Compose source USD face/corner maps with DGF1 triangle/corner remap.
    let mut triangle_provenance = Vec::with_capacity(encoded.triangle_remap.len());
    for remap in &encoded.triangle_remap {
        let t = remap.input_triangle as usize;
        let src_corners = extracted.source_face_corners[t];
        let composed_corners = [
            src_corners[usize::from(remap.input_corners[0])],
            src_corners[usize::from(remap.input_corners[1])],
            src_corners[usize::from(remap.input_corners[2])],
        ];
        triangle_provenance.push(BakedTriangleProvenance {
            usd_face: extracted.source_faces[t],
            usd_face_corners: composed_corners,
            input_triangle: remap.input_triangle,
            input_corners: remap.input_corners,
        });
    }

    let normals = match &extracted.normals {
        Some(attr) => Some(remap_extracted_attribute(
            attr,
            &encoded.vertex_remap,
            &encoded.triangle_remap,
            &triangle_provenance,
        )?),
        None => None,
    };

    let mut primvars = BTreeMap::new();
    for (name, attr) in &extracted.primvars {
        primvars.insert(
            name.clone(),
            remap_extracted_attribute(
                attr,
                &encoded.vertex_remap,
                &encoded.triangle_remap,
                &triangle_provenance,
            )?,
        );
    }

    Ok(BakedUsdMesh {
        content_hash,
        encoded,
        triangle_provenance,
        material_subsets: extracted.material_subsets.clone(),
        default_material_path: extracted.default_material_path.clone(),
        normals,
        primvars,
        scene_metadata: extracted.scene_metadata,
    })
}

fn remap_extracted_attribute(
    attr: &ExtractedAttribute,
    vertex_remap: &[u32],
    triangle_remap: &[TriangleRemap],
    provenance: &[BakedTriangleProvenance],
) -> Result<RemappedUsdAttribute, UsdDgfError> {
    let values = match attr.interpolation {
        Interpolation::Constant => RemappedAttributeValues::Constant(attr.values.clone()),
        Interpolation::Uniform => {
            RemappedAttributeValues::PerTriangle(remap_uniform(&attr.values, triangle_remap)?)
        }
        Interpolation::Vertex | Interpolation::Varying => {
            RemappedAttributeValues::PerBlockVertex(remap_vertex(&attr.values, vertex_remap)?)
        }
        Interpolation::FaceVarying => {
            RemappedAttributeValues::PerTriangleCorner(remap_face_varying(&attr.values, provenance))
        }
    };

    Ok(RemappedUsdAttribute {
        name: attr.name.clone(),
        source_interpolation: attr.interpolation,
        values,
    })
}

fn remap_vertex(
    values: &UsdAttributeValues,
    vertex_remap: &[u32],
) -> Result<UsdAttributeValues, UsdDgfError> {
    Ok(match values {
        UsdAttributeValues::Vec2f(v) => {
            UsdAttributeValues::Vec2f(remap_vertex_attributes(v, vertex_remap)?)
        }
        UsdAttributeValues::Vec3f(v) => {
            UsdAttributeValues::Vec3f(remap_vertex_attributes(v, vertex_remap)?)
        }
        UsdAttributeValues::Vec4f(v) => {
            UsdAttributeValues::Vec4f(remap_vertex_attributes(v, vertex_remap)?)
        }
        UsdAttributeValues::Float(v) => {
            UsdAttributeValues::Float(remap_vertex_attributes(v, vertex_remap)?)
        }
        UsdAttributeValues::Int(v) => {
            UsdAttributeValues::Int(remap_vertex_attributes(v, vertex_remap)?)
        }
    })
}

fn remap_uniform(
    values: &UsdAttributeValues,
    triangle_remap: &[TriangleRemap],
) -> Result<UsdAttributeValues, UsdDgfError> {
    Ok(match values {
        UsdAttributeValues::Vec2f(v) => {
            UsdAttributeValues::Vec2f(remap_face_attributes(v, triangle_remap)?)
        }
        UsdAttributeValues::Vec3f(v) => {
            UsdAttributeValues::Vec3f(remap_face_attributes(v, triangle_remap)?)
        }
        UsdAttributeValues::Vec4f(v) => {
            UsdAttributeValues::Vec4f(remap_face_attributes(v, triangle_remap)?)
        }
        UsdAttributeValues::Float(v) => {
            UsdAttributeValues::Float(remap_face_attributes(v, triangle_remap)?)
        }
        UsdAttributeValues::Int(v) => {
            UsdAttributeValues::Int(remap_face_attributes(v, triangle_remap)?)
        }
    })
}

fn remap_face_varying(
    values: &UsdAttributeValues,
    provenance: &[BakedTriangleProvenance],
) -> UsdAttributeValues {
    fn gather<T: Copy>(src: &[T], prov: &[BakedTriangleProvenance]) -> Vec<T> {
        let mut out = Vec::with_capacity(prov.len() * 3);
        for p in prov {
            for &c in &p.usd_face_corners {
                out.push(src[c as usize]);
            }
        }
        out
    }

    match values {
        UsdAttributeValues::Vec2f(v) => UsdAttributeValues::Vec2f(gather(v, provenance)),
        UsdAttributeValues::Vec3f(v) => UsdAttributeValues::Vec3f(gather(v, provenance)),
        UsdAttributeValues::Vec4f(v) => UsdAttributeValues::Vec4f(gather(v, provenance)),
        UsdAttributeValues::Float(v) => UsdAttributeValues::Float(gather(v, provenance)),
        UsdAttributeValues::Int(v) => UsdAttributeValues::Int(gather(v, provenance)),
    }
}
