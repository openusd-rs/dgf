//! Extraction and independent topology validation of resolved `UsdGeomMesh` prims.

use std::collections::BTreeMap;

use dgf::TriangleMetadata;
use openusd::usd::{self, SchemaBase};
use openusd::{gf, vt};
use openusd_schemas::geom::{
    self, ElementType, GprimSchema, Interpolation, MeshSchema, Orientation, PointBasedSchema,
    PrimvarsAPI, SubdivisionScheme, Subset, SubsetSchema, UpAxis, XformCache,
    stage_meters_per_unit, stage_up_axis,
};
use openusd_schemas::shade::MaterialBindingAPI;

use crate::error::UsdDgfError;

/// Options controlling how a `UsdGeomMesh` prim is extracted at a given time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsdExtractOptions {
    /// Require `subdivisionScheme == "none"` (default `true`).
    pub require_subdivision_none: bool,
    /// Extract authored `normals` into sideband data (default `true`).
    pub extract_normals: bool,
    /// Extract authored/inherited supported primvars (`st`, `displayColor`, etc.) (default `true`).
    pub extract_primvars: bool,
    /// Material binding purpose passed to `MaterialBindingAPI` (`""` = allPurpose).
    pub material_purpose: String,
}

impl Default for UsdExtractOptions {
    fn default() -> Self {
        Self {
            require_subdivision_none: true,
            extract_normals: true,
            extract_primvars: true,
            material_purpose: String::new(),
        }
    }
}

/// Scene- and prim-level USD metadata preserved alongside encoded DGF1 blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UsdSceneMetadata {
    /// Mesh face winding convention (`RightHanded` or `LeftHanded`).
    pub orientation: Orientation,
    /// Whether the prim is authored as `doubleSided`.
    pub double_sided: bool,
    /// Stage `upAxis` (`Y` or `Z`).
    pub up_axis: UpAxis,
    /// Stage `metersPerUnit` scale factor.
    pub meters_per_unit: f64,
}

/// Face material subset metadata corresponding to a DGF1 `geometry_id` (`1..=N`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UsdMaterialSubset {
    /// DGF1 `geometry_id` assigned to faces in this subset (`1..=N`; `0` is default/unbound).
    pub geometry_id: u32,
    /// Subset prim name (e.g., `"Roof"`).
    pub name: String,
    /// Full USD prim path of the `GeomSubset`.
    pub path: String,
    /// Subset `familyName` token (e.g., `"materialBind"`).
    pub family_name: String,
    /// Resolved bound material path for this subset, if any.
    pub material_path: Option<String>,
}

/// Typed array payload extracted from USD normals or primvars.
#[derive(Debug, Clone, PartialEq)]
pub enum UsdAttributeValues {
    /// 2D floating-point vectors (e.g., UVs / `primvars:st`).
    Vec2f(Vec<[f32; 2]>),
    /// 3D floating-point vectors (e.g., normals, `displayColor`).
    Vec3f(Vec<[f32; 3]>),
    /// 4D floating-point vectors (e.g., tangents, RGBA colors).
    Vec4f(Vec<[f32; 4]>),
    /// Scalar floats (e.g., `displayOpacity`, weights).
    Float(Vec<f32>),
    /// Scalar signed 32-bit integers.
    Int(Vec<i32>),
}

impl UsdAttributeValues {
    /// Returns the number of elements stored in this attribute array.
    pub fn len(&self) -> usize {
        match self {
            Self::Vec2f(v) => v.len(),
            Self::Vec3f(v) => v.len(),
            Self::Vec4f(v) => v.len(),
            Self::Float(v) => v.len(),
            Self::Int(v) => v.len(),
        }
    }

    /// Returns `true` if this attribute array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Resolved USD attribute or primvar flattened over its declared interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedAttribute {
    /// Attribute or primvar name (without the `primvars:` prefix for primvars).
    pub name: String,
    /// USD interpolation rule (`Constant`, `Uniform`, `Varying`, `Vertex`, or `FaceVarying`).
    pub interpolation: Interpolation,
    /// Flattened typed values.
    pub values: UsdAttributeValues,
}

/// Resolved, topology-validated USD triangle mesh ready for DGF1 baking and content hashing.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedUsdMesh {
    /// USD prim path of this mesh.
    pub prim_path: String,
    /// Local-space vertex positions (`points`).
    pub positions: Vec<[f32; 3]>,
    /// Validated `faceVertexCounts` (each `== 3`).
    pub face_vertex_counts: Vec<i32>,
    /// Validated `faceVertexIndices`.
    pub face_vertex_indices: Vec<i32>,
    /// Triangle vertex indices ready for [`dgf::MeshView`].
    pub triangles: Vec<[u32; 3]>,
    /// Original USD face index for each extracted triangle.
    pub source_faces: Vec<u32>,
    /// Original index into `faceVertexIndices` for each corner `[0, 1, 2]` of each triangle.
    pub source_face_corners: Vec<[u32; 3]>,
    /// Per-triangle DGF1 metadata (`geometry_id` derived from material subsets).
    pub triangle_metadata: Vec<TriangleMetadata>,
    /// Direct/resolved material bound to the mesh prim itself (used for `geometry_id == 0`).
    pub default_material_path: Option<String>,
    /// Material subsets mapped to `geometry_id` `1..=N`.
    pub material_subsets: Vec<UsdMaterialSubset>,
    /// Optional resolved `normals` attribute.
    pub normals: Option<ExtractedAttribute>,
    /// Resolved supported primvars keyed deterministically by primvar name.
    pub primvars: BTreeMap<String, ExtractedAttribute>,
    /// Orientation, double-sided flag, stage up-axis, and stage meters-per-unit.
    pub scene_metadata: UsdSceneMetadata,
    /// Row-major local-to-world 4x4 transform matrix (`gf::Mat4d.0`) at the queried time.
    /// Kept separate from geometry hashing so transform edits do not invalidate baked DGF blocks.
    pub world_transform: [f64; 16],
}

/// Extracts and independently validates a resolved `UsdGeomMesh` at `time`.
///
/// Validates `faceVertexCounts` independently (rejecting negative counts such as `[4, -1]`
/// that `UsdGeomMesh::validate_topology` intentionally allows when sums cancel), rejects
/// non-triangle faces, and checks `subdivisionScheme == "none"` when requested.
pub fn extract_usd_mesh(
    mesh: &geom::Mesh,
    time: impl Into<Option<usd::TimeCode>>,
    xform_cache: &mut XformCache,
    options: &UsdExtractOptions,
) -> Result<ExtractedUsdMesh, UsdDgfError> {
    let time = time.into();
    let prim = mesh.prim();
    let prim_path = prim.path().to_string();
    let stage = prim.stage();

    // 1. Check subdivisionScheme == none when required.
    let scheme = mesh
        .subdivision_scheme_attr()
        .get::<SubdivisionScheme>()?
        .unwrap_or_default();
    if options.require_subdivision_none && scheme != SubdivisionScheme::None {
        return Err(UsdDgfError::UnsupportedSubdivisionScheme {
            prim_path,
            scheme: scheme.as_token().to_string(),
        });
    }

    // 2. Read points, faceVertexCounts, faceVertexIndices at explicit time.
    let raw_points: Vec<gf::Vec3f> = mesh.points_at(time)?.unwrap_or_default();
    let positions: Vec<[f32; 3]> = raw_points.into_iter().map(|p| [p[0], p[1], p[2]]).collect();
    let face_vertex_counts: Vec<i32> = mesh.face_vertex_counts_at(time)?.unwrap_or_default();
    let face_vertex_indices: Vec<i32> = mesh.face_vertex_indices_at(time)?.unwrap_or_default();

    // 3. Validate faceVertexCounts independently (reject negative counts and non-triangle faces!).
    let mut checked_sum = 0usize;
    for (face, &count) in face_vertex_counts.iter().enumerate() {
        if count < 0 {
            return Err(UsdDgfError::NegativeFaceVertexCount {
                prim_path,
                face,
                count,
            });
        }
        if count != 3 {
            return Err(UsdDgfError::NonTriangleFace {
                prim_path,
                face,
                vertex_count: count,
            });
        }
        checked_sum = checked_sum.checked_add(count as usize).ok_or(
            UsdDgfError::FaceVertexCountSumMismatch {
                prim_path: prim_path.clone(),
                counts_sum: usize::MAX,
                indices_len: face_vertex_indices.len(),
            },
        )?;
    }
    if checked_sum != face_vertex_indices.len() {
        return Err(UsdDgfError::FaceVertexCountSumMismatch {
            prim_path,
            counts_sum: checked_sum,
            indices_len: face_vertex_indices.len(),
        });
    }

    let num_faces = face_vertex_counts.len();
    let num_points = positions.len();
    let mut triangles = Vec::with_capacity(num_faces);
    let mut source_faces = Vec::with_capacity(num_faces);
    let mut source_face_corners = Vec::with_capacity(num_faces);

    for face in 0..num_faces {
        let base = face * 3;
        let mut tri = [0u32; 3];
        for corner in 0..3 {
            let raw_idx = face_vertex_indices[base + corner];
            if raw_idx < 0 || (raw_idx as usize) >= num_points {
                return Err(UsdDgfError::FaceVertexIndexOutOfRange {
                    prim_path,
                    face,
                    corner,
                    index: raw_idx,
                    num_points,
                });
            }
            tri[corner] = raw_idx as u32;
        }
        triangles.push(tri);
        source_faces.push(face as u32);
        source_face_corners.push([base as u32, (base + 1) as u32, (base + 2) as u32]);
    }

    // 4. Scene metadata (orientation, doubleSided, upAxis, metersPerUnit) and world transform.
    let orientation = mesh
        .orientation_attr()
        .get::<Orientation>()?
        .unwrap_or_default();
    let double_sided = mesh.double_sided_attr().get::<bool>()?.unwrap_or(false);
    let up_axis = stage_up_axis(stage)?;
    let meters_per_unit = stage_meters_per_unit(stage)?;
    let scene_metadata = UsdSceneMetadata {
        orientation,
        double_sided,
        up_axis,
        meters_per_unit,
    };

    xform_cache.set_time(time);
    let world_transform = xform_cache.local_to_world_transform(prim)?.0;

    // 5. Material bindings & face subsets -> per-triangle geometry_id.
    let mesh_binding_api = MaterialBindingAPI::from_prim_unchecked(prim.clone());
    let default_material_path =
        resolve_prim_material_path(&mesh_binding_api, &options.material_purpose)?;

    let mut triangle_metadata = vec![
        TriangleMetadata {
            geometry_id: 0,
            opaque: true,
        };
        num_faces
    ];
    let mut material_subsets = Vec::new();

    // Prefer `materialBind` subsets; if none exist, fall back to any Face GeomSubsets.
    let mut subsets = mesh_binding_api.subsets()?;
    if subsets.is_empty() {
        subsets = Subset::all(mesh)?;
    }

    let mut next_geom_id = 1u32;
    for subset in subsets {
        let elem_type = subset
            .element_type_attr()
            .get::<ElementType>()?
            .unwrap_or_default();
        if elem_type != ElementType::Face {
            continue;
        }
        let subset_indices = subset.indices_at(time)?.unwrap_or_default();
        let subset_path = subset.prim().path().to_string();
        let subset_name = subset_path
            .rsplit('/')
            .next()
            .unwrap_or(&subset_path)
            .to_string();
        let family_name = subset
            .family_name_attr()
            .get::<openusd::tf::Token>()?
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();
        let subset_binding_api = MaterialBindingAPI::from_prim_unchecked(subset.prim().clone());
        let material_path =
            resolve_prim_material_path(&subset_binding_api, &options.material_purpose)?;

        let gid = next_geom_id;
        next_geom_id += 1;

        for &face_idx in &subset_indices {
            if face_idx < 0 || (face_idx as usize) >= num_faces {
                return Err(UsdDgfError::SubsetFaceOutOfRange {
                    prim_path,
                    subset_path,
                    face: face_idx,
                    num_faces,
                });
            }
            triangle_metadata[face_idx as usize].geometry_id = gid;
        }

        material_subsets.push(UsdMaterialSubset {
            geometry_id: gid,
            name: subset_name,
            path: subset_path,
            family_name,
            material_path,
        });
    }

    // 6. Normals (optional).
    let normals = if options.extract_normals {
        extract_normals(
            mesh,
            time,
            &prim_path,
            num_points,
            num_faces,
            face_vertex_indices.len(),
        )?
    } else {
        None
    };

    // 7. Primvars (optional, supporting indexed & inherited constant primvars).
    let primvars = if options.extract_primvars {
        extract_primvars(
            prim,
            time,
            &prim_path,
            num_points,
            num_faces,
            face_vertex_indices.len(),
        )?
    } else {
        BTreeMap::new()
    };

    Ok(ExtractedUsdMesh {
        prim_path,
        positions,
        face_vertex_counts,
        face_vertex_indices,
        triangles,
        source_faces,
        source_face_corners,
        triangle_metadata,
        default_material_path,
        material_subsets,
        normals,
        primvars,
        scene_metadata,
        world_transform,
    })
}

fn resolve_prim_material_path(
    binding_api: &MaterialBindingAPI,
    purpose: &str,
) -> Result<Option<String>, UsdDgfError> {
    if let Some(direct) = binding_api.direct_binding(purpose)? {
        return Ok(Some(direct.material_path().to_string()));
    }
    if !purpose.is_empty()
        && let Some(direct) = binding_api.direct_binding("")?
    {
        return Ok(Some(direct.material_path().to_string()));
    }
    Ok(None)
}

fn extract_normals(
    mesh: &geom::Mesh,
    time: Option<usd::TimeCode>,
    prim_path: &str,
    num_points: usize,
    num_faces: usize,
    num_face_verts: usize,
) -> Result<Option<ExtractedAttribute>, UsdDgfError> {
    let attr = mesh.normals_attr();
    let Some(raw) = mesh.normals_at(time)? else {
        return Ok(None);
    };
    let interpolation = attr
        .metadata::<Interpolation>("interpolation")?
        .unwrap_or(Interpolation::FaceVarying);
    let values: Vec<[f32; 3]> = raw.into_iter().map(|n| [n[0], n[1], n[2]]).collect();
    validate_attribute_len(
        prim_path,
        "normals",
        interpolation,
        values.len(),
        num_points,
        num_faces,
        num_face_verts,
    )?;
    Ok(Some(ExtractedAttribute {
        name: "normals".to_string(),
        interpolation,
        values: UsdAttributeValues::Vec3f(values),
    }))
}

fn extract_primvars(
    prim: &usd::Prim,
    time: Option<usd::TimeCode>,
    prim_path: &str,
    num_points: usize,
    num_faces: usize,
    num_face_verts: usize,
) -> Result<BTreeMap<String, ExtractedAttribute>, UsdDgfError> {
    let api = PrimvarsAPI::from_prim_unchecked(prim.clone());
    let mut out = BTreeMap::new();

    for pv in api.primvars_with_inheritance()? {
        if !pv.attribute().has_authored_value()? {
            continue;
        }
        let name = pv.primvar_name().to_string();
        let interpolation = pv.interpolation()?;
        let Some(raw_val) = pv.flattened::<vt::Value>(time)? else {
            continue;
        };
        let Some(values) = vt_value_to_attribute_values(raw_val) else {
            continue;
        };
        validate_attribute_len(
            prim_path,
            &format!("primvars:{name}"),
            interpolation,
            values.len(),
            num_points,
            num_faces,
            num_face_verts,
        )?;
        out.insert(
            name.clone(),
            ExtractedAttribute {
                name,
                interpolation,
                values,
            },
        );
    }

    Ok(out)
}

fn vt_value_to_attribute_values(value: vt::Value) -> Option<UsdAttributeValues> {
    match value {
        vt::Value::Vec2fVec(v) => Some(UsdAttributeValues::Vec2f(
            v.into_iter().map(|a| [a[0], a[1]]).collect(),
        )),
        vt::Value::Vec3fVec(v) => Some(UsdAttributeValues::Vec3f(
            v.into_iter().map(|a| [a[0], a[1], a[2]]).collect(),
        )),
        vt::Value::Vec4fVec(v) => Some(UsdAttributeValues::Vec4f(
            v.into_iter().map(|a| [a[0], a[1], a[2], a[3]]).collect(),
        )),
        vt::Value::FloatVec(v) => Some(UsdAttributeValues::Float(v)),
        vt::Value::IntVec(v) => Some(UsdAttributeValues::Int(v)),
        vt::Value::Vec2f(a) => Some(UsdAttributeValues::Vec2f(vec![[a[0], a[1]]])),
        vt::Value::Vec3f(a) => Some(UsdAttributeValues::Vec3f(vec![[a[0], a[1], a[2]]])),
        vt::Value::Vec4f(a) => Some(UsdAttributeValues::Vec4f(vec![[a[0], a[1], a[2], a[3]]])),
        vt::Value::Float(x) => Some(UsdAttributeValues::Float(vec![x])),
        vt::Value::Int(x) => Some(UsdAttributeValues::Int(vec![x])),
        _ => None,
    }
}

fn validate_attribute_len(
    prim_path: &str,
    attribute: &str,
    interpolation: Interpolation,
    found: usize,
    num_points: usize,
    num_faces: usize,
    num_face_verts: usize,
) -> Result<(), UsdDgfError> {
    let expected = match interpolation {
        Interpolation::Constant => 1,
        Interpolation::Uniform => num_faces,
        Interpolation::Vertex | Interpolation::Varying => num_points,
        Interpolation::FaceVarying => num_face_verts,
    };
    if found != expected {
        return Err(UsdDgfError::AttributeLengthMismatch {
            prim_path: prim_path.to_string(),
            attribute: attribute.to_string(),
            interpolation: interpolation.as_token(),
            expected,
            found,
        });
    }
    Ok(())
}
