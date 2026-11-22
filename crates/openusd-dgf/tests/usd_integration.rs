//! Integration tests for `openusd-dgf` over real in-memory OpenUSD stages.

#![forbid(unsafe_code)]

use std::sync::Arc;

use dgf::{DecodeLimits, EncodeOptions, QuantizationMode};
use openusd::usd::{self, SchemaBase};
use openusd::{gf, vt};
use openusd_dgf::{
    CacheStatus, RemappedAttributeValues, UsdAttributeValues, UsdBakeOptions, UsdDgfCache,
    UsdDgfError, UsdExtractOptions, extract_usd_mesh,
};
use openusd_schemas::geom::{
    self, ElementType, GprimSchema, Interpolation, MeshSchema, Orientation, PointBasedSchema,
    PrimvarsAPI, StageMetadata, SubdivisionScheme, UpAxis, XformCache, XformableExt,
    set_stage_up_axis,
};
use openusd_schemas::shade::{self, MaterialBindingAPI};

fn new_stage() -> usd::Stage {
    usd::Stage::builder()
        .schema_registry(openusd_schemas::schema_registry())
        .in_memory("test.usda")
        .expect("create in-memory stage")
}

/// Builds a two-triangle quad mesh with:
/// - `subdivisionScheme = none`
/// - a UV seam (face-varying indexed `primvars:st` where shared edge vertices have DIFFERENT UVs on each triangle!)
/// - vertex normals
/// - two face `GeomSubset`s bound to two distinct `UsdShadeMaterial`s
fn define_quad_with_uv_seam_and_subsets(stage: &usd::Stage, path: &str) -> geom::Mesh {
    let mat_a = shade::Material::define(stage, "/Materials/MatA").unwrap();
    let mat_b = shade::Material::define(stage, "/Materials/MatB").unwrap();

    let mesh = geom::Mesh::define(stage, path).unwrap();
    mesh.create_subdivision_scheme_attr()
        .unwrap()
        .set(SubdivisionScheme::None)
        .unwrap();
    mesh.create_orientation_attr()
        .unwrap()
        .set(Orientation::RightHanded)
        .unwrap();
    mesh.create_double_sided_attr().unwrap().set(true).unwrap();

    // 4 vertices, 2 triangles sharing edge (0, 2)
    mesh.create_points_attr()
        .unwrap()
        .set(vec![
            gf::vec3f(0.0, 0.0, 0.0),
            gf::vec3f(1.0, 0.0, 0.0),
            gf::vec3f(1.0, 1.0, 0.0),
            gf::vec3f(0.0, 1.0, 0.0),
        ])
        .unwrap();
    mesh.create_face_vertex_counts_attr()
        .unwrap()
        .set(vec![3i32, 3i32])
        .unwrap();
    mesh.create_face_vertex_indices_attr()
        .unwrap()
        .set(vec![0i32, 1, 2, 0, 2, 3])
        .unwrap();

    // Vertex normals
    mesh.create_normals_attr()
        .unwrap()
        .set(vec![
            gf::vec3f(0.0, 0.0, 1.0),
            gf::vec3f(0.0, 0.0, 1.0),
            gf::vec3f(0.0, 0.0, 1.0),
            gf::vec3f(0.0, 0.0, 1.0),
        ])
        .unwrap()
        .set_metadata("interpolation", Interpolation::Vertex)
        .unwrap();

    // Face-varying indexed `primvars:st` with an explicit seam along edge (0, 2):
    // Face 0 corners [0, 1, 2] -> UVs [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
    // Face 1 corners [0, 2, 3] -> UVs [(0.5, 0.5), (1.0, 1.0), (0.0, 1.0)] (different UV on vertex 0 and 2!)
    PrimvarsAPI::from_prim_unchecked(mesh.prim().clone())
        .primvar_builder("st", "texCoord2f[]")
        .interpolation(Interpolation::FaceVarying)
        .set(vt::Value::Vec2fVec(vec![
            gf::vec2f(0.0, 0.0), // 0
            gf::vec2f(0.5, 0.0), // 1
            gf::vec2f(0.5, 0.5), // 2
            gf::vec2f(1.0, 1.0), // 3
            gf::vec2f(0.0, 1.0), // 4
        ]))
        .indices(vec![0, 1, 2, 2, 3, 4])
        .build()
        .unwrap();

    // Two face material subsets
    let binding = MaterialBindingAPI::from_prim_unchecked(mesh.prim().clone());
    let sub_a = binding
        .create_subset("TriA", ElementType::Face, &[0])
        .unwrap();
    MaterialBindingAPI::from_prim_unchecked(sub_a.prim().clone())
        .bind(mat_a.path())
        .unwrap();

    let sub_b = binding
        .create_subset("TriB", ElementType::Face, &[1])
        .unwrap();
    MaterialBindingAPI::from_prim_unchecked(sub_b.prim().clone())
        .bind(mat_b.path())
        .unwrap();

    mesh
}

#[test]
fn uv_seam_and_material_subsets_survive_full_usd_dgf_pipeline() {
    let stage = new_stage();
    set_stage_up_axis(&stage, UpAxis::Z).unwrap();
    stage.set_meters_per_unit(1.0).unwrap();

    let mesh = define_quad_with_uv_seam_and_subsets(&stage, "/World/Quad");
    mesh.clone()
        .set_translate(gf::vec3d(10.0, 20.0, 30.0))
        .unwrap();

    let mut cache = UsdDgfCache::new();
    let mut xform_cache = XformCache::default();
    let opts = UsdBakeOptions {
        encode: EncodeOptions {
            quantization: QuantizationMode::MaxCoordError(1e-4),
            ..EncodeOptions::default()
        },
        ..UsdBakeOptions::default()
    };

    let (instance, status) = cache
        .bake_or_get(&mesh, None, &mut xform_cache, &opts)
        .unwrap();
    assert_eq!(status, CacheStatus::Miss);
    assert_eq!(instance.prim_path, "/World/Quad");
    assert_eq!(
        instance.world_transform,
        gf::Mat4d::translation([10.0, 20.0, 30.0]).0
    );

    let baked = &instance.geometry;
    assert_eq!(baked.scene_metadata.up_axis, UpAxis::Z);
    assert_eq!(baked.scene_metadata.meters_per_unit, 1.0);
    assert_eq!(baked.scene_metadata.orientation, Orientation::RightHanded);
    assert!(baked.scene_metadata.double_sided);

    // Decode DGF blocks and verify material subsets + face-varying UV seam survived cyclic corner rotation!
    let decoded = baked.decode_mesh(&DecodeLimits::default()).unwrap();
    assert_eq!(decoded.triangles.len(), 2);

    let expected_face_uvs: [[[f32; 2]; 3]; 2] = [
        [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
        [[0.5, 0.5], [1.0, 1.0], [0.0, 1.0]],
    ];

    let st = baked.primvars.get("st").expect("remapped st primvar");
    let RemappedAttributeValues::PerTriangleCorner(UsdAttributeValues::Vec2f(remapped_uvs)) =
        &st.values
    else {
        panic!("expected PerTriangleCorner Vec2f for st");
    };
    assert_eq!(remapped_uvs.len(), 6);

    for (out_t, prov) in baked.triangle_provenance.iter().enumerate() {
        let gid = decoded.geometry_ids[out_t];
        let mat = baked.material_for_geometry_id(gid).unwrap();
        if prov.usd_face == 0 {
            assert_eq!(mat, "/Materials/MatA");
        } else {
            assert_eq!(mat, "/Materials/MatB");
        }

        // Verify each output corner's remapped UV matches the exact rotated USD face corner!
        for c in 0..3 {
            let src_corner = usize::from(prov.input_corners[c]);
            assert_eq!(
                remapped_uvs[out_t * 3 + c],
                expected_face_uvs[prov.usd_face as usize][src_corner]
            );
        }
    }
}

#[test]
fn transform_edit_hits_cache_while_geometry_edit_invalidates_and_instances_reuse_arc() {
    let stage = new_stage();
    let mesh_a = define_quad_with_uv_seam_and_subsets(&stage, "/World/QuadA");
    let mesh_b = define_quad_with_uv_seam_and_subsets(&stage, "/World/QuadB");
    mesh_a
        .clone()
        .set_translate(gf::vec3d(1.0, 0.0, 0.0))
        .unwrap();
    mesh_b
        .clone()
        .set_translate(gf::vec3d(5.0, 2.0, -3.0))
        .unwrap();

    let mut cache = UsdDgfCache::new();
    let opts = UsdBakeOptions::default();

    // Baking the stage with two identically shaped meshes at different transforms produces 1 miss + 1 hit!
    let instances = cache.bake_stage(&stage, None, &opts).unwrap();
    assert_eq!(instances.len(), 2);
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 1);
    assert!(Arc::ptr_eq(&instances[0].geometry, &instances[1].geometry));
    assert_ne!(instances[0].world_transform, instances[1].world_transform);

    // Changing transform alone does NOT rebake!
    stage
        .attribute("/World/QuadA.xformOp:translate")
        .unwrap()
        .set(gf::vec3d(99.0, 0.0, 0.0))
        .unwrap();
    let mut xform_cache = XformCache::default();
    let (moved, status) = cache
        .bake_or_get(&mesh_a, None, &mut xform_cache, &opts)
        .unwrap();
    assert_eq!(status, CacheStatus::Hit);
    assert!(Arc::ptr_eq(&moved.geometry, &instances[0].geometry));
    assert_eq!(
        moved.world_transform,
        gf::Mat4d::translation([99.0, 0.0, 0.0]).0
    );

    // Changing vertex positions INVALIDATES cache (new content hash -> Miss)!
    mesh_a
        .points_attr()
        .set(vec![
            gf::vec3f(0.0, 0.0, 0.0),
            gf::vec3f(2.0, 0.0, 0.0),
            gf::vec3f(2.0, 1.0, 0.0),
            gf::vec3f(0.0, 1.0, 0.0),
        ])
        .unwrap();
    let (edited, status2) = cache
        .bake_or_get(&mesh_a, None, &mut xform_cache, &opts)
        .unwrap();
    assert_eq!(status2, CacheStatus::Miss);
    assert!(!Arc::ptr_eq(&edited.geometry, &instances[0].geometry));
    assert_ne!(
        edited.geometry.content_hash,
        instances[0].geometry.content_hash
    );
}

#[test]
fn independent_validation_rejects_negative_counts_quads_and_subdivision() {
    let stage = new_stage();
    let mut xform_cache = XformCache::default();
    let opts = UsdExtractOptions::default();

    // 1. Negative count pattern [4, -1] (sum = 3 == indices.len() — passes Mesh::validate_topology,
    //    MUST be rejected by openusd-dgf!)
    let neg_mesh = geom::Mesh::define(&stage, "/NegCounts").unwrap();
    neg_mesh
        .create_subdivision_scheme_attr()
        .unwrap()
        .set(SubdivisionScheme::None)
        .unwrap();
    neg_mesh
        .create_points_attr()
        .unwrap()
        .set(vec![
            gf::vec3f(0.0, 0.0, 0.0),
            gf::vec3f(1.0, 0.0, 0.0),
            gf::vec3f(0.0, 1.0, 0.0),
        ])
        .unwrap();
    neg_mesh
        .create_face_vertex_counts_attr()
        .unwrap()
        .set(vec![4i32, -1i32])
        .unwrap();
    neg_mesh
        .create_face_vertex_indices_attr()
        .unwrap()
        .set(vec![0i32, 1, 2])
        .unwrap();
    // Confirm upstream Mesh::validate_topology allows it while our extractor rejects it:
    assert_eq!(
        geom::Mesh::validate_topology(&[0, 1, 2], &[4, -1], 3),
        Ok(())
    );
    assert!(matches!(
        extract_usd_mesh(&neg_mesh, None, &mut xform_cache, &opts),
        Err(UsdDgfError::NonTriangleFace {
            face: 0,
            vertex_count: 4,
            ..
        })
    ));

    // Explicit negative count at face 0: [-3]
    neg_mesh.face_vertex_counts_attr().set(vec![-3i32]).unwrap();
    assert!(matches!(
        extract_usd_mesh(&neg_mesh, None, &mut xform_cache, &opts),
        Err(UsdDgfError::NegativeFaceVertexCount {
            face: 0,
            count: -3,
            ..
        })
    ));

    // 2. Non-`none` subdivision scheme rejected by default
    let subdiv_mesh = geom::Mesh::define(&stage, "/Subdiv").unwrap();
    subdiv_mesh
        .create_subdivision_scheme_attr()
        .unwrap()
        .set(SubdivisionScheme::CatmullClark)
        .unwrap();
    assert!(matches!(
        extract_usd_mesh(&subdiv_mesh, None, &mut xform_cache, &opts),
        Err(UsdDgfError::UnsupportedSubdivisionScheme { .. })
    ));
}

#[test]
fn time_sampled_mesh_extracts_and_caches_per_time() {
    let stage = new_stage();
    let mesh = geom::Mesh::define(&stage, "/Anim").unwrap();
    mesh.create_subdivision_scheme_attr()
        .unwrap()
        .set(SubdivisionScheme::None)
        .unwrap();
    mesh.create_face_vertex_counts_attr()
        .unwrap()
        .set(vec![3i32])
        .unwrap();
    mesh.create_face_vertex_indices_attr()
        .unwrap()
        .set(vec![0i32, 1, 2])
        .unwrap();

    mesh.create_points_attr()
        .unwrap()
        .set_at(
            vec![
                gf::vec3f(0.0, 0.0, 0.0),
                gf::vec3f(1.0, 0.0, 0.0),
                gf::vec3f(0.0, 1.0, 0.0),
            ],
            usd::TimeCode::new(1.0),
        )
        .unwrap()
        .set_at(
            vec![
                gf::vec3f(0.0, 0.0, 0.0),
                gf::vec3f(3.0, 0.0, 0.0),
                gf::vec3f(0.0, 1.0, 0.0),
            ],
            usd::TimeCode::new(2.0),
        )
        .unwrap();

    let mut cache = UsdDgfCache::new();
    let mut xform_cache = XformCache::default();
    let opts = UsdBakeOptions::default();

    let (t1, s1) = cache
        .bake_or_get(&mesh, usd::TimeCode::new(1.0), &mut xform_cache, &opts)
        .unwrap();
    let (t2, s2) = cache
        .bake_or_get(&mesh, usd::TimeCode::new(2.0), &mut xform_cache, &opts)
        .unwrap();
    let (t1_again, s3) = cache
        .bake_or_get(&mesh, usd::TimeCode::new(1.0), &mut xform_cache, &opts)
        .unwrap();

    assert_eq!(s1, CacheStatus::Miss);
    assert_eq!(s2, CacheStatus::Miss);
    assert_eq!(s3, CacheStatus::Hit);
    assert_ne!(t1.geometry.content_hash, t2.geometry.content_hash);
    assert!(Arc::ptr_eq(&t1.geometry, &t1_again.geometry));
}
