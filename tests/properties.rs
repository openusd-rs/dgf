//! Deterministic mesh property tests covering all 12 properties from PLAN.md Section 13.4.

#![forbid(unsafe_code)]

mod reference;

use dgf::{
    ClusterConfig, DecodeLimits, DegeneratePolicy, EncodeError, EncodeOptions, MAX_GEOM_ID,
    MAX_PRIM_ID, MeshView, PackingPolicy, QuantizationMode, TriangleMetadata, ValidationMode,
    decode_block, decode_mesh, encode_mesh, encode_mesh_with_user_data, encode_preclustered_mesh,
    quantize_mesh, reconstruct_with_remap, remap_corner_attributes, remap_face_attributes,
    remap_flat_corner_attributes, remap_vertex_attributes, validate_block_with_mode,
};
use reference::assert_matches_reference;

/// Simple deterministic SplitMix64 PRNG (zero external dependencies).
#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    }
}

struct GeneratedMesh {
    positions: Vec<[f32; 3]>,
    triangles: Vec<[u32; 3]>,
    metadata: Vec<TriangleMetadata>,
    omm_ids: Option<Vec<Option<u32>>>,
    vertex_attrs: Vec<u64>,
    face_attrs: Vec<u64>,
    corner_attrs: Vec<[u64; 3]>,
}

fn generate_mesh(rng: &mut Rng, kind: usize) -> GeneratedMesh {
    let mut positions = Vec::new();
    let mut triangles = Vec::new();

    match kind % 5 {
        0 => {
            // Regular deformed grid with UV seams (duplicated seam vertices with identical positions)
            let w = 2 + rng.next_usize(10);
            let h = 2 + rng.next_usize(10);
            for y in 0..=h {
                for x in 0..=w {
                    positions.push([
                        x as f32 * 0.5 + rng.next_f32(-0.05, 0.05),
                        y as f32 * 0.5 + rng.next_f32(-0.05, 0.05),
                        rng.next_f32(-1.0, 1.0),
                    ]);
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let i00 = (y * (w + 1) + x) as u32;
                    let i10 = i00 + 1;
                    let i01 = ((y + 1) * (w + 1) + x) as u32;
                    let i11 = i01 + 1;
                    triangles.push([i00, i10, i11]);
                    triangles.push([i00, i11, i01]);
                }
            }
            // Add an unused vertex + a seam duplicate with identical position
            positions.push([99.0, -99.0, 0.0]);
            let seam_v = positions.len() as u32;
            positions.push(positions[0]);
            if !triangles.is_empty() {
                triangles[0][0] = seam_v;
            }
        }
        1 => {
            // Disconnected random triangle soup with occasional repeated-index degenerates
            let n_tris = 1 + rng.next_usize(140);
            for _ in 0..n_tris {
                let base = positions.len() as u32;
                let cx = rng.next_f32(-20.0, 20.0);
                let cy = rng.next_f32(-20.0, 20.0);
                let cz = rng.next_f32(-20.0, 20.0);
                for _ in 0..3 {
                    positions.push([
                        cx + rng.next_f32(-1.0, 1.0),
                        cy + rng.next_f32(-1.0, 1.0),
                        cz + rng.next_f32(-1.0, 1.0),
                    ]);
                }
                if rng.next_usize(12) == 0 {
                    triangles.push([base, base, base + 2]); // degenerate!
                } else {
                    triangles.push([base, base + 1, base + 2]);
                }
            }
        }
        2 => {
            // Non-manifold fan + mixed winding + duplicate faces
            let n_verts = 6 + rng.next_usize(40);
            for _ in 0..n_verts {
                positions.push([
                    rng.next_f32(-5.0, 5.0),
                    rng.next_f32(-5.0, 5.0),
                    rng.next_f32(-5.0, 5.0),
                ]);
            }
            let n_tris = 10 + rng.next_usize(100);
            for i in 0..n_tris {
                let a = (i % 3) as u32;
                let b = ((i % (n_verts - 3)) + 3) as u32;
                let c = (((i + 1) % (n_verts - 3)) + 3) as u32;
                if a != b && b != c && a != c {
                    if rng.next_usize(2) == 0 {
                        triangles.push([a, b, c]);
                    } else {
                        triangles.push([a, c, b]); // mixed winding
                    }
                }
            }
            if !triangles.is_empty() {
                triangles.push(triangles[0]); // duplicate face
            }
        }
        3 => {
            // All equal centroids / coplanar stacked triangles (SAH tie/fallback stress)
            let n_tris = 5 + rng.next_usize(90);
            positions.push([0.0, 0.0, 0.0]);
            positions.push([1.0, 0.0, 0.0]);
            positions.push([0.0, 1.0, 0.0]);
            positions.push([-1.0, 0.0, 0.0]);
            positions.push([0.0, -1.0, 0.0]);
            for i in 0..n_tris {
                let b = (1 + (i % 4)) as u32;
                let c = (1 + ((i + 1) % 4)) as u32;
                triangles.push([0, b, c]);
            }
        }
        _ => {
            // Closed UV sphere
            let stacks = 4 + rng.next_usize(5);
            let slices = 6 + rng.next_usize(7);
            for s in 0..=stacks {
                let phi = std::f32::consts::PI * (s as f32) / (stacks as f32);
                for l in 0..=slices {
                    let theta = 2.0 * std::f32::consts::PI * (l as f32) / (slices as f32);
                    positions.push([
                        phi.sin() * theta.cos() * 3.0,
                        phi.cos() * 3.0,
                        phi.sin() * theta.sin() * 3.0,
                    ]);
                }
            }
            for s in 0..stacks {
                for l in 0..slices {
                    let a = (s * (slices + 1) + l) as u32;
                    let b = a + 1;
                    let c = ((s + 1) * (slices + 1) + l) as u32;
                    let d = c + 1;
                    triangles.push([a, c, b]);
                    triangles.push([b, c, d]);
                }
            }
        }
    }

    let n_tris = triangles.len();
    let metadata: Vec<TriangleMetadata> = (0..n_tris)
        .map(|i| {
            let gid = match rng.next_usize(4) {
                0 => 0,
                1 => (i as u32) % 5,
                2 => 510 + (rng.next_usize(5) as u32), // straddles 511/512 constant/palette boundary!
                _ => (rng.next_u64() as u32) & MAX_GEOM_ID,
            };
            TriangleMetadata {
                geometry_id: gid,
                opaque: rng.next_usize(2) == 0,
            }
        })
        .collect();

    let omm_ids = if rng.next_usize(3) == 0 {
        let num_distinct = 1 + rng.next_usize(12); // exercises 1..=12 (forces >7 splits)
        Some(
            (0..n_tris)
                .map(|_| {
                    if rng.next_usize(5) == 0 {
                        None
                    } else {
                        Some(100 + (rng.next_usize(num_distinct) as u32))
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    let vertex_attrs: Vec<u64> = (0..positions.len()).map(|i| (i as u64) * 13 + 7).collect();
    let face_attrs: Vec<u64> = (0..n_tris).map(|i| (i as u64) * 31 + 11).collect();
    let corner_attrs: Vec<[u64; 3]> = (0..n_tris)
        .map(|i| {
            let b = (i as u64) * 10;
            [b, b + 1, b + 2]
        })
        .collect();

    GeneratedMesh {
        positions,
        triangles,
        metadata,
        omm_ids,
        vertex_attrs,
        face_attrs,
        corner_attrs,
    }
}

/// Checks properties 1..=11 for a given mesh and encoding options.
fn verify_all_properties(gm: &GeneratedMesh, options: &EncodeOptions) {
    let view = MeshView {
        positions: &gm.positions,
        triangles: &gm.triangles,
        triangle_metadata: Some(&gm.metadata),
        triangle_omm_ids: gm.omm_ids.as_deref(),
    };

    let encoded = encode_mesh(view, options).expect("encode_mesh failed");
    let raw = encoded.to_bytes();
    let decoded_mesh = decode_mesh(&raw, &DecodeLimits::default()).expect("decode_mesh failed");

    // Property 8: Every emitted block passes Asset validation AND independent reference interpreter.
    let mut decoded_blocks = Vec::with_capacity(encoded.blocks.len());
    for block in &encoded.blocks {
        assert_matches_reference(block.as_bytes(), ValidationMode::Asset);
        let v = validate_block_with_mode(block.as_bytes(), ValidationMode::Asset).unwrap();
        decoded_blocks.push(decode_block(&v).unwrap());
    }

    // Property 1: Every retained triangle appears exactly once in triangle_remap;
    // every dropped degenerate appears exactly once in dropped_triangles.
    let mut seen_tri = vec![false; gm.triangles.len()];
    for &dropped in &encoded.dropped_triangles {
        let idx = dropped as usize;
        assert!(!seen_tri[idx], "triangle {idx} dropped twice");
        seen_tri[idx] = true;
        let t = gm.triangles[idx];
        assert!(t[0] == t[1] || t[1] == t[2] || t[0] == t[2]);
    }
    for remap in &encoded.triangle_remap {
        let idx = remap.input_triangle as usize;
        assert!(
            !seen_tri[idx],
            "triangle {idx} emitted twice or also dropped"
        );
        seen_tri[idx] = true;
    }
    assert!(
        seen_tri.into_iter().all(|s| s),
        "some source triangle missing"
    );

    // Property 3 & 4: Compute expected mesh-wide quantization and check exact decoded positions & tolerance.
    let retained_tris: Vec<[u32; 3]> = gm
        .triangles
        .iter()
        .copied()
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect();
    let qm = quantize_mesh(
        &gm.positions,
        &retained_tris,
        options.quantization,
        options.reject_quantization_collapses,
    )
    .unwrap();
    if let QuantizationMode::MaxCoordError(tol) = options.quantization {
        assert!(encoded.quantization.max_coord_error <= f64::from(tol));
    }

    // Property 5: Shared source vertices decode identically in every block copy.
    let reconstructed = reconstruct_with_remap(
        &decoded_mesh,
        &encoded.vertex_remap,
        &encoded.triangle_remap,
        gm.positions.len(),
    )
    .expect("reconstruct_with_remap must succeed with identical shared vertices");

    for (dv, (&dec_pos, &src_v)) in decoded_mesh
        .positions
        .iter()
        .zip(encoded.vertex_remap.iter())
        .enumerate()
    {
        let expected = qm.reconstructed[src_v as usize];
        assert_eq!(
            dec_pos[0].to_bits(),
            expected[0].to_bits(),
            "X mismatch at decoded vertex {dv}"
        );
        assert_eq!(
            dec_pos[1].to_bits(),
            expected[1].to_bits(),
            "Y mismatch at decoded vertex {dv}"
        );
        assert_eq!(
            dec_pos[2].to_bits(),
            expected[2].to_bits(),
            "Z mismatch at decoded vertex {dv}"
        );
    }

    // Property 2 & 6: Cyclic corner winding preservation + metadata + OMM match.
    let mut out_tri_cursor = 0usize;
    for (bi, dec_blk) in decoded_blocks.iter().enumerate() {
        let range = encoded.block_ranges[bi];
        assert_eq!(range.triangle_offset, out_tri_cursor);
        assert_eq!(range.triangle_count, usize::from(dec_blk.num_triangles));
        assert_eq!(range.vertex_count, usize::from(dec_blk.num_vertices));
        assert_eq!(dec_blk.user_data, options.user_data);

        for local_t in 0..usize::from(dec_blk.num_triangles) {
            let global_t = out_tri_cursor + local_t;
            let remap = encoded.triangle_remap[global_t];
            let src_t = gm.triangles[remap.input_triangle as usize];

            // Cyclic rotation check (winding preserved!)
            assert!(
                matches!(remap.input_corners, [0, 1, 2] | [1, 2, 0] | [2, 0, 1]),
                "non-cyclic corner remap {:?}",
                remap.input_corners
            );
            let expected_src_corners = [
                src_t[usize::from(remap.input_corners[0])],
                src_t[usize::from(remap.input_corners[1])],
                src_t[usize::from(remap.input_corners[2])],
            ];
            assert_eq!(reconstructed.triangles[global_t], expected_src_corners);

            // Geometry ID & opacity match
            let expected_meta = gm.metadata[remap.input_triangle as usize];
            assert_eq!(
                decoded_mesh.geometry_ids[global_t],
                expected_meta.geometry_id
            );
            assert_eq!(decoded_mesh.opaque[global_t], expected_meta.opaque);

            // OMM match via block_omm_maps
            if let Some(ref omm_ids) = gm.omm_ids {
                let expected_omm = omm_ids[remap.input_triangle as usize];
                match expected_omm {
                    None => assert_eq!(dec_blk.omm.descriptor_count, 0),
                    Some(id) => {
                        let sel = dec_blk.omm_triangle_descriptors()[local_t];
                        assert_eq!(encoded.block_omm_maps[bi].descriptors[usize::from(sel)], id);
                    }
                }
            }

            // Contiguous primitive ID check
            assert_eq!(
                decoded_mesh.prim_ids[global_t],
                options.prim_id_base + global_t as u32
            );
        }
        out_tri_cursor += usize::from(dec_blk.num_triangles);
    }

    // Property 7: Vertex, face, and face-varying corner attributes reconstruct accurately.
    let remapped_v = remap_vertex_attributes(&gm.vertex_attrs, &encoded.vertex_remap).unwrap();
    let remapped_f = remap_face_attributes(&gm.face_attrs, &encoded.triangle_remap).unwrap();
    let remapped_c = remap_corner_attributes(&gm.corner_attrs, &encoded.triangle_remap).unwrap();
    let flat_src: Vec<u64> = gm.corner_attrs.iter().flatten().copied().collect();
    let remapped_flat = remap_flat_corner_attributes(&flat_src, &encoded.triangle_remap).unwrap();

    for (t_idx, (tri, remap)) in decoded_mesh
        .triangles
        .iter()
        .zip(encoded.triangle_remap.iter())
        .enumerate()
    {
        assert_eq!(
            remapped_f[t_idx],
            gm.face_attrs[remap.input_triangle as usize]
        );
        let src_tri = gm.triangles[remap.input_triangle as usize];
        for c in 0..3 {
            let src_corner = usize::from(remap.input_corners[c]);
            assert_eq!(
                remapped_v[tri[c] as usize],
                gm.vertex_attrs[src_tri[src_corner] as usize]
            );
            assert_eq!(
                remapped_c[t_idx][c],
                gm.corner_attrs[remap.input_triangle as usize][src_corner]
            );
            assert_eq!(remapped_flat[t_idx * 3 + c], remapped_c[t_idx][c]);
        }
    }

    // Property 9: Repeated runs produce bit-for-bit identical bytes and maps.
    let encoded2 = encode_mesh(view, options).unwrap();
    assert_eq!(encoded.blocks, encoded2.blocks);
    assert_eq!(encoded.triangle_remap, encoded2.triangle_remap);
    assert_eq!(encoded.vertex_remap, encoded2.vertex_remap);
    assert_eq!(encoded.block_omm_maps, encoded2.block_omm_maps);
}

#[test]
fn property_suite_across_policies_quantization_omm_and_user_data() {
    let mut rng = Rng::new(0xD6F1_2026_1122);

    let policies = [
        PackingPolicy::BaselineRestartOnly,
        PackingPolicy::BaselineWithStrips,
        PackingPolicy::Sah,
    ];

    for case in 0..35 {
        let gm = generate_mesh(&mut rng, case);
        let quant = match case % 3 {
            0 => QuantizationMode::ExplicitExponent(-6),
            1 => QuantizationMode::MaxCoordError(1e-3),
            _ => QuantizationMode::TargetBitWidth(14),
        };
        let user_data = if case % 4 == 0 {
            Some(0xA500_0000 | (case as u32))
        } else {
            None
        };
        let exp_adj = case % 5 == 0;

        for &packing in &policies {
            let options = EncodeOptions {
                quantization: quant,
                packing,
                degenerate_policy: DegeneratePolicy::DropAndReport,
                prim_id_base: (case as u32) * 1000,
                user_data,
                enable_exponent_adjustment: exp_adj,
                cluster_config: ClusterConfig {
                    max_triangles: 64,
                    max_vertices: 128,
                    num_bins: 16,
                },
                ..EncodeOptions::default()
            };
            verify_all_properties(&gm, &options);
        }
    }
}

#[test]
fn property_10_parallel_and_serial_produce_identical_bytes_and_maps() {
    let mut rng = Rng::new(0x9A7A_11E1);
    let gm = generate_mesh(&mut rng, 4); // UV sphere + random metadata

    let view = MeshView {
        positions: &gm.positions,
        triangles: &gm.triangles,
        triangle_metadata: Some(&gm.metadata),
        triangle_omm_ids: gm.omm_ids.as_deref(),
    };

    let serial_opts = EncodeOptions {
        quantization: QuantizationMode::MaxCoordError(1e-4),
        packing: PackingPolicy::Sah,
        degenerate_policy: DegeneratePolicy::DropAndReport,
        cluster_config: ClusterConfig {
            max_triangles: 32,
            max_vertices: 64,
            num_bins: 8,
        },
        parallel: false,
        ..EncodeOptions::default()
    };
    let parallel_opts = EncodeOptions {
        parallel: true,
        ..serial_opts
    };

    let serial = encode_mesh(view, &serial_opts).unwrap();
    let parallel = encode_mesh(view, &parallel_opts).unwrap();

    assert!(serial.blocks.len() > 2);
    assert_eq!(serial.blocks, parallel.blocks);
    assert_eq!(serial.triangle_remap, parallel.triangle_remap);
    assert_eq!(serial.vertex_remap, parallel.vertex_remap);
    assert_eq!(serial.block_ranges, parallel.block_ranges);
    assert_eq!(serial.block_omm_maps, parallel.block_omm_maps);
    assert_eq!(serial.statistics, parallel.statistics);
}

#[test]
fn property_preclustered_and_user_data_callback_and_empty_mesh() {
    // Empty mesh is valid with 0 blocks
    let empty = encode_mesh(
        MeshView {
            positions: &[],
            triangles: &[],
            triangle_metadata: None,
            triangle_omm_ids: None,
        },
        &EncodeOptions::default(),
    )
    .unwrap();
    assert!(empty.blocks.is_empty());

    // All-dropped mesh is valid with 0 blocks
    let all_degen = encode_mesh(
        MeshView {
            positions: &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            triangles: &[[0, 0, 1], [1, 1, 1]],
            triangle_metadata: None,
            triangle_omm_ids: None,
        },
        &EncodeOptions {
            degenerate_policy: DegeneratePolicy::DropAndReport,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    assert!(all_degen.blocks.is_empty());
    assert_eq!(all_degen.dropped_triangles, vec![0, 1]);

    // Preclustered input and per-block user_data callback
    let mut rng = Rng::new(0xC105_7E75);
    let gm = generate_mesh(&mut rng, 0);
    let view = MeshView {
        positions: &gm.positions,
        triangles: &gm.triangles,
        triangle_metadata: Some(&gm.metadata),
        triangle_omm_ids: None,
    };
    let mid = gm.triangles.len() / 2;
    let c0: Vec<u32> = (0..mid as u32).collect();
    let c1: Vec<u32> = (mid as u32..gm.triangles.len() as u32).collect();
    let pre = encode_preclustered_mesh(view, &[&c0, &c1], &EncodeOptions::default()).unwrap();
    assert_eq!(pre.triangle_remap.len(), gm.triangles.len());
    for blk in &pre.blocks {
        assert_matches_reference(blk.as_bytes(), ValidationMode::Asset);
    }

    let with_ud = encode_mesh_with_user_data(view, &EncodeOptions::default(), |idx, range| {
        ((idx as u32) << 16) | (range.triangle_count as u32)
    })
    .unwrap();
    for (idx, (blk, range)) in with_ud
        .blocks
        .iter()
        .zip(with_ud.block_ranges.iter())
        .enumerate()
    {
        assert_matches_reference(blk.as_bytes(), ValidationMode::Asset);
        let dec =
            decode_block(&validate_block_with_mode(blk.as_bytes(), ValidationMode::Asset).unwrap())
                .unwrap();
        assert_eq!(
            dec.user_data,
            Some(((idx as u32) << 16) | (range.triangle_count as u32))
        );
    }
}

#[test]
fn property_12_out_of_domain_inputs_return_structured_errors() {
    // NaN coordinate
    assert!(matches!(
        encode_mesh(
            MeshView {
                positions: &[[f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: &[[0, 1, 2]],
                triangle_metadata: None,
                triangle_omm_ids: None,
            },
            &EncodeOptions::default(),
        ),
        Err(EncodeError::NonFiniteCoordinate {
            vertex: 0,
            axis: 0,
            ..
        })
    ));

    // OOB vertex index
    assert!(matches!(
        encode_mesh(
            MeshView {
                positions: &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                triangles: &[[0, 1, 2]],
                triangle_metadata: None,
                triangle_omm_ids: None,
            },
            &EncodeOptions::default(),
        ),
        Err(EncodeError::VertexIndexOutOfBounds {
            triangle: 0,
            corner: 2,
            index: 2,
            ..
        })
    ));

    // Degenerate rejected by default
    assert!(matches!(
        encode_mesh(
            MeshView {
                positions: &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: &[[0, 0, 2]],
                triangle_metadata: None,
                triangle_omm_ids: None,
            },
            &EncodeOptions::default(),
        ),
        Err(EncodeError::DegenerateTriangle { triangle: 0, .. })
    ));

    // Geometry ID > 24-bit
    assert!(matches!(
        encode_mesh(
            MeshView {
                positions: &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                triangles: &[[0, 1, 2]],
                triangle_metadata: Some(&[TriangleMetadata {
                    geometry_id: MAX_GEOM_ID + 1,
                    opaque: true,
                }]),
                triangle_omm_ids: None,
            },
            &EncodeOptions::default(),
        ),
        Err(EncodeError::InvalidGeometryId { triangle: 0, .. })
    ));

    // Primitive ID range overflow
    assert!(matches!(
        encode_mesh(
            MeshView {
                positions: &[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0]
                ],
                triangles: &[[0, 1, 2], [2, 1, 3]],
                triangle_metadata: None,
                triangle_omm_ids: None,
            },
            &EncodeOptions {
                prim_id_base: MAX_PRIM_ID,
                ..EncodeOptions::default()
            },
        ),
        Err(EncodeError::PrimitiveIdRangeExceeded { .. })
    ));
}
