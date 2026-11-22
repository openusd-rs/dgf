//! Executable DGF1 conformance tests: hand-derived starter fixtures, independent reference
//! cross-checks, exhaustive boundary/malformed cases, and control-sequence enumeration.

#![forbid(unsafe_code)]

mod reference;

use dgf::{
    ANCHOR_MAX, ANCHOR_MIN, BLOCK_BYTES, BiasedExponent, DecodeError, DecodeLimits, EncodeError,
    EncodeOptions, FitFailure, MAX_PRIM_ID, MeshView, PackingPolicy, PaddingSection, PreparedBlock,
    PreparedOmm, PreparedTopology, QuantizationMode, TriControl, TriangleMetadata, ValidationMode,
    decode_block, decode_mesh, decode_stream, encode_block, encode_mesh, validate_block,
    validate_block_with_mode,
};
use reference::{assert_matches_reference, decode_block_independent, ref_write_bits, sha256_hex};

/// Section 21 hand-derived starter vector bytes (literally transcribed, independent of encoder).
fn starter_single_tri_bytes() -> [u8; 128] {
    let mut b = [0u8; 128];
    b[0..22].copy_from_slice(&[
        0x06, 0x08, 0x40, 0x00, 0x7f, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x04,
    ]);
    b
}

#[test]
fn fixture_section21_starter_vector_exact_sha256_and_independent_decode() {
    let bytes = starter_single_tri_bytes();
    assert_eq!(
        sha256_hex(&bytes),
        "cd94a17fc87f32017a41d7a7c48329edf232012621e346d9708fba45447f79de"
    );

    // 1. Independent reference decode
    let r = decode_block_independent(&bytes, true).expect("independent decode");
    assert_eq!(r.quantized_positions, vec![[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
    assert_eq!(r.triangles, vec![[0, 1, 2]]);
    assert_eq!(r.geometry_ids, vec![0]);
    assert_eq!(r.opaque, vec![true]);

    // 2. Production decode matches independent reference
    assert_matches_reference(&bytes, ValidationMode::Asset);

    // 3. Production encoder reproduces the EXACT 128 bytes bit-for-bit!
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
    let encoded = encode_block(&prepared).unwrap();
    assert_eq!(encoded.as_bytes(), &bytes);
    assert_eq!(
        sha256_hex(encoded.as_bytes()),
        "cd94a17fc87f32017a41d7a7c48329edf232012621e346d9708fba45447f79de"
    );
}

#[test]
fn fixture_hand_assembled_two_tris_edge1() {
    // Hand-assembled independently via ref_write_bits:
    // V=4: [0,0,0], [1,0,0], [0,1,0], [1,1,0]
    // T=2: tri0=RESTART [0,1,2], tri1=EDGE1 -> [2,1,3] (new vertex 3, is_first[3]=1)
    // anchor=[0,0,0], widths=[2,1,1], exp=127, constant geom_id=0 opaque=true
    let mut b = [0u8; 128];
    let w0: u32 = 0x06 | (3 << 10) | (1 << 16) | (1 << 22);
    let w1: u32 = 127;
    let w2: u32 = 1; // x_bits=2, y_bits=1
    b[0..4].copy_from_slice(&w0.to_le_bytes());
    b[4..8].copy_from_slice(&w1.to_le_bytes());
    b[8..12].copy_from_slice(&w2.to_le_bytes());
    // 4 vertices * 4 bits = 16 bits at bit 160: (0,0,0)=0, (1,0,0)=1, (0,1,0)=4, (1,1,0)=5
    b[20] = 0x10;
    b[21] = 0x54;
    // Control 1 = EDGE1 (1) at bit 1022
    ref_write_bits(&mut b, 1022, 2, 1);
    // First-use flag for strip index 3 at bit 1024 - 2 - 1 - 0 = 1021 -> 1
    ref_write_bits(&mut b, 1021, 1, 1);

    assert_matches_reference(&b, ValidationMode::Asset);
    let dec = decode_block(&validate_block(&b).unwrap()).unwrap();
    assert_eq!(dec.triangles(), &[[0, 1, 2], [2, 1, 3]]);

    // Encoder must produce the EXACT same 128 bytes!
    let enc = encode_block(&PreparedBlock {
        quantized_positions: vec![[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        exponent: BiasedExponent::new(127).unwrap(),
        topology: PreparedTopology {
            controls: vec![TriControl::Restart, TriControl::Edge1],
            strip_indices: vec![0, 1, 2, 3],
            num_vertices: 4,
        },
        geometry_ids: vec![0, 0],
        opaque: vec![true, true],
        prim_id_base: 0,
        user_data: None,
        omm: None,
        allow_exponent_adjustment: false,
    })
    .unwrap();
    assert_eq!(enc.as_bytes(), &b);
}

#[test]
fn fixture_hand_assembled_two_disconnected_tris_restart() {
    // V=6, T=2: tri0=[0,1,2], tri1=RESTART [3,4,5]; strip_indices=[0,1,2,3,4,5], all first-use=1
    let mut b = [0u8; 128];
    let w0: u32 = 0x06 | (5 << 10) | (1 << 16) | (1 << 22);
    let w1: u32 = 127;
    let w2: u32 = 1; // widths [2,1,1]
    b[0..4].copy_from_slice(&w0.to_le_bytes());
    b[4..8].copy_from_slice(&w1.to_le_bytes());
    b[8..12].copy_from_slice(&w2.to_le_bytes());
    // Offsets: (0,0,0), (1,0,0), (0,1,0), (2,0,0), (3,0,0), (2,1,0)
    let offs = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (2, 0, 0),
        (3, 0, 0),
        (2, 1, 0),
    ];
    for (v, (x, y, z)) in offs.into_iter().enumerate() {
        let base = 160 + v * 4;
        ref_write_bits(&mut b, base, 2, x);
        ref_write_bits(&mut b, base + 2, 1, y);
        ref_write_bits(&mut b, base + 3, 1, z);
    }
    // Control 1 = RESTART (0) at bit 1022
    ref_write_bits(&mut b, 1022, 2, 0);
    // First-use flags j=3,4,5 at bits 1021, 1020, 1019 = 1, 1, 1
    ref_write_bits(&mut b, 1021, 1, 1);
    ref_write_bits(&mut b, 1020, 1, 1);
    ref_write_bits(&mut b, 1019, 1, 1);

    assert_matches_reference(&b, ValidationMode::Asset);
    let dec = decode_block(&validate_block(&b).unwrap()).unwrap();
    assert_eq!(dec.triangles(), &[[0, 1, 2], [3, 4, 5]]);

    let enc = encode_block(&PreparedBlock {
        quantized_positions: vec![
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 0],
            [2, 1, 0],
        ],
        exponent: BiasedExponent::new(127).unwrap(),
        topology: PreparedTopology {
            controls: vec![TriControl::Restart, TriControl::Restart],
            strip_indices: vec![0, 1, 2, 3, 4, 5],
            num_vertices: 6,
        },
        geometry_ids: vec![0, 0],
        opaque: vec![true, true],
        prim_id_base: 0,
        user_data: None,
        omm: None,
        allow_exponent_adjustment: false,
    })
    .unwrap();
    assert_eq!(enc.as_bytes(), &b);
}

#[test]
fn fixture_backtrack_both_directions() {
    // Direction 1: RESTART [0,1,2], EDGE1 -> [2,1,3], BACKTRACK -> [0,2,4]
    let p1 = PreparedBlock {
        quantized_positions: vec![[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 1, 0]],
        exponent: BiasedExponent::new(127).unwrap(),
        topology: PreparedTopology {
            controls: vec![
                TriControl::Restart,
                TriControl::Edge1,
                TriControl::Backtrack,
            ],
            strip_indices: vec![0, 1, 2, 3, 4],
            num_vertices: 5,
        },
        geometry_ids: vec![0; 3],
        opaque: vec![true; 3],
        prim_id_base: 0,
        user_data: None,
        omm: None,
        allow_exponent_adjustment: false,
    };
    let b1 = encode_block(&p1).unwrap();
    assert_matches_reference(b1.as_bytes(), ValidationMode::Asset);
    let d1 = decode_block(&validate_block(b1.as_bytes()).unwrap()).unwrap();
    assert_eq!(d1.triangles(), &[[0, 1, 2], [2, 1, 3], [0, 2, 4]]);

    // Direction 2: RESTART [0,1,2], EDGE2 -> [0,2,3], BACKTRACK -> [2,1,4]
    let p2 = PreparedBlock {
        quantized_positions: vec![[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 1, 0], [1, 1, 0]],
        exponent: BiasedExponent::new(127).unwrap(),
        topology: PreparedTopology {
            controls: vec![
                TriControl::Restart,
                TriControl::Edge2,
                TriControl::Backtrack,
            ],
            strip_indices: vec![0, 1, 2, 3, 4],
            num_vertices: 5,
        },
        geometry_ids: vec![0; 3],
        opaque: vec![true; 3],
        prim_id_base: 0,
        user_data: None,
        omm: None,
        allow_exponent_adjustment: false,
    };
    let b2 = encode_block(&p2).unwrap();
    assert_matches_reference(b2.as_bytes(), ValidationMode::Asset);
    let d2 = decode_block(&validate_block(b2.as_bytes()).unwrap()).unwrap();
    assert_eq!(d2.triangles(), &[[0, 1, 2], [0, 2, 3], [2, 1, 4]]);
}

#[test]
fn fixture_repeated_low_vertex_3bit_reuse_and_high_vertex_count() {
    // 20 vertices total, but ONLY vertex 0 and 1 are repeated -> bits_per_index MUST be 3!
    // Tri 0: [0,1,2]
    // Tri 1..=16: EDGE1 adding new vertices 3..=18
    // Tri 17: EDGE1 adding vertex 19
    // Tri 18: EDGE1 reusing vertex 0 (low-numbered repeat!)
    // Tri 19: EDGE2 reusing vertex 1 (low-numbered repeat!)
    let mut controls = vec![TriControl::Restart];
    let mut strip = vec![0u8, 1, 2];
    for v in 3u8..20 {
        controls.push(TriControl::Edge1);
        strip.push(v);
    }
    controls.push(TriControl::Edge1);
    strip.push(0);
    controls.push(TriControl::Edge2);
    strip.push(1);

    let positions: Vec<[i32; 3]> = (0..20).map(|i| [i, i % 3, 0]).collect();
    let t_count = controls.len();
    let block = encode_block(&PreparedBlock {
        quantized_positions: positions,
        exponent: BiasedExponent::new(127).unwrap(),
        topology: PreparedTopology {
            controls,
            strip_indices: strip,
            num_vertices: 20,
        },
        geometry_ids: vec![7; t_count],
        opaque: vec![true; t_count],
        prim_id_base: 100,
        user_data: None,
        omm: None,
        allow_exponent_adjustment: false,
    })
    .unwrap();

    assert_matches_reference(block.as_bytes(), ValidationMode::Asset);
    let dec = decode_block(&validate_block(block.as_bytes()).unwrap()).unwrap();
    assert_eq!(dec.header.bits_per_index, 3);
    assert_eq!(dec.num_vertices, 20);
    assert_eq!(dec.layout.repeated_index_count, 2);
}

#[test]
fn fixture_repeated_index_width_thresholds_3_4_5_6() {
    for &(repeat_idx, expected_bpi) in &[
        (7u8, 3u8),
        (8, 4),
        (15, 4),
        (16, 5),
        (31, 5),
        (32, 6),
        (62, 6),
    ] {
        let num_v = repeat_idx + 2; // ensure repeat_idx is introduced and < 64
        let mut controls = vec![TriControl::Restart];
        let mut strip = vec![0u8, 1, 2];
        for v in 3..num_v {
            controls.push(TriControl::Edge1);
            strip.push(v);
        }
        controls.push(TriControl::Edge1);
        strip.push(repeat_idx);
        let t_count = controls.len();
        let positions: Vec<[i32; 3]> = (0..i32::from(num_v)).map(|i| [i % 4, i / 4, 0]).collect();
        let block = encode_block(&PreparedBlock {
            quantized_positions: positions,
            exponent: BiasedExponent::new(127).unwrap(),
            topology: PreparedTopology {
                controls,
                strip_indices: strip,
                num_vertices: num_v,
            },
            geometry_ids: vec![0; t_count],
            opaque: vec![true; t_count],
            prim_id_base: 0,
            user_data: None,
            omm: None,
            allow_exponent_adjustment: false,
        })
        .unwrap();
        assert_matches_reference(block.as_bytes(), ValidationMode::Asset);
        let dec = decode_block(&validate_block(block.as_bytes()).unwrap()).unwrap();
        assert_eq!(
            dec.header.bits_per_index, expected_bpi,
            "repeat_idx={repeat_idx}"
        );
    }
}

#[test]
fn fixture_geom_id_511_constant_vs_512_one_entry_25bit_prefix() {
    // 511 fits constant mode
    let b511 = encode_block(&PreparedBlock {
        quantized_positions: vec![[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        exponent: BiasedExponent::new(127).unwrap(),
        topology: PreparedTopology {
            controls: vec![TriControl::Restart],
            strip_indices: vec![0, 1, 2],
            num_vertices: 3,
        },
        geometry_ids: vec![511],
        opaque: vec![false],
        prim_id_base: 0,
        user_data: None,
        omm: None,
        allow_exponent_adjustment: false,
    })
    .unwrap();
    assert_matches_reference(b511.as_bytes(), ValidationMode::Asset);
    let d511 = decode_block(&validate_block(b511.as_bytes()).unwrap()).unwrap();
    assert!(matches!(
        d511.header.geom_id_mode,
        dgf::GeomIdHeaderMode::Constant {
            geometry_id: 511,
            opaque: false
        }
    ));
    assert_eq!(d511.layout.front.geom_bits, 0);

    // 512 requires 1-entry palette with prefix_bits = 25, selector_bits = 0, payload_bits = 0
    let b512 = encode_block(&PreparedBlock {
        quantized_positions: vec![[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        exponent: BiasedExponent::new(127).unwrap(),
        topology: PreparedTopology {
            controls: vec![TriControl::Restart],
            strip_indices: vec![0, 1, 2],
            num_vertices: 3,
        },
        geometry_ids: vec![512],
        opaque: vec![true],
        prim_id_base: 0,
        user_data: None,
        omm: None,
        allow_exponent_adjustment: false,
    })
    .unwrap();
    assert_matches_reference(b512.as_bytes(), ValidationMode::Asset);
    let d512 = decode_block(&validate_block(b512.as_bytes()).unwrap()).unwrap();
    assert_eq!(
        d512.header.geom_id_mode,
        dgf::GeomIdHeaderMode::Palette {
            prefix_bits: 25,
            entry_count: 1
        }
    );
    assert_eq!(d512.layout.front.geom_selector_bits_per_tri, 0);
    assert_eq!(d512.layout.front.geom_payload_bits_per_entry, 0);
    assert_eq!(d512.layout.front.geom_bits, 32); // align8(25) = 32
    assert_eq!(d512.geometry_ids(), &[512]);
    assert_eq!(d512.opaque_flags(), &[true]);
}

#[test]
fn fixture_user_data_plus_omm_all_counts_0_through_7_and_geom_palette() {
    for d in 0u8..=7 {
        let num_tris = 8usize;
        let mut controls = vec![TriControl::Restart];
        let mut strip = vec![0u8, 1, 2];
        for i in 1..num_tris {
            controls.push(TriControl::Edge1);
            strip.push((i + 2) as u8);
        }
        let positions: Vec<[i32; 3]> = (0..10).map(|i| [i, i % 2, 0]).collect();
        let geom_ids: Vec<u32> = vec![1000, 1001, 1000, 1002, 1001, 1003, 1000, 1002];
        let opaque: Vec<bool> = vec![true, false, false, true, false, true, true, true];
        let omm = if d == 0 {
            None
        } else {
            Some(PreparedOmm {
                descriptor_count: d,
                triangle_descriptors: (0..num_tris).map(|t| (t as u8) % d).collect(),
                hot_patch_bytes: None,
            })
        };

        let block = encode_block(&PreparedBlock {
            quantized_positions: positions,
            exponent: BiasedExponent::new(127).unwrap(),
            topology: PreparedTopology {
                controls,
                strip_indices: strip,
                num_vertices: 10,
            },
            geometry_ids: geom_ids.clone(),
            opaque: opaque.clone(),
            prim_id_base: 42,
            user_data: Some(0xDEAD_BEEF),
            omm,
            allow_exponent_adjustment: false,
        })
        .unwrap();

        assert_matches_reference(block.as_bytes(), ValidationMode::Asset);
        let dec = decode_block(
            &validate_block_with_mode(block.as_bytes(), ValidationMode::Asset).unwrap(),
        )
        .unwrap();
        assert_eq!(dec.user_data, Some(0xDEAD_BEEF));
        assert_eq!(dec.geometry_ids(), geom_ids.as_slice());
        assert_eq!(dec.opaque_flags(), opaque.as_slice());
        assert_eq!(dec.omm.descriptor_count, d);
        if d > 0 {
            assert_eq!(dec.omm.hot_patch().len(), 4 * (2 + usize::from(d)));
            assert!(dec.omm.hot_patch().iter().all(|&b| b == 0));

            // Patch nonzero runtime bytes into the hot-patch region:
            // Must succeed in Runtime mode (preserving bytes verbatim) and FAIL in Asset mode!
            let mut patched = *block.as_bytes();
            let hp_byte_start = dec.layout.front.omm_start / 8;
            patched[hp_byte_start] = 0xAB;
            patched[hp_byte_start + 4 * (2 + usize::from(d)) - 1] = 0xCD;

            assert_matches_reference(&patched, ValidationMode::Runtime);
            let dec_rt = decode_block(&validate_block(&patched).unwrap()).unwrap();
            assert_eq!(dec_rt.omm.hot_patch()[0], 0xAB);
            assert_eq!(*dec_rt.omm.hot_patch().last().unwrap(), 0xCD);

            assert_eq!(
                validate_block_with_mode(&patched, ValidationMode::Asset).unwrap_err(),
                DecodeError::OmmHotPatchNonZero {
                    byte_offset: 0,
                    value: 0xAB
                }
            );

            // For d in {3, 5, 6}, test invalid unused selector encoding (selector == d)!
            if matches!(d, 3 | 5 | 6) {
                let mut bad_sel = *block.as_bytes();
                let sel_w = usize::from(dec.layout.front.omm_selector_bits_per_tri);
                ref_write_bits(
                    &mut bad_sel,
                    dec.layout.front.omm_selector_start,
                    sel_w,
                    u64::from(d),
                );
                assert_eq!(
                    validate_block(&bad_sel).unwrap_err(),
                    DecodeError::OmmSelectorOutOfRange {
                        triangle: 0,
                        selector: d,
                        descriptor_count: d,
                    }
                );
                assert!(decode_block_independent(&bad_sel, false).is_err());
            }
        }
    }
}

#[test]
fn fixture_more_than_64_first_use_flags_with_many_restarts() {
    // 25 triangles all RESTART -> 75 strip indices -> 72 first-use flags (> 64!)
    // Use 4 shared vertices repeated via restarts so vertex section stays tiny.
    let num_tris = 25usize;
    let controls = vec![TriControl::Restart; num_tris];
    let mut strip = vec![0u8, 1, 2];
    for t in 1..num_tris {
        if t == 1 {
            strip.extend_from_slice(&[3, 0, 1]);
        } else {
            strip.extend_from_slice(&[0, 1, 2]);
        }
    }
    // Wait: 75 strip indices - 4 unique vertices = 71 repeated indices -> 71*3 = 213 bits > 192!
    // Instead, introduce 60 unique vertices across 25 restarts (75 strip indices, 15 repeated indices -> 15*6 = 90 bits <= 192)!
    let mut strip2 = vec![0u8, 1, 2];
    let mut next_v = 3u8;
    for _ in 1..num_tris {
        for _ in 0..3 {
            if next_v < 60 {
                strip2.push(next_v);
                next_v += 1;
            } else {
                strip2.push(0); // low repeated index -> 3-bit reuse!
            }
        }
    }
    assert_eq!(strip2.len(), 75); // 72 first-use flags (> 64!)
    let positions: Vec<[i32; 3]> = (0..60).map(|i| [i % 2, (i / 2) % 2, 0]).collect();

    let block = encode_block(&PreparedBlock {
        quantized_positions: positions,
        exponent: BiasedExponent::new(127).unwrap(),
        topology: PreparedTopology {
            controls,
            strip_indices: strip2,
            num_vertices: 60,
        },
        geometry_ids: vec![0; num_tris],
        opaque: vec![true; num_tris],
        prim_id_base: 0,
        user_data: None,
        omm: None,
        allow_exponent_adjustment: false,
    })
    .unwrap();

    assert_matches_reference(block.as_bytes(), ValidationMode::Asset);
    let dec = decode_block(&validate_block(block.as_bytes()).unwrap()).unwrap();
    assert_eq!(dec.layout.strip_index_count, 75);
    assert_eq!(dec.layout.first_use_bits, 72);
    assert_eq!(dec.num_triangles, 25);
    assert_eq!(dec.num_vertices, 60);
    assert_eq!(dec.header.bits_per_index, 3);
}

#[test]
fn fixture_anchors_exponents_overflow_and_prim_id_boundaries() {
    // Extreme signed 24-bit anchors: -8_388_608, -1, 0, 8_388_607
    for &anc in &[ANCHOR_MIN, -1, 0, ANCHOR_MAX - 1] {
        let block = encode_block(&PreparedBlock {
            quantized_positions: vec![[anc, anc, anc], [anc + 1, anc, anc], [anc, anc + 1, anc]],
            exponent: BiasedExponent::new(1).unwrap(),
            topology: PreparedTopology {
                controls: vec![TriControl::Restart],
                strip_indices: vec![0, 1, 2],
                num_vertices: 3,
            },
            geometry_ids: vec![0],
            opaque: vec![true],
            prim_id_base: MAX_PRIM_ID, // 1 triangle -> MAX_PRIM_ID fits!
            user_data: None,
            omm: None,
            allow_exponent_adjustment: false,
        })
        .unwrap();
        assert_matches_reference(block.as_bytes(), ValidationMode::Asset);
    }

    // Exponent 232 with anchor = 8_388_607 and offset = 0 -> finite f32::MAX!
    let ok_232 = encode_block(&PreparedBlock {
        quantized_positions: vec![[ANCHOR_MAX, 0, 0], [ANCHOR_MAX, 0, 0], [ANCHOR_MAX, 0, 0]],
        exponent: BiasedExponent::new(232).unwrap(),
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
    })
    .unwrap();
    assert_matches_reference(ok_232.as_bytes(), ValidationMode::Asset);

    // Mutate vertex 1's X offset from 0 to 1 at exponent 232: anchor=8_388_607 + 1 = 8_388_608 -> +inf!
    let mut overflow_232 = *ok_232.as_bytes();
    // Vertex 1 X offset is at bit 160 + 4 = 164 (widths [2,1,1])
    ref_write_bits(&mut overflow_232, 164, 2, 1);
    assert!(matches!(
        validate_block(&overflow_232),
        Err(DecodeError::NonFinitePosition {
            vertex: 1,
            axis: 0,
            exponent: 232,
            ..
        })
    ));
    assert!(decode_block_independent(&overflow_232, false).is_err());
}

#[test]
fn fixture_malformed_padding_overlap_vertex_count_and_noncanonical_valid() {
    let valid = starter_single_tri_bytes();

    // 1. Nonzero vertex padding bit (bit 172..176 in byte 21 high nibble)
    let mut bad_vpad = valid;
    bad_vpad[21] |= 0x80;
    assert_eq!(
        validate_block(&bad_vpad).unwrap_err(),
        DecodeError::PaddingNonZero {
            section: PaddingSection::VertexOffsets,
            bit_offset: 172,
        }
    );
    assert!(decode_block_independent(&bad_vpad, false).is_err());

    // 2. Nonzero unused gap bit (byte 60) is VALID (non-canonical) in DGF1!
    let mut nonzero_gap = valid;
    nonzero_gap[60] = 0xFF;
    assert_matches_reference(&nonzero_gap, ValidationMode::Asset);

    // 3. Header vertex count 1 or 2 -> rejected via topology validation!
    for bad_v in [1u32, 2u32] {
        let mut b = valid;
        let w0 =
            (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & !(0x3F << 10)) | ((bad_v - 1) << 10);
        b[0..4].copy_from_slice(&w0.to_le_bytes());
        // Zero byte 20/21 vertex padding so we reach topology check cleanly
        b[20] = 0;
        b[21] = 0;
        assert!(matches!(
            validate_block(&b),
            Err(DecodeError::VertexIndexOutOfRange { .. })
        ));
        assert!(decode_block_independent(&b, false).is_err());
    }

    // 4. Unused trailing stored vertex (V=4 while only 0,1,2 referenced): structurally legal non-canonical!
    let mut unused_v = valid;
    let w0 = (u32::from_le_bytes([unused_v[0], unused_v[1], unused_v[2], unused_v[3]])
        & !(0x3F << 10))
        | (3u32 << 10); // V=4
    unused_v[0..4].copy_from_slice(&w0.to_le_bytes());
    assert_matches_reference(&unused_v, ValidationMode::Asset);

    // 5. Section overlap despite individually legal front_bits (768) and repeat_bits (<=192):
    // V=64, widths=[4,4,4] -> front_bits = 768, have_user_data=true -> repeat_start = 160 + 32 + 768 = 960
    // T=32 all EDGE1 -> control_bits = 62, first_use_bits = 31 -> tail_start = 1024 - 93 = 931 < 960!
    let mut overlap = [0u8; 128];
    let w0: u32 = 0x06 | (63 << 10) | (31 << 16) | (1 << 22);
    let w1: u32 = 127;
    let w2: u32 = 3 | (3 << 4); // x=4, y=4
    let w3: u32 = 3; // z=4
    let w4: u32 = 1 << 29; // have_user_data = true
    overlap[0..4].copy_from_slice(&w0.to_le_bytes());
    overlap[4..8].copy_from_slice(&w1.to_le_bytes());
    overlap[8..12].copy_from_slice(&w2.to_le_bytes());
    overlap[12..16].copy_from_slice(&w3.to_le_bytes());
    overlap[16..20].copy_from_slice(&w4.to_le_bytes());
    // Set controls 1..32 to EDGE1 (1)
    for i in 1..32 {
        ref_write_bits(&mut overlap, 1024 - 2 * i, 2, 1);
    }
    assert!(matches!(
        validate_block(&overlap),
        Err(DecodeError::SectionOverlap { .. })
    ));
    assert!(decode_block_independent(&overlap, false).is_err());
}

#[test]
fn fixture_stream_truncation_empty_and_limits() {
    // Empty stream is a valid empty mesh
    assert!(
        decode_stream(&[], &DecodeLimits::default())
            .unwrap()
            .is_empty()
    );
    let empty_mesh = decode_mesh(&[], &DecodeLimits::default()).unwrap();
    assert!(empty_mesh.positions.is_empty());
    assert!(empty_mesh.triangles.is_empty());

    // Partial final blocks rejected
    for len in [1usize, 19, 20, 127, 129, 255] {
        let data = vec![0u8; len];
        let err = decode_stream(&data, &DecodeLimits::default()).unwrap_err();
        assert_eq!(err.kind, DecodeError::TruncatedStream { byte_len: len });
    }

    // Limits enforced
    let b = starter_single_tri_bytes();
    let mut two_blocks = Vec::with_capacity(2 * BLOCK_BYTES);
    two_blocks.extend_from_slice(&b);
    two_blocks.extend_from_slice(&b);
    let lim_blocks = DecodeLimits {
        max_blocks: 1,
        ..DecodeLimits::default()
    };
    assert_eq!(
        decode_stream(&two_blocks, &lim_blocks).unwrap_err().kind,
        DecodeError::BlockLimitExceeded { count: 2, limit: 1 }
    );
    let lim_verts = DecodeLimits {
        max_vertices: 5,
        ..DecodeLimits::default()
    };
    assert_eq!(
        decode_mesh(&two_blocks, &lim_verts).unwrap_err().kind,
        DecodeError::VertexLimitExceeded { count: 6, limit: 5 }
    );
}

#[test]
fn enumerate_all_short_control_sequences() {
    // Enumerate all 4^4 = 256 control sequences of length 5 (tri 0 is RESTART, tris 1..=4 in 0..4)
    for mask in 0u32..256 {
        let mut controls = vec![TriControl::Restart];
        let mut legal = true;
        for step in 0..4 {
            let raw = ((mask >> (step * 2)) & 3) as u8;
            let ctrl = TriControl::from_u8(raw).unwrap();
            if ctrl == TriControl::Backtrack {
                let i = step + 1;
                if i < 2 || !matches!(controls[i - 1], TriControl::Edge1 | TriControl::Edge2) {
                    legal = false;
                }
            }
            controls.push(ctrl);
        }

        let strip_len: usize = controls.iter().map(|c| c.index_count()).sum();
        let strip: Vec<u8> = (0..strip_len as u8).collect();
        let positions: Vec<[i32; 3]> = (0..strip_len as i32).map(|v| [v, v % 3, 0]).collect();

        let prepared = PreparedBlock {
            quantized_positions: positions,
            exponent: BiasedExponent::new(127).unwrap(),
            topology: PreparedTopology {
                controls: controls.clone(),
                strip_indices: strip,
                num_vertices: strip_len as u8,
            },
            geometry_ids: vec![0; 5],
            opaque: vec![true; 5],
            prim_id_base: 0,
            user_data: None,
            omm: None,
            allow_exponent_adjustment: false,
        };

        let res = encode_block(&prepared);
        if legal {
            let block = res.expect("legal control sequence must encode");
            assert_matches_reference(block.as_bytes(), ValidationMode::Asset);
        } else {
            assert!(
                matches!(res, Err(EncodeError::InvalidTopology(_))),
                "illegal control sequence {controls:?} must be rejected"
            );
        }
    }
}

#[test]
fn mesh_baker_splits_33_geom_pairs_and_8_omm_descriptors_cleanly() {
    // 40 triangles with 40 distinct geometry IDs -> forces splitting across blocks (>32 pairs)
    let mut positions = Vec::new();
    let mut triangles = Vec::new();
    let mut metadata = Vec::new();
    let mut omm_ids = Vec::new();
    for i in 0..40u32 {
        let x = i as f32;
        let base = positions.len() as u32;
        positions.push([x, 0.0, 0.0]);
        positions.push([x + 0.5, 0.0, 0.0]);
        positions.push([x, 0.5, 0.0]);
        triangles.push([base, base + 1, base + 2]);
        metadata.push(TriangleMetadata {
            geometry_id: i * 17,
            opaque: i % 2 == 0,
        });
        omm_ids.push(Some(i % 10)); // 10 distinct OMM IDs (> 7 per block!)
    }

    let encoded = encode_mesh(
        MeshView {
            positions: &positions,
            triangles: &triangles,
            triangle_metadata: Some(&metadata),
            triangle_omm_ids: Some(&omm_ids),
        },
        &EncodeOptions {
            quantization: QuantizationMode::ExplicitExponent(-4),
            packing: PackingPolicy::Sah,
            ..EncodeOptions::default()
        },
    )
    .unwrap();

    assert!(encoded.blocks.len() >= 2);
    for (bi, block) in encoded.blocks.iter().enumerate() {
        assert_matches_reference(block.as_bytes(), ValidationMode::Asset);
        let dec = decode_block(&validate_block(block.as_bytes()).unwrap()).unwrap();
        assert_eq!(
            usize::from(dec.omm.descriptor_count),
            encoded.block_omm_maps[bi].descriptors.len()
        );
        assert!(dec.omm.descriptor_count <= 7);
    }

    // Single triangle spanning > 65535 grid steps returns SingleTriangleDoesNotFit
    let huge_tri_pos = [[0.0, 0.0, 0.0], [100_000.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let err = encode_mesh(
        MeshView {
            positions: &huge_tri_pos,
            triangles: &[[0, 1, 2]],
            triangle_metadata: None,
            triangle_omm_ids: None,
        },
        &EncodeOptions {
            quantization: QuantizationMode::ExplicitExponent(0),
            packing: PackingPolicy::Sah,
            ..EncodeOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        err,
        EncodeError::SingleTriangleDoesNotFit {
            input_triangle: 0,
            reason: FitFailure::PositionSpanTooLarge { axis: 0, .. }
        }
    ));
}
