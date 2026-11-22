//! Pure-Rust malformed-input mutation smoke campaign (PLAN.md Section 13.5).
//!
//! Mutates valid seed blocks across every structural field category, generates arbitrary
//! 128-byte inputs, exercises stream lengths/limits, and verifies that the production
//! decoder NEVER panics and ALWAYS agrees with the independent reference interpreter.

#![forbid(unsafe_code)]

mod reference;

use dgf::{
    BiasedExponent, DecodeLimits, EncodeOptions, MeshView, PackingPolicy, PreparedBlock,
    PreparedOmm, PreparedTopology, QuantizationMode, TriControl, TriangleMetadata, ValidationMode,
    decode_mesh, decode_stream, encode_block, encode_mesh, validate_block,
};
use reference::assert_matches_reference;

struct Rng(u64);
impl Rng {
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
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }
}

fn seed_blocks() -> Vec<[u8; 128]> {
    let mut seeds = Vec::new();

    // 1. Minimal single triangle
    let b1 = encode_block(&PreparedBlock {
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
    })
    .unwrap();
    seeds.push(*b1.as_bytes());

    // 2. Complex block with user data + OMM (5 descriptors) + geometry palette + BACKTRACK + reuse
    let b2 = encode_block(&PreparedBlock {
        quantized_positions: vec![
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [10, 10, 0],
            [-5, 10, 2],
            [5, 20, -3],
        ],
        exponent: BiasedExponent::new(120).unwrap(),
        topology: PreparedTopology {
            controls: vec![
                TriControl::Restart,
                TriControl::Edge1,
                TriControl::Backtrack,
                TriControl::Edge2,
                TriControl::Edge1,
            ],
            strip_indices: vec![0, 1, 2, 3, 4, 5, 0],
            num_vertices: 6,
        },
        geometry_ids: vec![10, 20, 10, 5000, 20],
        opaque: vec![true, false, false, true, false],
        prim_id_base: 12345,
        user_data: Some(0xCAFE_BABE),
        omm: Some(PreparedOmm {
            descriptor_count: 5,
            triangle_descriptors: vec![0, 1, 2, 3, 4],
            hot_patch_bytes: None,
        }),
        allow_exponent_adjustment: false,
    })
    .unwrap();
    seeds.push(*b2.as_bytes());

    // 3. Dense SAH block from a small grid
    let mut pos = Vec::new();
    let mut tris = Vec::new();
    let mut meta = Vec::new();
    for y in 0..=4 {
        for x in 0..=4 {
            pos.push([x as f32, y as f32, ((x * y) % 3) as f32 * 0.25]);
        }
    }
    for y in 0..4 {
        for x in 0..4 {
            let a = (y * 5 + x) as u32;
            tris.push([a, a + 1, a + 6]);
            meta.push(TriangleMetadata {
                geometry_id: (x + y) as u32,
                opaque: true,
            });
            tris.push([a, a + 6, a + 5]);
            meta.push(TriangleMetadata {
                geometry_id: 600 + (x as u32),
                opaque: false,
            });
        }
    }
    let enc = encode_mesh(
        MeshView {
            positions: &pos,
            triangles: &tris,
            triangle_metadata: Some(&meta),
            triangle_omm_ids: None,
        },
        &EncodeOptions {
            quantization: QuantizationMode::ExplicitExponent(-4),
            packing: PackingPolicy::Sah,
            ..EncodeOptions::default()
        },
    )
    .unwrap();
    for b in enc.blocks {
        seeds.push(*b.as_bytes());
    }

    seeds
}

#[test]
fn mutation_single_bit_flips_all_1024_bits_on_seed_blocks() {
    let seeds = seed_blocks();
    for seed in &seeds {
        for bit in 0..1024 {
            let mut mutated = *seed;
            mutated[bit / 8] ^= 1 << (bit % 8);
            assert_matches_reference(&mutated, ValidationMode::Runtime);
            assert_matches_reference(&mutated, ValidationMode::Asset);
        }
    }
}

#[test]
fn mutation_structured_havoc_and_arbitrary_128_byte_campaign() {
    let seeds = seed_blocks();
    let mut rng = Rng(0x5EED_F00D_BAAD_CAFE);

    // 1. Structured multi-mutation havoc (10,000 cases) targeting headers, counts, widths,
    // prefixes, controls, first-use flags, selectors, primitive bases, and section boundaries.
    for _ in 0..10_000 {
        let mut block = seeds[rng.next_usize(seeds.len())];
        let edits = 1 + rng.next_usize(6);
        for _ in 0..edits {
            match rng.next_usize(6) {
                0 => {
                    // Single bit flip anywhere
                    let bit = rng.next_usize(1024);
                    block[bit / 8] ^= 1 << (bit % 8);
                }
                1 => {
                    // Mutate header byte (keeping magic 0x06 80% of the time to reach deep parsers!)
                    let byte = if rng.next_usize(5) == 0 {
                        0
                    } else {
                        1 + rng.next_usize(19)
                    };
                    block[byte] = rng.next_u8();
                }
                2 => {
                    // Mutate backward control / first-use tail bytes
                    let byte = 96 + rng.next_usize(32);
                    block[byte] = rng.next_u8();
                }
                3 => {
                    // Mutate front payload / palette / repeat region
                    let byte = 20 + rng.next_usize(76);
                    block[byte] = rng.next_u8();
                }
                4 => {
                    // Extreme header values (counts=64, widths=16, exp=232, prim_base=MAX)
                    match rng.next_usize(4) {
                        0 => block[4] = 232,    // max exponent
                        1 => block[8] = 0xFF,   // x_bits=16, y_bits=16
                        2 => block[12] |= 0xFF, // z_bits=16, omm=7, palette mode
                        _ => block[16..20].copy_from_slice(&0x1FFF_FFFFu32.to_le_bytes()),
                    }
                }
                _ => {
                    // Zero or 0xFF a small slice
                    let start = rng.next_usize(124);
                    let len = 1 + rng.next_usize(4);
                    let fill = if rng.next_usize(2) == 0 { 0x00 } else { 0xFF };
                    block[start..start + len].fill(fill);
                }
            }
        }
        assert_matches_reference(&block, ValidationMode::Runtime);
        assert_matches_reference(&block, ValidationMode::Asset);
    }

    // 2. Completely arbitrary random 128-byte arrays (5,000 cases, forcing magic=0x06 half the time)
    for i in 0..5_000 {
        let mut block = [0u8; 128];
        for chunk in block.as_chunks_mut::<8>().0 {
            *chunk = rng.next_u64().to_le_bytes();
        }
        if i % 2 == 0 {
            block[0] = 0x06;
            block[19] &= 0x3F; // clear reserved bits to dig deeper
        }
        let _ = validate_block(&block);
        assert_matches_reference(&block, ValidationMode::Runtime);
    }

    // 3. Arbitrary stream lengths 0..=300
    let mut stream = vec![0u8; 300];
    for b in &mut stream {
        *b = rng.next_u8();
    }
    for len in 0..=300 {
        let _ = decode_stream(&stream[..len], &DecodeLimits::default());
        let _ = decode_mesh(&stream[..len], &DecodeLimits::default());
    }
}
