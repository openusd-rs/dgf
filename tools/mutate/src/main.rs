//! Standalone pure-Rust malformed-input mutation campaign runner (no libFuzzer / C++ required).
//!
//! Usage:
//!   cargo run --manifest-path tools/mutate/Cargo.toml --release -- [iterations] [seed]

#![forbid(unsafe_code)]

use std::env;
use std::time::Instant;

use dgf::{
    BiasedExponent, PreparedBlock, PreparedOmm, PreparedTopology, TriControl, ValidationMode,
    decode_block, encode_block, validate_block_with_mode,
};

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

fn main() {
    let args: Vec<String> = env::args().collect();
    let iterations: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let seed: u64 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xD6F1_5EAF_00D0_0001);

    let seed_block = encode_block(&PreparedBlock {
        quantized_positions: vec![[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0], [-5, 10, 0]],
        exponent: BiasedExponent::new(124).unwrap(),
        topology: PreparedTopology {
            controls: vec![
                TriControl::Restart,
                TriControl::Edge1,
                TriControl::Backtrack,
            ],
            strip_indices: vec![0, 1, 2, 3, 4],
            num_vertices: 5,
        },
        geometry_ids: vec![1, 512, 1],
        opaque: vec![true, false, true],
        prim_id_base: 0,
        user_data: Some(0x1234_5678),
        omm: Some(PreparedOmm {
            descriptor_count: 3,
            triangle_descriptors: vec![0, 1, 2],
            hot_patch_bytes: None,
        }),
        allow_exponent_adjustment: false,
    })
    .expect("build seed block");

    let mut rng = Rng(seed);
    let mut valid_count = 0usize;
    let mut invalid_count = 0usize;
    let start = Instant::now();

    for iter in 0..iterations {
        let mut block = *seed_block.as_bytes();
        if iter % 5 == 0 {
            for chunk in block.chunks_exact_mut(8) {
                chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
            block[0] = 0x06;
            block[19] &= 0x3F;
        } else {
            let flips = 1 + rng.next_usize(8);
            for _ in 0..flips {
                let idx = rng.next_usize(128);
                if rng.next_usize(2) == 0 {
                    block[idx] ^= 1 << rng.next_usize(8);
                } else {
                    block[idx] = rng.next_u8();
                }
            }
        }

        match validate_block_with_mode(&block, ValidationMode::Runtime)
            .and_then(|v| decode_block(&v))
        {
            Ok(dec) => {
                // Ensure bounded output on any valid decode
                assert!(dec.num_vertices <= 64);
                assert!(dec.num_triangles <= 64);
                valid_count += 1;
            }
            Err(_) => {
                invalid_count += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    println!(
        "Mutation campaign finished: {iterations} cases (seed=0x{seed:016x}) in {:.3}s ({:.0} blocks/s): valid={valid_count}, rejected={invalid_count}, panics=0",
        elapsed.as_secs_f64(),
        iterations as f64 / elapsed.as_secs_f64().max(1e-9)
    );
}
