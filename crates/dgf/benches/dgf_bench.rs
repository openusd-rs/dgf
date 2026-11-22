//! Reproducible pure-Rust benchmark suite covering all 11 corpus classes (PLAN.md Section 14).
//!
//! Run with:
//!   cargo bench --all-features

#![forbid(unsafe_code)]

use std::time::Instant;

use dgf::{
    DecodeLimits, DegeneratePolicy, EncodeOptions, MeshView, PackingPolicy, QuantizationMode,
    TriangleMetadata, decode_mesh, encode_mesh,
};

struct CorpusEntry {
    name: &'static str,
    positions: Vec<[f32; 3]>,
    triangles: Vec<[u32; 3]>,
    metadata: Option<Vec<TriangleMetadata>>,
}

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
        let u = ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32);
        lo + u * (hi - lo)
    }
}

fn make_grid(side: usize, displace: bool) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut positions = Vec::with_capacity((side + 1) * (side + 1));
    for y in 0..=side {
        for x in 0..=side {
            let fx = x as f32 / side as f32 * 10.0;
            let fy = y as f32 / side as f32 * 10.0;
            let fz = if displace {
                (fx * 0.7).sin() * (fy * 0.9).cos() * 1.5 + (fx * 2.3 + fy * 1.7).sin() * 0.2
            } else {
                0.0
            };
            positions.push([fx, fy, fz]);
        }
    }
    let mut triangles = Vec::with_capacity(side * side * 2);
    for y in 0..side {
        for x in 0..side {
            let a = (y * (side + 1) + x) as u32;
            let b = a + 1;
            let c = ((y + 1) * (side + 1) + x) as u32;
            let d = c + 1;
            triangles.push([a, b, d]);
            triangles.push([a, d, c]);
        }
    }
    (positions, triangles)
}

fn make_sphere(stacks: usize, slices: usize) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut positions = Vec::with_capacity((stacks + 1) * (slices + 1));
    for s in 0..=stacks {
        let phi = std::f32::consts::PI * (s as f32) / (stacks as f32);
        for l in 0..=slices {
            let theta = 2.0 * std::f32::consts::PI * (l as f32) / (slices as f32);
            positions.push([
                phi.sin() * theta.cos() * 5.0,
                phi.cos() * 5.0,
                phi.sin() * theta.sin() * 5.0,
            ]);
        }
    }
    let mut triangles = Vec::with_capacity(stacks * slices * 2);
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
    (positions, triangles)
}

fn build_corpus() -> Vec<CorpusEntry> {
    let mut corpus = Vec::new();
    let mut rng = Rng(0xBE4C_4001);

    // 1. Minimal quad
    corpus.push(CorpusEntry {
        name: "two_tri_quad",
        positions: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        triangles: vec![[0, 1, 2], [2, 1, 3]],
        metadata: None,
    });

    // 2. Regular grid 32x32 (2048 tris)
    let (gp, gt) = make_grid(32, false);
    corpus.push(CorpusEntry {
        name: "regular_grid_32x32",
        positions: gp,
        triangles: gt,
        metadata: None,
    });

    // 3. Closed sphere (2048 tris)
    let (sp, st) = make_sphere(32, 32);
    corpus.push(CorpusEntry {
        name: "closed_sphere_32x32",
        positions: sp,
        triangles: st,
        metadata: None,
    });

    // 4. Terrain patch (displaced 32x32)
    let (tp, tt) = make_grid(32, true);
    corpus.push(CorpusEntry {
        name: "terrain_patch_32x32",
        positions: tp,
        triangles: tt,
        metadata: None,
    });

    // 5. CAD-style anisotropic thin ribs
    let (mut cp, ct) = make_grid(24, false);
    for p in &mut cp {
        p[0] *= 10.0;
        p[1] *= 0.05;
        p[2] = if (p[0] as i32) % 2 == 0 { 0.01 } else { 0.0 };
    }
    corpus.push(CorpusEntry {
        name: "cad_anisotropic_thin",
        positions: cp,
        triangles: ct,
        metadata: None,
    });

    // 6. Dense scanned surface (64x64 = 8192 tris)
    let (dp, dt) = make_grid(64, true);
    corpus.push(CorpusEntry {
        name: "dense_surface_64x64",
        positions: dp,
        triangles: dt,
        metadata: None,
    });

    // 7. Many disconnected tiny triangles (1024 tris)
    let mut disc_p = Vec::with_capacity(3072);
    let mut disc_t = Vec::with_capacity(1024);
    for _ in 0..1024 {
        let b = disc_p.len() as u32;
        let cx = rng.next_f32(-10.0, 10.0);
        let cy = rng.next_f32(-10.0, 10.0);
        let cz = rng.next_f32(-10.0, 10.0);
        disc_p.push([cx, cy, cz]);
        disc_p.push([cx + 0.05, cy, cz]);
        disc_p.push([cx, cy + 0.05, cz]);
        disc_t.push([b, b + 1, b + 2]);
    }
    corpus.push(CorpusEntry {
        name: "disconnected_soup_1k",
        positions: disc_p,
        triangles: disc_t,
        metadata: None,
    });

    // 8. Duplicate faces & non-manifold edges
    let (nm_p, mut nm_t) = make_grid(16, false);
    let orig_len = nm_t.len();
    for i in 0..orig_len {
        nm_t.push(nm_t[i]); // duplicate face
    }
    corpus.push(CorpusEntry {
        name: "non_manifold_dup_1k",
        positions: nm_p,
        triangles: nm_t,
        metadata: None,
    });

    // 9. Randomized material IDs (palette pressure)
    let (mp, mt) = make_grid(24, true);
    let mmeta: Vec<TriangleMetadata> = (0..mt.len())
        .map(|_| TriangleMetadata {
            geometry_id: (rng.next_u64() % 64) as u32,
            opaque: (rng.next_u64() & 1) == 0,
        })
        .collect();
    corpus.push(CorpusEntry {
        name: "random_materials_64",
        positions: mp,
        triangles: mt,
        metadata: Some(mmeta),
    });

    // 10. Equal centroids / zero-area SAH fallback
    let eq_p = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    let eq_t: Vec<[u32; 3]> = (0..512)
        .map(|i| [0, (1 + (i % 4)) as u32, (1 + ((i + 1) % 4)) as u32])
        .collect();
    corpus.push(CorpusEntry {
        name: "equal_centroids_512",
        positions: eq_p,
        triangles: eq_t,
        metadata: None,
    });

    corpus
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn main() {
    let corpus = build_corpus();
    let policies = [
        (
            "baseline_restart",
            PackingPolicy::BaselineRestartOnly,
            false,
        ),
        ("baseline_strips", PackingPolicy::BaselineWithStrips, false),
        ("sah_serial", PackingPolicy::Sah, false),
        ("sah_parallel", PackingPolicy::Sah, true),
    ];

    println!(
        "{:<22} {:<17} {:>6} {:>6} {:>7} {:>6} {:>7} {:>7} {:>9} {:>10} {:>10}",
        "corpus",
        "policy",
        "tris",
        "blks",
        "B/tri",
        "vdup",
        "T/blk",
        "V/blk",
        "area",
        "enc_ktri/s",
        "dec_ktri/s"
    );
    println!("{}", "-".repeat(122));

    for entry in &corpus {
        let view = MeshView {
            positions: &entry.positions,
            triangles: &entry.triangles,
            triangle_metadata: entry.metadata.as_deref(),
            triangle_omm_ids: None,
        };

        for &(label, packing, parallel) in &policies {
            let opts = EncodeOptions {
                quantization: QuantizationMode::MaxCoordError(1e-3),
                packing,
                degenerate_policy: DegeneratePolicy::DropAndReport,
                parallel,
                ..EncodeOptions::default()
            };

            // Warmup
            let encoded = encode_mesh(view, &opts).expect("bench encode");
            let raw = encoded.to_bytes();
            let _ = decode_mesh(&raw, &DecodeLimits::default()).expect("bench decode");

            let samples = 5usize;
            let mut enc_times = Vec::with_capacity(samples);
            let mut dec_times = Vec::with_capacity(samples);

            for _ in 0..samples {
                let t0 = Instant::now();
                let e = encode_mesh(view, &opts).expect("bench encode");
                enc_times.push(t0.elapsed().as_secs_f64());
                std::hint::black_box(&e.blocks);

                let t1 = Instant::now();
                let d = decode_mesh(&raw, &DecodeLimits::default()).expect("bench decode");
                dec_times.push(t1.elapsed().as_secs_f64());
                std::hint::black_box(&d.triangles);
            }

            let enc_med = median(enc_times).max(1e-9);
            let dec_med = median(dec_times).max(1e-9);
            let tris = entry.triangles.len() as f64;
            let enc_ktps = (tris / enc_med) / 1000.0;
            let dec_ktps = (tris / dec_med) / 1000.0;

            let s = encoded.statistics;
            println!(
                "{:<22} {:<17} {:>6} {:>6} {:>7.2} {:>6.2} {:>7.1} {:>7.1} {:>9.1} {:>10.1} {:>10.1}",
                entry.name,
                label,
                s.retained_triangles,
                s.output_blocks,
                s.bytes_per_triangle,
                s.vertex_duplication_factor,
                s.mean_triangles_per_block,
                s.mean_vertices_per_block,
                s.total_block_surface_area,
                enc_ktps,
                dec_ktps
            );
        }
    }

    // Class 11: Verify very large translated coordinates produce an actionable precision diagnostic.
    let huge_pos = [
        [1.0e8f32, 0.0, 0.0],
        [1.0e8 + 0.001, 0.0, 0.0],
        [1.0e8, 0.001, 0.0],
    ];
    let huge_res = encode_mesh(
        MeshView {
            positions: &huge_pos,
            triangles: &[[0, 1, 2]],
            triangle_metadata: None,
            triangle_omm_ids: None,
        },
        &EncodeOptions {
            quantization: QuantizationMode::MaxCoordError(1e-4),
            ..EncodeOptions::default()
        },
    );
    println!();
    println!(
        "Large translated coordinate diagnostic: {}",
        huge_res.expect_err("expected diagnostic error")
    );
}
