//! End-to-end pure Rust DGF1 example encoding and decoding a generated grid mesh.

#![forbid(unsafe_code)]

use dgf::{
    DecodeLimits, EncodeOptions, MeshView, PackingPolicy, QuantizationMode, TriangleMetadata,
    decode_mesh, encode_mesh, reconstruct_with_remap,
};

fn main() {
    // Generate an 8x8 regular triangle grid (81 vertices, 128 triangles).
    let side = 8usize;
    let mut positions = Vec::with_capacity((side + 1) * (side + 1));
    for y in 0..=side {
        for x in 0..=side {
            let fx = x as f32 / side as f32;
            let fy = y as f32 / side as f32;
            let fz = (fx * std::f32::consts::PI).sin() * (fy * std::f32::consts::PI).sin() * 0.25;
            positions.push([fx, fy, fz]);
        }
    }

    let mut triangles = Vec::with_capacity(side * side * 2);
    let mut metadata = Vec::with_capacity(side * side * 2);
    for y in 0..side {
        for x in 0..side {
            let i00 = (y * (side + 1) + x) as u32;
            let i10 = i00 + 1;
            let i01 = ((y + 1) * (side + 1) + x) as u32;
            let i11 = i01 + 1;
            let geom = ((x + y) % 2) as u32;
            triangles.push([i00, i10, i11]);
            metadata.push(TriangleMetadata {
                geometry_id: geom,
                opaque: true,
            });
            triangles.push([i00, i11, i01]);
            metadata.push(TriangleMetadata {
                geometry_id: geom,
                opaque: true,
            });
        }
    }

    let mesh = MeshView {
        positions: &positions,
        triangles: &triangles,
        triangle_metadata: Some(&metadata),
        triangle_omm_ids: None,
    };

    let options = EncodeOptions {
        quantization: QuantizationMode::MaxCoordError(1e-4),
        packing: PackingPolicy::Sah,
        ..EncodeOptions::default()
    };

    let encoded = encode_mesh(mesh, &options).expect("encode grid mesh");
    let raw_stream = encoded.to_bytes();
    let decoded = decode_mesh(&raw_stream, &DecodeLimits::default()).expect("decode DGF1 stream");
    let reconstructed = reconstruct_with_remap(
        &decoded,
        &encoded.vertex_remap,
        &encoded.triangle_remap,
        positions.len(),
    )
    .expect("reconstruct globally indexed mesh");

    println!("DGF1 pure Rust encode/decode demo:");
    println!(
        "  Input:  {} vertices, {} triangles",
        positions.len(),
        triangles.len()
    );
    println!(
        "  Blocks: {} ({} bytes, {:.2} B/tri)",
        encoded.blocks.len(),
        raw_stream.len(),
        encoded.statistics.bytes_per_triangle
    );
    println!(
        "  Quantization: exponent={} (step={:.6}), max_coord_err={:.6}, max_euclid_err={:.6}",
        encoded.quantization.exponent.value(),
        encoded.quantization.grid_step,
        encoded.quantization.max_coord_error,
        encoded.quantization.max_euclidean_error
    );
    println!(
        "  Topology: restarts={}, edge1={}, edge2={}, backtrack={}, repeated_idx={}",
        encoded.statistics.total_restarts,
        encoded.statistics.total_edge1,
        encoded.statistics.total_edge2,
        encoded.statistics.total_backtracks,
        encoded.statistics.total_repeated_indices
    );
    println!(
        "  Decoded: {} block-local verts, {} tris; reconstructed {} global tris",
        decoded.positions.len(),
        decoded.triangles.len(),
        reconstructed.triangles.len()
    );
}
