//! Pure Rust CLI to inspect, validate, encode, and decode DGF1 block streams.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use dgf::{
    BLOCK_BYTES, DecodeLimits, EncodeOptions, MeshView, PackingPolicy, QuantizationMode,
    ValidationMode, decode_block, decode_stream, encode_mesh, inspect_header,
    validate_block_with_mode,
};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return ExitCode::from(2);
    }

    match args[1].as_str() {
        "inspect" => cmd_inspect(&args[2..]),
        "validate" => cmd_validate(&args[2..]),
        "demo-encode" => cmd_demo_encode(&args[2..]),
        "--help" | "-h" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn print_usage() {
    println!("dgf-cli — Pure Rust DGF1 inspector, validator, and demo encoder");
    println!();
    println!("Usage:");
    println!(
        "  dgf-cli inspect <file.dgf>                  Inspect block headers and decoded stats"
    );
    println!("  dgf-cli validate [--asset] <file.dgf>       Validate raw DGF1 block stream");
    println!(
        "  dgf-cli demo-encode <side> <out.dgf>        Encode an NxN grid mesh into a DGF1 stream"
    );
}

fn cmd_inspect(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("missing <file.dgf>");
        return ExitCode::from(2);
    };
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    if !data.len().is_multiple_of(BLOCK_BYTES) {
        eprintln!(
            "stream truncated: {} bytes is not a multiple of {BLOCK_BYTES}",
            data.len()
        );
        return ExitCode::FAILURE;
    }

    let num_blocks = data.len() / BLOCK_BYTES;
    println!("File: {path} ({} bytes, {num_blocks} blocks)", data.len());
    for (idx, arr) in data.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let raw = inspect_header(arr);
        match validate_block_with_mode(arr, ValidationMode::Runtime).and_then(|v| decode_block(&v))
        {
            Ok(dec) => {
                println!(
                    "  block[{idx}]: VALID V={} T={} exp={} (2^{}) anchor={:?} widths={:?} omm={} geom={:?} prim_base={} user_data={:?} front={}b repeat={}b tail={}b gap={}b",
                    dec.num_vertices,
                    dec.num_triangles,
                    dec.exponent.raw(),
                    dec.exponent.unbiased().value(),
                    dec.anchor,
                    dec.header.offset_bits,
                    dec.omm.descriptor_count,
                    dec.header.geom_id_mode,
                    dec.prim_id_base,
                    dec.user_data,
                    dec.layout.front.front_bits,
                    dec.layout.repeat_bits,
                    dec.layout.control_bits + dec.layout.first_use_bits,
                    dec.layout.unused_gap_bits()
                );
            }
            Err(e) => {
                println!(
                    "  block[{idx}]: INVALID ({e}) [raw magic=0x{:02x} V={} T={} exp={} reserved={}]",
                    raw.magic, raw.num_vertices, raw.num_triangles, raw.raw_exponent, raw.reserved
                );
            }
        }
    }
    ExitCode::SUCCESS
}

fn cmd_validate(args: &[String]) -> ExitCode {
    let mut mode = ValidationMode::Runtime;
    let mut file: Option<PathBuf> = None;
    for arg in args {
        if arg == "--asset" {
            mode = ValidationMode::Asset;
        } else {
            file = Some(PathBuf::from(arg));
        }
    }
    let Some(path) = file else {
        eprintln!("missing <file.dgf>");
        return ExitCode::from(2);
    };
    let data = match fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("failed to read {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let limits = DecodeLimits {
        validation_mode: mode,
        ..DecodeLimits::default()
    };
    match decode_stream(&data, &limits) {
        Ok(blocks) => {
            let total_tris: usize = blocks.iter().map(|b| usize::from(b.num_triangles)).sum();
            let total_verts: usize = blocks.iter().map(|b| usize::from(b.num_vertices)).sum();
            println!(
                "VALID ({mode:?}): {} blocks, {total_verts} vertices, {total_tris} triangles",
                blocks.len()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("INVALID: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_demo_encode(args: &[String]) -> ExitCode {
    if args.len() < 2 {
        eprintln!("usage: dgf-cli demo-encode <side> <out.dgf>");
        return ExitCode::from(2);
    }
    let side: usize = match args[0].parse() {
        Ok(s) if s > 0 => s,
        _ => {
            eprintln!("invalid grid side: {}", args[0]);
            return ExitCode::from(2);
        }
    };
    let out_path = &args[1];

    let mut positions = Vec::with_capacity((side + 1) * (side + 1));
    for y in 0..=side {
        for x in 0..=side {
            positions.push([x as f32 / side as f32, y as f32 / side as f32, 0.0]);
        }
    }
    let mut triangles = Vec::with_capacity(side * side * 2);
    for y in 0..side {
        for x in 0..side {
            let i00 = (y * (side + 1) + x) as u32;
            let i10 = i00 + 1;
            let i01 = ((y + 1) * (side + 1) + x) as u32;
            let i11 = i01 + 1;
            triangles.push([i00, i10, i11]);
            triangles.push([i00, i11, i01]);
        }
    }

    let mesh = MeshView {
        positions: &positions,
        triangles: &triangles,
        triangle_metadata: None,
        triangle_omm_ids: None,
    };
    let opts = EncodeOptions {
        quantization: QuantizationMode::MaxCoordError(1e-4),
        packing: PackingPolicy::Sah,
        ..EncodeOptions::default()
    };
    let encoded = match encode_mesh(mesh, &opts) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("encode failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let bytes = encoded.to_bytes();
    if let Err(e) = fs::write(out_path, &bytes) {
        eprintln!("failed to write {out_path}: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "Encoded {}x{} grid ({} tris) -> {} blocks ({} bytes, {:.2} B/tri) to {out_path}",
        side,
        side,
        triangles.len(),
        encoded.blocks.len(),
        bytes.len(),
        encoded.statistics.bytes_per_triangle
    );
    ExitCode::SUCCESS
}
