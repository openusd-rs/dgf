# dgf

Pure Rust **Dense Geometry Format (DGF1)** block validator, decoder, and triangle mesh encoder.

## Properties & Guarantees

- **100% pure Rust:** zero C++, FFI, `-sys` crates, build scripts, or external crate dependencies (`[dependencies]` and `[dev-dependencies]` are empty).
- **Memory safe:** `#![forbid(unsafe_code)]` across the entire codebase.
- **Allocation-free `no_std` block decoder:** build with `--no-default-features` for a zero-allocation DGF1 validator and scalar decoder.
- **Full DGF1 block codec:** signed 24-bit anchors, 1..=16-bit offsets, biased exponents `1..=232`, all four topology controls (`RESTART`, `EDGE1`, `EDGE2`, `BACKTRACK`), up to 192 strip indices, constant and palette geometry IDs/opacity (up to 32 pairs, 0..=25-bit prefix), optional 4-byte user data, and OMM palettes (`0..=7` descriptors, opaque hot-patch preservation in runtime mode, zero-check in fresh asset mode).
- **Mesh baking & mappings:** deterministic round-ties-to-even quantization (`ExplicitExponent`, `MaxCoordError`, `TargetBitWidth`), directed-edge adjacency and cyclic winding-preserving strip builder, coarse spatial clustering + exact-fit `f64` SAH subdivision, mandatory vertex/triangle/corner/OMM remaps, attribute remapping helpers, and deterministic multi-threaded packing (`parallel` feature using `std::thread::scope`).

## Cargo Features

| Feature | Default | Description |
|---|---|---|
| `std` | Yes | Multi-block stream/mesh decoding (`decode_blocks`, `decode_stream`, `decode_mesh`) and remap helpers |
| `encode` | Yes | Block planner/encoder, quantization, strip builder, SAH mesh baker, CLI |
| `parallel` | No | Deterministic multi-threaded coarse-cluster packing (bit-for-bit identical to serial output) |

Disabling all features (`--no-default-features`) compiles the `#![no_std]` block validator and decoder.

## Quick Verification & Commands

```sh
# Format, check, clippy, docs, and tests across feature combinations
cargo fmt --all -- --check
cargo check --no-default-features
cargo check --no-default-features --features std
cargo check --all-targets --all-features
cargo clippy --all-targets --all-features -- -Dwarnings
RUSTDOCFLAGS=-Dwarnings cargo doc --all-features --no-deps
cargo test --all-targets --all-features
cargo test --release --all-targets --all-features

# End-to-end demo & CLI
cargo run --example main
cargo run --bin dgf-cli -- --help

# Pure-Rust mutation campaign
cargo run --manifest-path tools/mutate/Cargo.toml --release -- 100000

# Benchmarks (11-class synthetic corpus)
cargo bench --all-features
```

## Documentation & Interoperability Evidence

- Format specification summary: [docs/format.md](docs/format.md)
- Pinned upstream sources, discrepancy ledger, support matrix, and round-trip contract: [docs/compatibility.md](docs/compatibility.md)
- Evidence status: **spec-tested and independently Rust-cross-validated** (including exact SHA-256 verification of the hand-derived starter fixture, an independent per-bit Rust reference interpreter in `tests/reference/`, property tests in `tests/properties.rs`, and pure-Rust mutation campaigns in `tests/mutation.rs` and `tools/mutate/`). External AMD SDK differential fixtures are not bundled in this repository.
