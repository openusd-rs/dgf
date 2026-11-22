# openusd-dgf

Pure-Rust **Dense Geometry Format (DGF1)** workspace providing both an engine-independent DGF1 codec (`dgf`) and an OpenUSD adapter (`openusd-dgf`).

## Workspace Crates

| Crate | Path | Responsibility | Dependencies |
|---|---|---|---|
| `dgf` | [crates/dgf](crates/dgf) | Standalone DGF1 128-byte block validator, `#![no_std]` scalar decoder, embedded WGSL GPU compute decoder (`WGSL_DECODER_SOURCE`), quantizer, strip builder, SAH mesh baker, remap helpers, and `dgf-cli` | Pure Rust; optional `rayon` under default `parallel` feature (zero deps without `parallel`) |
| `openusd-dgf` | [crates/openusd-dgf](crates/openusd-dgf) | OpenUSD `UsdGeomMesh` extraction, independent topology validation, material subset → `geometry_id` mapping, face-varying UV/normal/primvar sideband remapping, and content-addressed geometry caching (`UsdDgfCache`) | `dgf`, `openusd`, `openusd-schemas` (pinned to `mxpv/openusd@900ea315...`, matching `openusd-rs/bevy_openusd`) |

## Guarantees

- **100% pure Rust:** zero C++, FFI, `-sys` wrappers, or native build scripts in build, runtime, or required tests.
- **Memory safe:** `#![forbid(unsafe_code)]` enforced across both crates.
- **Allocation-free `no_std` block decoder:** `cargo check -p dgf --no-default-features` builds the DGF1 block validator and decoder without `std`.
- **Pure WGSL GPU compute decoder:** `dgf::WGSL_DECODER_SOURCE` (`crates/dgf/shaders/dgf_decode.wgsl`) implements a strictly 32-bit, zero-extension WebGPU/Vulkan/Metal/DX12 scalar DGF1 block decoder verified bit-for-bit against CPU `dgf::decode_block`.
- **Deterministic parallelism:** `rayon` parallel cluster packing is enabled by default in `dgf` and produces bit-for-bit identical output to serial execution.
- **OpenUSD provenance & caching:** `openusd-dgf` preserves UV seams (face-varying indexed primvars), normals, `UsdGeomSubset` material bindings, `orientation`, `doubleSided`, stage `upAxis`/`metersPerUnit`, and world transforms separately from geometry so transform edits hit the cache without rebaking.

## `dgf` Cargo Features

| Feature | Default | Description |
|---|---|---|
| `std` | Yes | Multi-block stream/mesh decoding (`decode_blocks`, `decode_stream`, `decode_mesh`) and remap helpers |
| `encode` | Yes | Block planner/encoder, quantization, strip builder, SAH mesh baker, CLI |
| `parallel` | Yes | Deterministic multi-threaded coarse-cluster packing via `rayon` |

## Verification & Commands

```sh
# Format, feature matrix, clippy, docs, and tests across the workspace
cargo fmt --all -- --check
cargo check -p dgf --no-default-features
cargo check -p dgf --no-default-features --features std
cargo check -p dgf --no-default-features --features encode
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -Dwarnings
RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --all-features --no-deps
cargo test --workspace --all-targets --all-features
cargo test --release --workspace --all-targets --all-features

# End-to-end demo & CLI
cargo run -p dgf --example main
cargo run -p dgf --bin dgf-cli -- --help

# Pure-Rust mutation campaign
cargo run --manifest-path tools/mutate/Cargo.toml --release -- 100000

# Benchmarks (11-class synthetic corpus)
cargo bench -p dgf --all-features
```

## Documentation & Interoperability Evidence

- Format specification summary: [docs/format.md](docs/format.md)
- Pinned upstream sources, discrepancy ledger, support matrix, and round-trip contract: [docs/compatibility.md](docs/compatibility.md)
- Evidence status: **spec-tested and independently Rust-cross-validated** (including exact SHA-256 verification of the hand-derived starter fixture, an independent per-bit Rust reference interpreter in `crates/dgf/tests/reference/`, property tests, pure-Rust mutation campaigns, and in-memory OpenUSD stage integration tests in `crates/openusd-dgf/tests/usd_integration.rs`). External AMD SDK differential fixtures are not bundled in this repository.
