# DGF1 Compatibility, Provenance, and Discrepancy Ledger

## Pinned Research Baseline

| Source | Revision / Identifier | Role |
|---|---|---|
| AMD DGF-SDK | `d97e46493b92d7b2e4445f740fc9dcc7ff53aff3` (6 May 2026) | Format behavior & reference algorithms (studied only; zero C++ or FFI dependency) |
| Khronos Vulkan-Docs | `33d3353757d83d6aca683120114fe74cc4eaf8fb` | Primary DGF1 specification (`dense_geometry_format.adoc`, CC-BY-4.0) |
| Khronos DGF chapter blob | `cd9f2bc0d03dc9fdea253dc7127ff47ac5cbfa5f` | Exact inspected chapter content |
| mxpv/openusd | `900ea31591348e3fbf416d4661d703aa8700771b` (aligned with `openusd-rs/bevy_openusd`) | `crates/openusd-dgf` adapter crate |
| HPG 2024 paper | DOI `10.1145/3675383` | Design rationale |

## Dependency & Safety Policy

- **Pure Rust throughout:** zero C/C++, FFI, `-sys` crates, CMake, bindgen, native fuzz engines, or external encoder binaries in build, runtime, or required tests.
- **Minimal audited dependencies:** `rayon` (pure Rust, optional, enabled by the default `parallel` feature); disabling `parallel` (`--no-default-features --features encode` or `--no-default-features`) has zero external dependencies.
- **`#![forbid(unsafe_code)]`** enforced crate-wide.
- **`no_std` block codec:** disabling default features (`--no-default-features`) builds the zero-allocation block validator and decoder without `std`.

## Validation & Interoperability Evidence Level

This release is **spec-tested and independently Rust-cross-validated**:
- Checked against hand-derived starter fixtures (including exact SHA-256 verification of the Section 21 starter block and feature fixtures covering EDGE1, RESTART, BACKTRACK, 3-bit reuse, 1-entry 25-bit prefix geometry palette, user data + palette, and OMM descriptors).
- Cross-checked against an independent, minimal per-bit Rust reference interpreter in `tests/reference/` that shares no layout or bit-extraction code with the production crate.
- Stress-tested by deterministic property tests (`tests/properties.rs`) and a pure-Rust malformed-input mutation campaign (`tests/mutation.rs` and `tools/mutate/`).
- **External AMD SDK differential fixtures:** none are bundled in this repository yet. No claim of externally executed AMD SDK or GPU hardware differential testing is made until externally generated fixtures with full provenance and checksums are checked in.

## Discrepancy Ledger

Every observed discrepancy between pinned sources and the behavior chosen by this implementation:

| ID | Sources / Symbols | Observed Discrepancy | Chosen Behavior | Regression Coverage |
|---|---|---|---|---|
| D01 | Khronos `dense_geometry_format.adoc` vs AMD `DGF.cpp::ComputeOMMPaletteBitSize` | Khronos prose/table specify `8 + 4*D` hot-patch bytes (`32*(2+D)` bits), whereas one pseudocode line writes `64 + 32*numTriangles`. | Follow the consistent Khronos prose/table and AMD `ComputeOMMPaletteBitSize`: `32*(2+D) + align8(T * ceil_log2(D))`. | `tests/fixtures_test.rs` OMM counts 0..=7 |
| D02 | Khronos `dense_geometry_format.adoc` vs AMD `DGF.cpp::DecodeGeomIDPalette` | Khronos palette pseudocode mixes a 2-arg `ReadBits` declaration with a 3-arg call site. | Address payload entry `i` at `payload_base + i * (25 - prefix_bits)`. Handle `prefix_bits == 25` (`payload_bits == 0`) and `entry_count == 1` (`selector_bits == 0`) explicitly. | `tests/fixtures_test.rs` 1-entry 25-bit prefix fixture |
| D03 | AMD `DGF.cpp::DecodeMetaData` vs `DecodeGeomIDPalette` | `DecodeMetaData` reads palette prefix immediately after header, whereas `DecodeGeomIDPalette` accounts for user data, vertex offsets, and OMM palette. | Use one central `FrontLayout` (`src/layout.rs`) for all section offsets; never read geometry palette data from header-relative shortcuts. | `tests/fixtures_test.rs` user data + OMM + geometry palette |
| D04 | AMD `DGF.cpp::ReadBits` | Reference helper computes `len - 1` assuming valid nonzero widths and trusted input. | Explicitly support `width == 0` returning `0` without indexing input, bounds-check `start + width <= 1024`, and never shift by machine word width. | `src/bits.rs` unit tests |
| D05 | AMD `DGF.cpp::EncodeTopology` | Local assertion does not by itself prove global front/repeat/tail non-overlap. | Enforce `front_bits <= 768`, `repeat_bits <= 192`, and `repeat_end <= tail_start` centrally in `BlockLayout`. | `src/layout.rs` & overlap fixtures |
| D06 | AMD `SAHBlockPacker.cpp::BuildBlock` | Uses `numVerts >= config.blockMaxVerts` in one check and adjusts trailing zeros on coordinates. | Allow full 64-vertex blocks legally; guard trailing-zero shift explicitly when all coordinates are zero (`0u32.trailing_zeros()` never shifts by word size). | `src/encode/block.rs` tests |
| D07 | AMD `DGFDecompression.hlsl::DGFLoadBlockInfo` | HLSL sample omits OMM palette offset between vertices and geometry palette. | CPU codec implements full OMM layout and offsets; HLSL sample is not treated as an OMM oracle. | `tests/fixtures_test.rs` OMM + geometry palette |
| D08 | AMD `MaterialCompression.h::GetMaterialIdsBitSize` vs `DGF.cpp` | Baker helper computes `size += 8 - (size % 8)` (adding 8 bits when already byte-aligned), while serializer pads only when unaligned. | Use exact `align8(n) = (n + 7) & !7` in both planner and decoder. | `tests/fixtures_test.rs` aligned & unaligned palettes |
| D09 | AMD `Quantization.cpp` / `SIMD.h::RoundToInt` | C++ SSE rounding depends on ambient floating-point rounding mode. | Define deterministic IEEE-754 round-ties-to-even in `f64` for `f32` inputs; never depend on ambient FP environment. | `src/encode/quantize.rs` tests |
| D10 | Unused trailing stored vertices & unused gap bits | Encoder never emits unused trailing vertices or nonzero gap bits, whereas the wire format only mandates zero padding in named sections and valid index bounds `< num_vertices`. | Decoder validates `index < next_new_vertex <= num_vertices` and checks named section padding, allowing valid non-canonical blocks with unused trailing vertices or nonzero gap bits. | `tests/fixtures_test.rs` non-canonical valid fixtures |

## Format Support Matrix (DGF1 Profile)

| Feature | Decode | Encode | Notes |
|---|---|---|---|
| 128-byte DGF1 block validation & decode | Yes | Yes | Zero-allocation `no_std` decoder |
| Signed 24-bit anchors & 1..=16-bit offsets | Yes | Yes | Deterministic 4-bit width-sum alignment |
| Biased exponents `1..=232` & overflow checks | Yes | Yes | Explicit exponent, max coordinate error, target bit-width |
| All 4 topology controls (`RESTART`, `EDGE1`, `EDGE2`, `BACKTRACK`) | Yes | Yes | Cyclic winding-preserving strip builder |
| Compressed index stream (up to 192 strip indices, 3..=6-bit reuse) | Yes | Yes | Tested above 64 first-use flags |
| Constant & palette geometry IDs + opacity (up to 32 pairs, prefix 0..=25) | Yes | Yes | Deduplicates `(geometry_id, opaque)` pairs |
| Optional 4-byte user data | Yes | Yes | Per-block callback / assignment |
| OMM descriptors `0..=7`, selectors, opaque hot-patch bytes | Yes | Yes | Asset mode checks zero patch bytes; Runtime mode preserves patched bytes |
| Primitive ID base (29-bit checked range) | Yes | Yes | Checked prefix sums in mesh baker |
| Triangle / corner / vertex remaps & attribute helpers | N/A | Yes | Mandatory output maps; no hidden welding |
| Spatial clustering & exact-fit SAH packing | N/A | Yes | Explicit work stack, `f64` SAH, preclustered input API |
| Deterministic parallel cluster packing (`parallel` feature) | N/A | Yes | Pure-Rust `rayon`, enabled by default, identical output to serial |
| DGFS supercompression | No | No | Explicitly deferred outside DGF1 |
| GPU / Vulkan hardware decoding | No | No | Explicitly deferred outside CPU codec |
| OpenUSD adapter (`crates/openusd-dgf`) | Yes | Yes | Static triangle meshes (`subdivisionScheme = none`), subsets, primvars/UV seams, content-addressed cache (`UsdDgfCache`); `crates/dgf` remains USD-free |

## Canonical Encoder Policy Version

- Wire format: **DGF1** (`magic = 0x06`)
- Canonical Rust encoder policy: **`dgf-rs-v1`** (deterministic width padding order X -> Y -> Z, deterministic first-occurrence palette order, deterministic round-ties-to-even quantization, deterministic SAH tie breaking and low-valence strip ordering). Wire compatibility is independent of future canonical encoder policy updates.

## Round-Trip Contract and Limitations

Encoding a mesh to DGF1 and decoding back is intentionally **not** a byte-for-byte or vertex-ID-preserving round trip of arbitrary source meshes:
1. Vertex positions are quantized to `q * 2^e`; error is bounded by the chosen exponent or `MaxCoordError` tolerance.
2. Triangle order and cyclic corner order (`[0,1,2]`, `[1,2,0]`, `[2,0,1]`) may change; winding is strictly preserved.
3. Source vertices shared across blocks are duplicated per block with identical quantized coordinates.
4. Original vertex IDs, triangle IDs, unused vertices, and vertex/face/corner attributes (normals, UVs, colors) require the returned `vertex_remap` and `triangle_remap` sideband maps.
5. `decode_mesh` preserves block-local vertex separation by default; global vertex reconstruction requires verified remap data via `reconstruct_with_remap`.
