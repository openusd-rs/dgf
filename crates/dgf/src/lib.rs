//! Pure Rust Dense Geometry Format (DGF1) block validator, decoder, and mesh encoder.
//!
//! # Guarantees and Design Policy
//!
//! - **Pure Rust:** zero C++, FFI, `-sys` wrappers, or build scripts (only optional `rayon` under the default `parallel` feature).
//! - **Memory safe:** `#![forbid(unsafe_code)]` across the entire crate.
//! - **Allocation-free block decoder:** disabling default features (`--no-default-features`)
//!   provides `#![no_std]` block validation and decoding into fixed-size arrays.
//! - **Explicit contracts:** quantization, deterministic packing, geometry IDs, opacity flags,
//!   OMM layouts, user data, and triangle/corner/vertex remaps are explicit.

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

pub mod bits;
pub mod block;
pub mod decode;
pub mod error;
pub mod header;
pub mod layout;
pub mod palette;
pub mod topology;

#[cfg(feature = "std")]
pub mod remap;

#[cfg(feature = "encode")]
pub mod encode;

pub use bits::{BLOCK_BITS, BLOCK_BYTES};
pub use block::{Block, ValidatedBlock, ValidationMode, validate_block, validate_block_with_mode};
pub use decode::{DecodedBlock, decode_block};
pub use error::{BitWriteError, DecodeError, FitFailure, PaddingSection, StreamDecodeError};
pub use header::{
    ANCHOR_MAX, ANCHOR_MIN, BiasedExponent, EXPONENT_BIAS, GeomIdHeaderMode, HEADER_BITS,
    HEADER_BYTES, Header, MAGIC, MAX_BIASED_EXPONENT, MAX_BITS_PER_INDEX, MAX_CONSTANT_GEOM_ID,
    MAX_GEOM_ID, MAX_GEOM_PALETTE_ENTRIES, MAX_GEOM_PREFIX_BITS, MAX_OMM_DESCRIPTORS, MAX_PRIM_ID,
    MAX_TRIANGLES, MAX_UNBIASED_EXPONENT, MAX_VERTICES, MIN_BIASED_EXPONENT, MIN_BITS_PER_INDEX,
    MIN_UNBIASED_EXPONENT, QuantizationExponent, RawHeaderInspection, inspect_header,
    pack_signed_24, sign_extend_24,
};
pub use layout::{
    BlockLayout, FrontLayout, MAX_FRONT_BUFFER_BITS, MAX_REPEAT_BUFFER_BITS, MAX_STRIP_INDICES,
    align8, ceil_log2,
};
pub use palette::{DecodedOmm, MAX_OMM_HOT_PATCH_BYTES};
pub use topology::TriControl;

#[cfg(feature = "std")]
pub use decode::{
    DecodeLimits, DecodedMesh, decode_blocks, decode_blocks_with_mode, decode_mesh, decode_stream,
};
#[cfg(feature = "std")]
pub use error::RemapError;
#[cfg(feature = "std")]
pub use remap::{
    BlockOmmMap, BlockRange, ReconstructedMesh, TriangleRemap, reconstruct_with_remap,
    remap_corner_attributes, remap_face_attributes, remap_flat_corner_attributes,
    remap_vertex_attributes,
};

#[cfg(feature = "encode")]
pub use encode::{
    ClusterConfig, DegeneratePolicy, EncodeOptions, EncodeStatistics, EncodedMesh,
    FitFailureCounts, MeshView, PackingPolicy, PlannedBlock, PreparedBlock, PreparedOmm,
    PreparedTopology, QuantizationMode, QuantizationReport, QuantizedMesh, TriangleMetadata,
    encode_block, encode_mesh, encode_mesh_with_user_data, encode_preclustered_mesh, plan_block,
    quantize_mesh,
};
#[cfg(feature = "encode")]
pub use error::EncodeError;

/// Canonical encoder policy identifier (independent of DGF1 wire format compatibility).
pub const CANONICAL_ENCODER_POLICY: &str = "dgf-rs-v1";
