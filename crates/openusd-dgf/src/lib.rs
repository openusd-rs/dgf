//! Pure-Rust OpenUSD ↔ Dense Geometry Format (DGF1) adapter.
//!
//! Bridges resolved `UsdGeomMesh` prims (from `openusd` + `openusd-schemas`) to the
//! engine-agnostic [`dgf`] codec while preserving:
//! - Independent topology validation (rejecting negative `faceVertexCounts` and non-triangle faces).
//! - Explicit `subdivisionScheme = none` checking.
//! - Source USD face/corner provenance composed with DGF1 strip/block remaps.
//! - Sideband normals, face-varying UV seams (`primvars:st`), and arbitrary supported primvars.
//! - Face `UsdGeomSubset` material bindings mapped cleanly to DGF1 `geometry_id`s.
//! - Stage metrics (`upAxis`, `metersPerUnit`), mesh `orientation`, `doubleSided`, and world transforms.
//! - Content-addressed geometry caching ([`UsdDgfCache`]) invariant to transform-only edits.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bake;
pub mod cache;
pub mod error;
pub mod extract;

pub use bake::{
    BakedTriangleProvenance, BakedUsdMesh, RemappedAttributeValues, RemappedUsdAttribute,
    UsdBakeOptions, UsdMeshInstance, bake_extracted_mesh,
};
pub use cache::{CacheStatus, ContentHash, UsdDgfCache, compute_content_hash};
pub use dgf;
pub use error::UsdDgfError;
pub use extract::{
    ExtractedAttribute, ExtractedUsdMesh, UsdAttributeValues, UsdExtractOptions, UsdMaterialSubset,
    UsdSceneMetadata, extract_usd_mesh,
};
