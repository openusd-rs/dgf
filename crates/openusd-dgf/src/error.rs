//! Structured errors for OpenUSD-to-DGF1 extraction, validation, baking, and caching.

use core::fmt;

/// Errors returned while extracting, validating, or baking OpenUSD meshes into DGF1.
#[derive(Debug)]
pub enum UsdDgfError {
    /// Error from the underlying `openusd` runtime.
    Usd(openusd::Error),
    /// Error from `openusd-schemas`.
    Schema(openusd_schemas::SchemaError),
    /// Error returned by the DGF1 encoder.
    DgfEncode(dgf::EncodeError),
    /// Error returned by the DGF1 decoder.
    DgfDecode(dgf::StreamDecodeError),
    /// Error returned while remapping attributes through DGF1 maps.
    DgfRemap(dgf::RemapError),
    /// Mesh uses a subdivision scheme other than `none` when `require_subdivision_none` is set.
    UnsupportedSubdivisionScheme {
        /// Prim path of the mesh.
        prim_path: String,
        /// Authored or fallback subdivision scheme token.
        scheme: String,
    },
    /// `faceVertexCounts` contained a negative count.
    NegativeFaceVertexCount {
        /// Prim path of the mesh.
        prim_path: String,
        /// Zero-based USD face index.
        face: usize,
        /// Negative count encountered.
        count: i32,
    },
    /// Mesh contained a non-triangle face (polygon with `count != 3`).
    NonTriangleFace {
        /// Prim path of the mesh.
        prim_path: String,
        /// Zero-based USD face index.
        face: usize,
        /// Vertex count of the face.
        vertex_count: i32,
    },
    /// Sum of `faceVertexCounts` did not match `faceVertexIndices.len()`.
    FaceVertexCountSumMismatch {
        /// Prim path of the mesh.
        prim_path: String,
        /// Checked sum of `faceVertexCounts`.
        counts_sum: usize,
        /// Length of `faceVertexIndices`.
        indices_len: usize,
    },
    /// A `faceVertexIndices` entry was negative or `>= points.len()`.
    FaceVertexIndexOutOfRange {
        /// Prim path of the mesh.
        prim_path: String,
        /// Zero-based USD face index.
        face: usize,
        /// Corner index within the face.
        corner: usize,
        /// Out-of-range point index.
        index: i32,
        /// Total number of points on the mesh.
        num_points: usize,
    },
    /// A `GeomSubset` referenced a face index outside `0..num_faces`.
    SubsetFaceOutOfRange {
        /// Prim path of the owning mesh.
        prim_path: String,
        /// Prim path of the subset.
        subset_path: String,
        /// Invalid face index.
        face: i32,
        /// Total faces on the mesh.
        num_faces: usize,
    },
    /// Authored attribute or primvar element count did not match its USD interpolation.
    AttributeLengthMismatch {
        /// Prim path of the owning mesh.
        prim_path: String,
        /// Attribute or primvar name.
        attribute: String,
        /// Interpolation token.
        interpolation: &'static str,
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements found.
        found: usize,
    },
}

impl fmt::Display for UsdDgfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usd(e) => write!(f, "OpenUSD error: {e}"),
            Self::Schema(e) => write!(f, "OpenUSD schema error: {e}"),
            Self::DgfEncode(e) => write!(f, "DGF1 encode error: {e}"),
            Self::DgfDecode(e) => write!(f, "DGF1 decode error: {e}"),
            Self::DgfRemap(e) => write!(f, "DGF1 remap error: {e}"),
            Self::UnsupportedSubdivisionScheme { prim_path, scheme } => write!(
                f,
                "mesh {prim_path} has subdivisionScheme={scheme:?}; only \"none\" is supported initially"
            ),
            Self::NegativeFaceVertexCount {
                prim_path,
                face,
                count,
            } => write!(
                f,
                "mesh {prim_path} has negative faceVertexCounts[{face}] = {count}"
            ),
            Self::NonTriangleFace {
                prim_path,
                face,
                vertex_count,
            } => write!(
                f,
                "mesh {prim_path} face {face} has {vertex_count} vertices; only triangle faces are supported"
            ),
            Self::FaceVertexCountSumMismatch {
                prim_path,
                counts_sum,
                indices_len,
            } => write!(
                f,
                "mesh {prim_path} faceVertexCounts sum ({counts_sum}) != faceVertexIndices length ({indices_len})"
            ),
            Self::FaceVertexIndexOutOfRange {
                prim_path,
                face,
                corner,
                index,
                num_points,
            } => write!(
                f,
                "mesh {prim_path} face {face} corner {corner} references point {index}, out of bounds for {num_points} points"
            ),
            Self::SubsetFaceOutOfRange {
                prim_path,
                subset_path,
                face,
                num_faces,
            } => write!(
                f,
                "mesh {prim_path} subset {subset_path} references face {face}, out of bounds for {num_faces} faces"
            ),
            Self::AttributeLengthMismatch {
                prim_path,
                attribute,
                interpolation,
                expected,
                found,
            } => write!(
                f,
                "mesh {prim_path} attribute {attribute} ({interpolation}) has {found} elements, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for UsdDgfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Usd(e) => Some(e),
            Self::Schema(e) => Some(e),
            Self::DgfEncode(e) => Some(e),
            Self::DgfDecode(e) => Some(e),
            Self::DgfRemap(e) => Some(e),
            _ => None,
        }
    }
}

impl From<openusd::Error> for UsdDgfError {
    fn from(err: openusd::Error) -> Self {
        Self::Usd(err)
    }
}

impl From<openusd_schemas::SchemaError> for UsdDgfError {
    fn from(err: openusd_schemas::SchemaError) -> Self {
        Self::Schema(err)
    }
}

impl From<dgf::EncodeError> for UsdDgfError {
    fn from(err: dgf::EncodeError) -> Self {
        Self::DgfEncode(err)
    }
}

impl From<dgf::StreamDecodeError> for UsdDgfError {
    fn from(err: dgf::StreamDecodeError) -> Self {
        Self::DgfDecode(err)
    }
}

impl From<dgf::RemapError> for UsdDgfError {
    fn from(err: dgf::RemapError) -> Self {
        Self::DgfRemap(err)
    }
}
