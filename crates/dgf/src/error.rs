//! Structured error types for DGF1 validation, decoding, encoding, and remapping.

use core::fmt;

/// Named padded section within a DGF1 block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PaddingSection {
    /// Trailing pad bits at the end of the vertex offset table.
    VertexOffsets,
    /// Trailing pad bits at the end of the OMM descriptor selector array.
    OmmSelectors,
    /// Trailing pad bits at the end of the geometry ID palette section.
    GeometryPalette,
}

impl fmt::Display for PaddingSection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexOffsets => f.write_str("vertex offsets"),
            Self::OmmSelectors => f.write_str("OMM selectors"),
            Self::GeometryPalette => f.write_str("geometry palette"),
        }
    }
}

/// Errors produced while validating or decoding a DGF1 block or stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// Header magic byte did not equal `0x06`.
    InvalidMagic {
        /// Byte found at offset 0.
        found: u8,
    },
    /// Reserved bits (word 4 bits 30..=31) were nonzero.
    ReservedBitsNonZero {
        /// Two-bit reserved value found.
        bits: u8,
    },
    /// Biased exponent was outside `1..=232`.
    InvalidExponent {
        /// Raw 8-bit biased exponent found.
        raw: u8,
    },
    /// Sum of vertex offset widths `x_bits + y_bits + z_bits` was not a multiple of 4.
    InvalidVertexWidthSum {
        /// X offset width (`1..=16`).
        x_bits: u8,
        /// Y offset width (`1..=16`).
        y_bits: u8,
        /// Z offset width (`1..=16`).
        z_bits: u8,
        /// Sum `x_bits + y_bits + z_bits`.
        sum: u8,
    },
    /// `prim_id_base + (num_triangles - 1)` exceeded the 29-bit primitive ID range.
    PrimitiveIdOverflow {
        /// 29-bit primitive ID base stored in header.
        prim_id_base: u32,
        /// Number of triangles in the block (`1..=64`).
        num_triangles: u8,
    },
    /// Byte stream length was not a multiple of 128 bytes.
    TruncatedStream {
        /// Total byte length supplied.
        byte_len: usize,
    },
    /// Attempted bit access exceeded the buffer or bit-width bounds.
    BitOutOfBounds {
        /// Starting bit offset.
        start: usize,
        /// Requested bit width.
        width: usize,
    },
    /// Mandated zero padding bits in a named section contained a nonzero bit.
    PaddingNonZero {
        /// Section containing the nonzero padding.
        section: PaddingSection,
        /// Bit offset where the section padding starts.
        bit_offset: usize,
    },
    /// Front buffer (`vertex_bits + omm_bits + geom_bits`) exceeded 768 bits (96 bytes).
    FrontBufferOverflow {
        /// Computed front buffer size in bits.
        front_bits: usize,
        /// Maximum allowed front buffer size in bits (768).
        max_bits: usize,
    },
    /// Repeated-index buffer exceeded 192 bits (24 bytes).
    RepeatBufferOverflow {
        /// Computed repeated-index size in bits.
        repeat_bits: usize,
        /// Maximum allowed repeated-index size in bits (192).
        max_bits: usize,
    },
    /// Forward repeated-index stream overlapped the backward topology tail.
    SectionOverlap {
        /// End bit of forward repeated-index stream.
        repeat_end: usize,
        /// Start bit of backward first-use/control tail.
        tail_start: usize,
    },
    /// Checked arithmetic failed during layout calculation.
    ArithmeticOverflow {
        /// Context describing the failing calculation.
        context: &'static str,
    },
    /// `BACKTRACK` appeared where no prior `EDGE1`/`EDGE2` triangle existed.
    IllegalBacktrack {
        /// Local triangle index (`1..64`).
        triangle: u8,
        /// Previous triangle's raw control value (`0..=3`), if any.
        prev_control: Option<u8>,
    },
    /// Strip index referenced a vertex ID `>= num_vertices`.
    VertexIndexOutOfRange {
        /// Position in the strip index stream (`0..192`).
        strip_position: u8,
        /// Referenced local vertex index.
        index: u8,
        /// Header vertex count (`1..=64`).
        num_vertices: u8,
    },
    /// Repeated index referenced a vertex not yet introduced by a prior first-use flag.
    RepeatedIndexNotYetIntroduced {
        /// Position in the strip index stream (`3..192`).
        strip_position: u8,
        /// Referenced local vertex index.
        index: u8,
        /// Next vertex index to be introduced.
        next_new_vertex: u8,
    },
    /// Decoded floating-point vertex coordinate was non-finite (e.g., exponent 232 overflow).
    NonFinitePosition {
        /// Local vertex index (`0..64`).
        vertex: u8,
        /// Coordinate axis (`0` = X, `1` = Y, `2` = Z).
        axis: u8,
        /// Integer quantized value `anchor + offset`.
        quantized: i32,
        /// Biased exponent (`1..=232`).
        exponent: u8,
    },
    /// Geometry palette prefix length exceeded 25 bits.
    InvalidGeomPrefixLength {
        /// Prefix length found in `geom_id_meta`.
        prefix_bits: u8,
    },
    /// Geometry palette selector was `>= entry_count`.
    GeomPaletteIndexOutOfRange {
        /// Local triangle index.
        triangle: u8,
        /// Selector value decoded from stream.
        selector: u8,
        /// Number of palette entries (`1..=32`).
        entry_count: u8,
    },
    /// OMM descriptor selector was `>= omm_descriptor_count`.
    OmmSelectorOutOfRange {
        /// Local triangle index.
        triangle: u8,
        /// Selector value decoded from stream.
        selector: u8,
        /// Number of OMM descriptors (`1..=7`).
        descriptor_count: u8,
    },
    /// Fresh-asset validation found a nonzero byte in the OMM hot-patch region.
    OmmHotPatchNonZero {
        /// Byte offset within the hot-patch region.
        byte_offset: usize,
        /// Nonzero byte value found.
        value: u8,
    },
    /// Decoded block count exceeded the caller's configured limit.
    BlockLimitExceeded {
        /// Block count required.
        count: usize,
        /// Caller-specified maximum.
        limit: usize,
    },
    /// Cumulative decoded vertex count exceeded the caller's configured limit.
    VertexLimitExceeded {
        /// Vertex count required.
        count: usize,
        /// Caller-specified maximum.
        limit: usize,
    },
    /// Cumulative decoded triangle count exceeded the caller's configured limit.
    TriangleLimitExceeded {
        /// Triangle count required.
        count: usize,
        /// Caller-specified maximum.
        limit: usize,
    },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic { found } => {
                write!(
                    f,
                    "invalid DGF1 magic byte: expected 0x06, found 0x{found:02x}"
                )
            }
            Self::ReservedBitsNonZero { bits } => {
                write!(f, "header reserved bits must be zero, found 0b{bits:02b}")
            }
            Self::InvalidExponent { raw } => {
                write!(
                    f,
                    "biased exponent {raw} is outside valid DGF1 range 1..=232"
                )
            }
            Self::InvalidVertexWidthSum {
                x_bits,
                y_bits,
                z_bits,
                sum,
            } => write!(
                f,
                "vertex offset width sum {x_bits}+{y_bits}+{z_bits}={sum} is not a multiple of 4"
            ),
            Self::PrimitiveIdOverflow {
                prim_id_base,
                num_triangles,
            } => write!(
                f,
                "primitive ID range [{prim_id_base}, {}] exceeds 29-bit limit",
                u64::from(*prim_id_base) + u64::from(*num_triangles).saturating_sub(1)
            ),
            Self::TruncatedStream { byte_len } => write!(
                f,
                "DGF1 stream byte length {byte_len} is not a multiple of 128 bytes"
            ),
            Self::BitOutOfBounds { start, width } => {
                write!(f, "bit range [{start}, + {width}) is out of bounds")
            }
            Self::PaddingNonZero {
                section,
                bit_offset,
            } => write!(
                f,
                "nonzero padding bits in {section} section at bit offset {bit_offset}"
            ),
            Self::FrontBufferOverflow {
                front_bits,
                max_bits,
            } => write!(
                f,
                "front buffer size {front_bits} bits exceeds limit of {max_bits} bits"
            ),
            Self::RepeatBufferOverflow {
                repeat_bits,
                max_bits,
            } => write!(
                f,
                "repeated-index buffer size {repeat_bits} bits exceeds limit of {max_bits} bits"
            ),
            Self::SectionOverlap {
                repeat_end,
                tail_start,
            } => write!(
                f,
                "forward stream end ({repeat_end} bits) overlaps backward tail start ({tail_start} bits)"
            ),
            Self::ArithmeticOverflow { context } => {
                write!(f, "checked arithmetic overflow during {context}")
            }
            Self::IllegalBacktrack {
                triangle,
                prev_control,
            } => write!(
                f,
                "illegal BACKTRACK control at triangle {triangle} (previous control: {prev_control:?})"
            ),
            Self::VertexIndexOutOfRange {
                strip_position,
                index,
                num_vertices,
            } => write!(
                f,
                "strip position {strip_position} references vertex {index}, but block has only {num_vertices} vertices"
            ),
            Self::RepeatedIndexNotYetIntroduced {
                strip_position,
                index,
                next_new_vertex,
            } => write!(
                f,
                "repeated index {index} at strip position {strip_position} has not yet been introduced (next new vertex is {next_new_vertex})"
            ),
            Self::NonFinitePosition {
                vertex,
                axis,
                quantized,
                exponent,
            } => write!(
                f,
                "vertex {vertex} axis {axis} decoded to non-finite f32 (quantized={quantized}, biased_exponent={exponent})"
            ),
            Self::InvalidGeomPrefixLength { prefix_bits } => write!(
                f,
                "geometry palette prefix length {prefix_bits} exceeds maximum of 25 bits"
            ),
            Self::GeomPaletteIndexOutOfRange {
                triangle,
                selector,
                entry_count,
            } => write!(
                f,
                "geometry palette selector {selector} at triangle {triangle} is out of range for {entry_count} entries"
            ),
            Self::OmmSelectorOutOfRange {
                triangle,
                selector,
                descriptor_count,
            } => write!(
                f,
                "OMM descriptor selector {selector} at triangle {triangle} is out of range for {descriptor_count} descriptors"
            ),
            Self::OmmHotPatchNonZero { byte_offset, value } => write!(
                f,
                "fresh asset validation failed: OMM hot-patch byte at offset {byte_offset} is 0x{value:02x}, expected 0x00"
            ),
            Self::BlockLimitExceeded { count, limit } => {
                write!(
                    f,
                    "decoded block count {count} exceeds configured limit {limit}"
                )
            }
            Self::VertexLimitExceeded { count, limit } => {
                write!(
                    f,
                    "decoded vertex count {count} exceeds configured limit {limit}"
                )
            }
            Self::TriangleLimitExceeded { count, limit } => {
                write!(
                    f,
                    "decoded triangle count {count} exceeds configured limit {limit}"
                )
            }
        }
    }
}

impl core::error::Error for DecodeError {}

/// Error returned when decoding a multi-block stream, optionally identifying the failing block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamDecodeError {
    /// Zero-based block index within the stream, if the error occurred within a specific block.
    pub block_index: Option<usize>,
    /// Underlying decode error.
    pub kind: DecodeError,
}

impl StreamDecodeError {
    /// Creates a stream-level error not tied to a single block.
    pub const fn stream(kind: DecodeError) -> Self {
        Self {
            block_index: None,
            kind,
        }
    }

    /// Creates a block-level error at `block_index`.
    pub const fn at_block(block_index: usize, kind: DecodeError) -> Self {
        Self {
            block_index: Some(block_index),
            kind,
        }
    }
}

impl fmt::Display for StreamDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.block_index {
            Some(idx) => write!(f, "block {idx}: {}", self.kind),
            None => write!(f, "{}", self.kind),
        }
    }
}

impl core::error::Error for StreamDecodeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Errors produced by low-level checked bit writing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitWriteError {
    /// Target bit range exceeds buffer bounds or 64-bit value width.
    OutOfBounds {
        /// Starting bit offset.
        start: usize,
        /// Requested bit width.
        width: usize,
    },
    /// Supplied value does not fit into `width` unsigned bits without truncation.
    ValueOverflow {
        /// Value supplied by caller.
        value: u64,
        /// Target bit width.
        width: usize,
    },
}

impl fmt::Display for BitWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { start, width } => {
                write!(f, "bit write [{start}, + {width}) is out of bounds")
            }
            Self::ValueOverflow { value, width } => {
                write!(f, "value 0x{value:x} does not fit in {width} bits")
            }
        }
    }
}

impl core::error::Error for BitWriteError {}

/// Structured reasons why a valid candidate set of triangles does not fit in a single DGF1 block.
///
/// Distinct from invalid input errors: a `FitFailure` signals to the mesh baker that a
/// multi-triangle candidate should be split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FitFailure {
    /// Candidate exceeded 64 triangles.
    TooManyTriangles {
        /// Candidate triangle count.
        count: usize,
        /// Block maximum (64).
        max: usize,
    },
    /// Candidate exceeded 64 unique block vertices.
    TooManyVertices {
        /// Candidate vertex count.
        count: usize,
        /// Block maximum (64).
        max: usize,
    },
    /// Candidate exceeded 32 distinct `(geometry_id, opaque)` pairs.
    TooManyGeomPaletteEntries {
        /// Candidate distinct pair count.
        count: usize,
        /// Block maximum (32).
        max: usize,
    },
    /// Candidate exceeded 7 distinct OMM descriptors.
    TooManyOmmDescriptors {
        /// Candidate distinct OMM descriptor count.
        count: usize,
        /// Block maximum (7).
        max: usize,
    },
    /// Quantized coordinate span on an axis exceeded 16-bit unsigned offset range (`65535`).
    PositionSpanTooLarge {
        /// Coordinate axis (`0` = X, `1` = Y, `2` = Z).
        axis: u8,
        /// Required unsigned span `max_q - min_q`.
        span: u64,
        /// Maximum representable unsigned offset (`65535`).
        max: u64,
    },
    /// Quantized anchor did not fit signed 24-bit integer bounds.
    AnchorOutOfRange {
        /// Coordinate axis (`0` = X, `1` = Y, `2` = Z).
        axis: u8,
        /// Anchor value.
        anchor: i32,
    },
    /// Front buffer (`vertex + OMM + geometry palette`) exceeded 768 bits.
    FrontBufferOverflow {
        /// Required front buffer bits.
        bits: usize,
        /// Maximum allowed bits (768).
        max_bits: usize,
    },
    /// Repeated-index buffer exceeded 192 bits.
    RepeatBufferOverflow {
        /// Required repeated-index bits.
        bits: usize,
        /// Maximum allowed bits (192).
        max_bits: usize,
    },
    /// Forward repeated-index end overlapped the backward topology tail.
    BlockCapacityOverflow {
        /// End of repeated-index buffer in bits.
        repeat_end: usize,
        /// Start of backward topology tail in bits.
        tail_start: usize,
    },
}

impl fmt::Display for FitFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyTriangles { count, max } => {
                write!(
                    f,
                    "candidate has {count} triangles, maximum per block is {max}"
                )
            }
            Self::TooManyVertices { count, max } => {
                write!(
                    f,
                    "candidate has {count} vertices, maximum per block is {max}"
                )
            }
            Self::TooManyGeomPaletteEntries { count, max } => write!(
                f,
                "candidate has {count} distinct (geometry_id, opaque) pairs, maximum per block is {max}"
            ),
            Self::TooManyOmmDescriptors { count, max } => write!(
                f,
                "candidate has {count} distinct OMM descriptors, maximum per block is {max}"
            ),
            Self::PositionSpanTooLarge { axis, span, max } => write!(
                f,
                "quantized position span on axis {axis} is {span}, exceeding 16-bit offset limit {max}"
            ),
            Self::AnchorOutOfRange { axis, anchor } => write!(
                f,
                "position anchor {anchor} on axis {axis} is outside signed 24-bit range"
            ),
            Self::FrontBufferOverflow { bits, max_bits } => write!(
                f,
                "front buffer requires {bits} bits, exceeding limit {max_bits} bits"
            ),
            Self::RepeatBufferOverflow { bits, max_bits } => write!(
                f,
                "repeated-index buffer requires {bits} bits, exceeding limit {max_bits} bits"
            ),
            Self::BlockCapacityOverflow {
                repeat_end,
                tail_start,
            } => write!(
                f,
                "block bit budget exceeded: forward end {repeat_end} > backward tail start {tail_start}"
            ),
        }
    }
}

impl core::error::Error for FitFailure {}

/// Errors produced during DGF1 block or mesh encoding.
#[cfg(feature = "encode")]
#[derive(Debug, Clone, PartialEq)]
pub enum EncodeError {
    /// Input vertex contained NaN or infinity.
    NonFiniteCoordinate {
        /// Source vertex index.
        vertex: usize,
        /// Coordinate axis (`0` = X, `1` = Y, `2` = Z).
        axis: u8,
        /// Non-finite value encountered.
        value: f32,
    },
    /// Triangle referenced a vertex index `>= positions.len()`.
    VertexIndexOutOfBounds {
        /// Source triangle index.
        triangle: usize,
        /// Corner (`0`, `1`, or `2`).
        corner: u8,
        /// Out-of-bounds vertex index.
        index: u32,
        /// Total number of input positions.
        num_vertices: usize,
    },
    /// `triangle_metadata` slice length did not match `triangles.len()`.
    MetadataLengthMismatch {
        /// Expected triangle count.
        expected: usize,
        /// Provided metadata slice length.
        found: usize,
    },
    /// `triangle_omm_ids` slice length did not match `triangles.len()`.
    OmmLengthMismatch {
        /// Expected triangle count.
        expected: usize,
        /// Provided OMM slice length.
        found: usize,
    },
    /// Input triangle had repeated vertex indices while degenerate policy was `Reject`.
    DegenerateTriangle {
        /// Source triangle index.
        triangle: usize,
        /// Repeated-index corner tuple.
        indices: [u32; 3],
    },
    /// Input vertex count exceeded `u32::MAX` (required by remap representation).
    TooManyInputVertices {
        /// Number of vertices supplied.
        count: usize,
    },
    /// Input triangle count exceeded `u32::MAX` (required by remap representation).
    TooManyInputTriangles {
        /// Number of triangles supplied.
        count: usize,
    },
    /// Preclustered input referenced an invalid source triangle index.
    InvalidClusterTriangle {
        /// Cluster index.
        cluster: usize,
        /// Invalid triangle index.
        triangle: u32,
        /// Total input triangles.
        num_triangles: usize,
    },
    /// Preclustered input listed the same source triangle more than once.
    DuplicateClusterTriangle {
        /// Duplicated source triangle index.
        triangle: u32,
    },
    /// Preclustered input omitted one or more retained source triangles.
    MissingClusterTriangles {
        /// Number of omitted triangles.
        missing: usize,
    },
    /// Unbiased quantization exponent was outside `-126..=105`.
    InvalidExponent {
        /// Supplied unbiased exponent.
        exponent: i16,
    },
    /// Coordinate error tolerance was negative, NaN, or infinite.
    InvalidTolerance {
        /// Supplied tolerance value.
        tolerance: f32,
    },
    /// Target bit-width quantization parameter was outside `1..=24`.
    InvalidTargetBitWidth {
        /// Supplied bit width.
        bits: u8,
    },
    /// Quantized coordinate fell outside the signed 24-bit integer range.
    QuantizedRangeExceeded {
        /// Source vertex index.
        vertex: usize,
        /// Coordinate axis (`0` = X, `1` = Y, `2` = Z).
        axis: u8,
        /// Source coordinate.
        value: f32,
        /// Unbiased exponent attempted.
        exponent: i16,
    },
    /// Reconstructed quantized coordinate was non-finite.
    QuantizedNonFinite {
        /// Source vertex index.
        vertex: usize,
        /// Coordinate axis.
        axis: u8,
        /// Unbiased exponent attempted.
        exponent: i16,
    },
    /// Requested coordinate error tolerance cannot be represented in DGF1 signed 24-bit coordinates.
    ToleranceUnachievable {
        /// Requested maximum coordinate error.
        requested: f32,
        /// Human-readable diagnostic (e.g., suggesting local object coordinates).
        reason: &'static str,
    },
    /// Strict collapse rejection was enabled and quantization collapsed one or more triangles.
    QuantizationCollapsedTriangles {
        /// Number of triangles whose quantized vertices became coincident.
        count: usize,
    },
    /// Attempted to encode a prepared block with zero triangles or fewer than 3 vertices.
    EmptyBlock,
    /// Geometry ID exceeded the 24-bit DGF1 limit (`0..=0xFF_FFFF`).
    InvalidGeometryId {
        /// Triangle index.
        triangle: usize,
        /// Out-of-range geometry ID.
        geometry_id: u32,
    },
    /// Primitive ID base + triangle count exceeded the 29-bit DGF1 limit.
    PrimitiveIdRangeExceeded {
        /// Starting primitive ID base.
        base: u32,
        /// Number of triangles requiring primitive IDs.
        total_triangles: usize,
    },
    /// Prepared topology violated DGF1 strip/control invariants.
    InvalidTopology(&'static str),
    /// Candidate block did not fit within DGF1 block budgets.
    DoesNotFit(FitFailure),
    /// Even a single triangle could not fit into a DGF1 block at the selected quantization.
    SingleTriangleDoesNotFit {
        /// Original source triangle index.
        input_triangle: u32,
        /// Specific fit failure preventing encoding.
        reason: FitFailure,
    },
    /// Output block count exceeded the caller's configured `max_blocks` limit.
    MaxBlocksExceeded {
        /// Output blocks produced.
        count: usize,
        /// Configured limit.
        limit: usize,
    },
    /// Internal validation or serialization invariant failed.
    InternalValidation(DecodeError),
}

#[cfg(feature = "encode")]
impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteCoordinate {
                vertex,
                axis,
                value,
            } => write!(
                f,
                "non-finite coordinate {value} at vertex {vertex} axis {axis}"
            ),
            Self::VertexIndexOutOfBounds {
                triangle,
                corner,
                index,
                num_vertices,
            } => write!(
                f,
                "triangle {triangle} corner {corner} references vertex {index}, out of bounds for {num_vertices} vertices"
            ),
            Self::MetadataLengthMismatch { expected, found } => write!(
                f,
                "triangle metadata length {found} does not match triangle count {expected}"
            ),
            Self::OmmLengthMismatch { expected, found } => write!(
                f,
                "triangle OMM slice length {found} does not match triangle count {expected}"
            ),
            Self::DegenerateTriangle { triangle, indices } => write!(
                f,
                "degenerate triangle {triangle} has repeated vertex indices {indices:?}"
            ),
            Self::TooManyInputVertices { count } => {
                write!(f, "input vertex count {count} exceeds u32::MAX")
            }
            Self::TooManyInputTriangles { count } => {
                write!(f, "input triangle count {count} exceeds u32::MAX")
            }
            Self::InvalidClusterTriangle {
                cluster,
                triangle,
                num_triangles,
            } => write!(
                f,
                "cluster {cluster} references triangle {triangle}, out of bounds for {num_triangles} triangles"
            ),
            Self::DuplicateClusterTriangle { triangle } => {
                write!(
                    f,
                    "preclustered input lists triangle {triangle} more than once"
                )
            }
            Self::MissingClusterTriangles { missing } => {
                write!(
                    f,
                    "preclustered input omitted {missing} retained source triangles"
                )
            }
            Self::InvalidExponent { exponent } => write!(
                f,
                "unbiased quantization exponent {exponent} is outside supported DGF1 range -126..=105"
            ),
            Self::InvalidTolerance { tolerance } => {
                write!(f, "invalid coordinate error tolerance {tolerance}")
            }
            Self::InvalidTargetBitWidth { bits } => {
                write!(
                    f,
                    "target bit width {bits} is outside supported range 1..=24"
                )
            }
            Self::QuantizedRangeExceeded {
                vertex,
                axis,
                value,
                exponent,
            } => write!(
                f,
                "vertex {vertex} axis {axis} ({value}) exceeds signed 24-bit range at exponent {exponent}; consider local object coordinates or a coarser exponent"
            ),
            Self::QuantizedNonFinite {
                vertex,
                axis,
                exponent,
            } => write!(
                f,
                "vertex {vertex} axis {axis} reconstructed to non-finite f32 at exponent {exponent}"
            ),
            Self::ToleranceUnachievable { requested, reason } => write!(
                f,
                "requested coordinate error tolerance {requested} cannot be satisfied: {reason}"
            ),
            Self::QuantizationCollapsedTriangles { count } => write!(
                f,
                "quantization collapsed {count} triangle(s) to coincident vertices under strict collapse policy"
            ),
            Self::EmptyBlock => {
                f.write_str("prepared block must contain at least 1 triangle and 3 vertices")
            }
            Self::InvalidGeometryId {
                triangle,
                geometry_id,
            } => write!(
                f,
                "triangle {triangle} geometry ID {geometry_id} exceeds 24-bit DGF1 limit"
            ),
            Self::PrimitiveIdRangeExceeded {
                base,
                total_triangles,
            } => write!(
                f,
                "primitive ID base {base} with {total_triangles} triangles exceeds 29-bit DGF1 range"
            ),
            Self::InvalidTopology(msg) => write!(f, "invalid prepared topology: {msg}"),
            Self::DoesNotFit(reason) => write!(f, "block does not fit: {reason}"),
            Self::SingleTriangleDoesNotFit {
                input_triangle,
                reason,
            } => write!(
                f,
                "single source triangle {input_triangle} cannot fit in a DGF1 block: {reason}"
            ),
            Self::MaxBlocksExceeded { count, limit } => {
                write!(
                    f,
                    "encoded block count {count} exceeds configured limit {limit}"
                )
            }
            Self::InternalValidation(err) => {
                write!(f, "internal block validation error: {err}")
            }
        }
    }
}

#[cfg(feature = "encode")]
impl core::error::Error for EncodeError {}

/// Errors produced when remapping attributes or reconstructing a globally indexed mesh.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq)]
pub enum RemapError {
    /// Decoded vertex count did not match `vertex_remap.len()`.
    VertexCountMismatch {
        /// Number of decoded vertices.
        decoded: usize,
        /// Number of vertex remap entries.
        remap: usize,
    },
    /// Decoded triangle count did not match `triangle_remap.len()`.
    TriangleCountMismatch {
        /// Number of decoded triangles.
        decoded: usize,
        /// Number of triangle remap entries.
        remap: usize,
    },
    /// A vertex remap entry exceeded the target vertex count or source attribute slice length.
    SourceVertexOutOfBounds {
        /// Decoded vertex index.
        decoded_vertex: usize,
        /// Source vertex index from remap.
        source_vertex: u32,
        /// Source slice length.
        source_len: usize,
    },
    /// A triangle remap entry exceeded the source attribute slice length.
    SourceTriangleOutOfBounds {
        /// Decoded triangle index.
        decoded_triangle: usize,
        /// Source triangle index from remap.
        source_triangle: u32,
        /// Source slice length.
        source_len: usize,
    },
    /// A corner remap entry contained a corner index `>= 3` or non-bijective corners.
    InvalidCornerMap {
        /// Decoded triangle index.
        decoded_triangle: usize,
        /// Invalid corner mapping array.
        corners: [u8; 3],
    },
    /// Multiple decoded block copies mapped to the same source vertex disagreed in position.
    InconsistentVertexPosition {
        /// Shared source vertex index.
        source_vertex: u32,
        /// Position decoded from an earlier block.
        first: [f32; 3],
        /// Conflicting position decoded from a later block.
        conflicting: [f32; 3],
    },
    /// Decoded triangle referenced an invalid decoded vertex index.
    DecodedIndexOutOfBounds {
        /// Decoded triangle index.
        triangle: usize,
        /// Decoded vertex index.
        vertex: u32,
        /// Total decoded vertices.
        num_vertices: usize,
    },
}

#[cfg(feature = "std")]
impl fmt::Display for RemapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexCountMismatch { decoded, remap } => write!(
                f,
                "decoded vertex count {decoded} does not match vertex remap length {remap}"
            ),
            Self::TriangleCountMismatch { decoded, remap } => write!(
                f,
                "decoded triangle count {decoded} does not match triangle remap length {remap}"
            ),
            Self::SourceVertexOutOfBounds {
                decoded_vertex,
                source_vertex,
                source_len,
            } => write!(
                f,
                "vertex remap[{decoded_vertex}] -> source vertex {source_vertex} is out of bounds (len {source_len})"
            ),
            Self::SourceTriangleOutOfBounds {
                decoded_triangle,
                source_triangle,
                source_len,
            } => write!(
                f,
                "triangle remap[{decoded_triangle}] -> source triangle {source_triangle} is out of bounds (len {source_len})"
            ),
            Self::InvalidCornerMap {
                decoded_triangle,
                corners,
            } => write!(
                f,
                "invalid corner mapping {corners:?} at decoded triangle {decoded_triangle}"
            ),
            Self::InconsistentVertexPosition {
                source_vertex,
                first,
                conflicting,
            } => write!(
                f,
                "inconsistent decoded positions for source vertex {source_vertex}: {first:?} vs {conflicting:?}"
            ),
            Self::DecodedIndexOutOfBounds {
                triangle,
                vertex,
                num_vertices,
            } => write!(
                f,
                "decoded triangle {triangle} references vertex {vertex} (num_vertices={num_vertices})"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl core::error::Error for RemapError {}
