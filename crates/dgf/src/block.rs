//! 128-byte DGF1 block container and validated immutable borrow.

use crate::bits::{BLOCK_BYTES, bits_are_zero, read_bits};
use crate::error::{DecodeError, PaddingSection};
use crate::header::{Header, MAX_TRIANGLES};
use crate::layout::{BlockLayout, FrontLayout, MAX_STRIP_INDICES};
use crate::palette::{decode_geom_ids, decode_omm};
use crate::topology::{TriControl, decode_controls, decode_strip_indices, scan_first_use_flags};

/// Validation policy distinguishing fresh encoded assets from runtime-patched blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// Allows nonzero OMM hot-patch bytes written by a runtime acceleration-structure builder.
    #[default]
    Runtime,
    /// Requires freshly encoded OMM hot-patch bytes to be zero.
    Asset,
}

/// Raw 128-byte DGF1 block byte container.
///
/// Holding a `Block` does **not** guarantee validity; call [`validate_block`] to validate
/// and obtain a [`ValidatedBlock`]. Does not impose 128-byte memory alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Block(pub [u8; BLOCK_BYTES]);

impl Block {
    /// Creates a `Block` from a 128-byte array.
    pub const fn from_bytes(bytes: [u8; BLOCK_BYTES]) -> Self {
        Self(bytes)
    }

    /// Creates a zero-filled 128-byte buffer.
    pub const fn zeroed() -> Self {
        Self([0u8; BLOCK_BYTES])
    }

    /// Returns an immutable reference to the underlying 128 bytes.
    pub const fn as_bytes(&self) -> &[u8; BLOCK_BYTES] {
        &self.0
    }

    /// Returns a mutable reference to the underlying 128 bytes.
    pub const fn as_bytes_mut(&mut self) -> &mut [u8; BLOCK_BYTES] {
        &mut self.0
    }

    /// Consumes the container and returns the raw 128-byte array.
    pub const fn into_bytes(self) -> [u8; BLOCK_BYTES] {
        self.0
    }
}

impl From<[u8; BLOCK_BYTES]> for Block {
    fn from(bytes: [u8; BLOCK_BYTES]) -> Self {
        Self(bytes)
    }
}

impl AsRef<[u8]> for Block {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

/// Proof of complete structural and semantic validity for a borrowed 128-byte DGF1 block.
///
/// Fields are private so callers cannot construct or mutate a `ValidatedBlock` without
/// passing through [`validate_block`] or [`validate_block_with_mode`].
#[derive(Debug, Clone, Copy)]
pub struct ValidatedBlock<'a> {
    bytes: &'a [u8; BLOCK_BYTES],
    header: Header,
    layout: BlockLayout,
    controls: [TriControl; MAX_TRIANGLES as usize],
    is_first: [bool; MAX_STRIP_INDICES],
    mode: ValidationMode,
}

impl<'a> ValidatedBlock<'a> {
    /// Returns the validated 128-byte block slice.
    pub const fn bytes(&self) -> &'a [u8; BLOCK_BYTES] {
        self.bytes
    }

    /// Returns the validated semantic header.
    pub const fn header(&self) -> &Header {
        &self.header
    }

    /// Returns the validated section layout.
    pub const fn layout(&self) -> &BlockLayout {
        &self.layout
    }

    /// Returns the [`ValidationMode`] used when validating this block.
    pub const fn validation_mode(&self) -> ValidationMode {
        self.mode
    }

    pub(crate) const fn controls(&self) -> &[TriControl; MAX_TRIANGLES as usize] {
        &self.controls
    }

    pub(crate) const fn is_first(&self) -> &[bool; MAX_STRIP_INDICES] {
        &self.is_first
    }
}

/// Validates a 128-byte DGF1 block under [`ValidationMode::Runtime`].
///
/// Checks header constraints, named section padding, front/repeat/tail budgets and non-overlap,
/// control legality, strip index bounds and first-use ordering, finite decoded vertex positions,
/// geometry palette selectors, and OMM selector bounds.
pub fn validate_block(bytes: &[u8; BLOCK_BYTES]) -> Result<ValidatedBlock<'_>, DecodeError> {
    validate_block_with_mode(bytes, ValidationMode::Runtime)
}

/// Validates a 128-byte DGF1 block using the specified [`ValidationMode`].
pub fn validate_block_with_mode(
    bytes: &[u8; BLOCK_BYTES],
    mode: ValidationMode,
) -> Result<ValidatedBlock<'_>, DecodeError> {
    let header = Header::parse(bytes)?;
    let front = FrontLayout::compute(&header)?;

    // Validate vertex offset trailing padding bits are zero.
    let vertex_pad_bits = front.vertex_bits - front.vertex_data_bits;
    let vertex_pad_start = front.vertex_start + front.vertex_data_bits;
    if vertex_pad_bits > 0 && !bits_are_zero(bytes, vertex_pad_start, vertex_pad_bits)? {
        return Err(DecodeError::PaddingNonZero {
            section: PaddingSection::VertexOffsets,
            bit_offset: vertex_pad_start,
        });
    }

    // Validate backward control stream and first-use flags, then full layout non-overlap.
    let (controls, strip_index_count) = decode_controls(bytes, header.num_triangles)?;
    let (is_first, _new_vertex_count, repeated_count) =
        scan_first_use_flags(bytes, header.num_triangles, strip_index_count)?;
    let layout = BlockLayout::compute(front, &header, strip_index_count, repeated_count)?;

    // Validate strip vertex indices (bounds and prior introduction).
    let _ = decode_strip_indices(bytes, &header, &layout, &is_first)?;

    // Validate decoded vertex positions remain finite (catches exponent 232 overflow).
    validate_vertex_positions(bytes, &header, &front)?;

    // Validate OMM and geometry palettes (selectors, padding, and Asset-mode hot-patch zeros).
    let _ = decode_omm(bytes, &header, &front, mode)?;
    let _ = decode_geom_ids(bytes, &header, &front)?;

    Ok(ValidatedBlock {
        bytes,
        header,
        layout,
        controls,
        is_first,
        mode,
    })
}

fn validate_vertex_positions(
    bytes: &[u8; BLOCK_BYTES],
    header: &Header,
    front: &FrontLayout,
) -> Result<(), DecodeError> {
    let scale = header.exponent.scale_f32();
    let bx = usize::from(header.offset_bits[0]);
    let by = usize::from(header.offset_bits[1]);
    let bz = usize::from(header.offset_bits[2]);
    let stride = bx + by + bz;

    for v in 0..usize::from(header.num_vertices) {
        let base = front.vertex_start + v * stride;
        let ox = read_bits(bytes, base, bx)? as i32;
        let oy = read_bits(bytes, base + bx, by)? as i32;
        let oz = read_bits(bytes, base + bx + by, bz)? as i32;
        let qs = [
            header.anchor[0] + ox,
            header.anchor[1] + oy,
            header.anchor[2] + oz,
        ];
        for (axis, &q) in qs.iter().enumerate() {
            let coord = (q as f32) * scale;
            if !coord.is_finite() {
                return Err(DecodeError::NonFinitePosition {
                    vertex: v as u8,
                    axis: axis as u8,
                    quantized: q,
                    exponent: header.exponent.raw(),
                });
            }
        }
    }
    Ok(())
}
