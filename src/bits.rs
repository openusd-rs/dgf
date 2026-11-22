//! Checked little-endian LSB-first bit reader and writer.
//!
//! All bit positions use explicit little-endian LSB-first numbering:
//! bit `k` is bit `(k % 8)` (0 = LSB, 7 = MSB) of `bytes[k / 8]`.
//! No operation ever reads beyond the slice bounds or shifts by the machine word width.

use crate::error::{BitWriteError, DecodeError};

/// Size of a DGF1 block in bytes.
pub const BLOCK_BYTES: usize = 128;

/// Size of a DGF1 block in bits.
pub const BLOCK_BITS: usize = BLOCK_BYTES * 8;

/// Reads `width` unsigned bits starting at `start` using bounded word loads.
///
/// - `width == 0` returns `Ok(0)` without indexing `bytes`, provided `start <= bytes.len() * 8`.
/// - `width > 64` or `start + width > bytes.len() * 8` returns [`DecodeError::BitOutOfBounds`].
pub fn read_bits(bytes: &[u8], start: usize, width: usize) -> Result<u64, DecodeError> {
    let total_bits = bytes
        .len()
        .checked_mul(8)
        .ok_or(DecodeError::ArithmeticOverflow {
            context: "bit buffer length",
        })?;
    let end = start
        .checked_add(width)
        .ok_or(DecodeError::BitOutOfBounds { start, width })?;
    if width > 64 || end > total_bits {
        return Err(DecodeError::BitOutOfBounds { start, width });
    }
    if width == 0 {
        return Ok(0);
    }

    let byte_idx = start / 8;
    let bit_off = start % 8;

    // Copy up to 8 available bytes starting at byte_idx into a local u64 buffer.
    let available = bytes.len() - byte_idx;
    let take = available.min(8);
    let mut buf = [0u8; 8];
    buf[..take].copy_from_slice(&bytes[byte_idx..byte_idx + take]);
    let mut val = u64::from_le_bytes(buf) >> bit_off;

    // When the bitfield spans into a 9th byte (bit_off + width > 64), pull the top bits from it.
    let low_bits = 64 - bit_off;
    if width > low_bits {
        let extra = u64::from(bytes[byte_idx + 8]);
        val |= extra << low_bits;
    }

    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    Ok(val & mask)
}

/// Scalar per-bit reference reader for differential testing against [`read_bits`].
pub fn read_bits_scalar(bytes: &[u8], start: usize, width: usize) -> Result<u64, DecodeError> {
    let total_bits = bytes
        .len()
        .checked_mul(8)
        .ok_or(DecodeError::ArithmeticOverflow {
            context: "bit buffer length",
        })?;
    let end = start
        .checked_add(width)
        .ok_or(DecodeError::BitOutOfBounds { start, width })?;
    if width > 64 || end > total_bits {
        return Err(DecodeError::BitOutOfBounds { start, width });
    }
    if width == 0 {
        return Ok(0);
    }

    let mut out = 0u64;
    for i in 0..width {
        let bit_pos = start + i;
        let bit = (bytes[bit_pos / 8] >> (bit_pos % 8)) & 1;
        out |= u64::from(bit) << i;
    }
    Ok(out)
}

/// Writes `width` unsigned bits of `value` starting at `start`.
///
/// Fails with [`BitWriteError::ValueOverflow`] if `value` does not fit in `width` bits
/// (no silent truncation). `width == 0` is a no-op only when `value == 0`.
pub fn write_bits(
    bytes: &mut [u8],
    start: usize,
    width: usize,
    value: u64,
) -> Result<(), BitWriteError> {
    let total_bits = bytes
        .len()
        .checked_mul(8)
        .ok_or(BitWriteError::OutOfBounds { start, width })?;
    let end = start
        .checked_add(width)
        .ok_or(BitWriteError::OutOfBounds { start, width })?;
    if width > 64 || end > total_bits {
        return Err(BitWriteError::OutOfBounds { start, width });
    }
    if width == 0 {
        if value != 0 {
            return Err(BitWriteError::ValueOverflow { value, width });
        }
        return Ok(());
    }
    if width < 64 && (value >> width) != 0 {
        return Err(BitWriteError::ValueOverflow { value, width });
    }

    write_bits_unchecked_width(bytes, start, width, value);
    Ok(())
}

/// Writes the low `width` bits of `value`, deliberately masking higher bits.
///
/// Separated from [`write_bits`] so public fields never silently truncate.
pub fn write_masked_bits(
    bytes: &mut [u8],
    start: usize,
    width: usize,
    value: u64,
) -> Result<(), BitWriteError> {
    let total_bits = bytes
        .len()
        .checked_mul(8)
        .ok_or(BitWriteError::OutOfBounds { start, width })?;
    let end = start
        .checked_add(width)
        .ok_or(BitWriteError::OutOfBounds { start, width })?;
    if width > 64 || end > total_bits {
        return Err(BitWriteError::OutOfBounds { start, width });
    }
    if width == 0 {
        return Ok(());
    }
    let masked = if width == 64 {
        value
    } else {
        value & ((1u64 << width) - 1)
    };
    write_bits_unchecked_width(bytes, start, width, masked);
    Ok(())
}

fn write_bits_unchecked_width(bytes: &mut [u8], start: usize, width: usize, value: u64) {
    let mut remaining = width;
    let mut bit_pos = start;
    let mut shift = 0usize;

    while remaining > 0 {
        let byte_idx = bit_pos / 8;
        let bit_off = bit_pos % 8;
        let take = remaining.min(8 - bit_off);
        let chunk_mask = if take == 8 {
            0xFFu8
        } else {
            ((1u8 << take) - 1) << bit_off
        };
        let chunk_bits = (((value >> shift) as u8) << bit_off) & chunk_mask;
        bytes[byte_idx] = (bytes[byte_idx] & !chunk_mask) | chunk_bits;

        bit_pos += take;
        shift += take;
        remaining -= take;
    }
}

/// Checks that every bit in `[start, start + width)` is zero.
pub fn bits_are_zero(bytes: &[u8], start: usize, width: usize) -> Result<bool, DecodeError> {
    let total_bits = bytes
        .len()
        .checked_mul(8)
        .ok_or(DecodeError::ArithmeticOverflow {
            context: "bit buffer length",
        })?;
    let end = start
        .checked_add(width)
        .ok_or(DecodeError::BitOutOfBounds { start, width })?;
    if end > total_bits {
        return Err(DecodeError::BitOutOfBounds { start, width });
    }

    let mut pos = start;
    let mut rem = width;
    while rem > 0 {
        let step = rem.min(64);
        if read_bits(bytes, pos, step)? != 0 {
            return Ok(false);
        }
        pos += step;
        rem -= step;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_small_buffer_matches_scalar_oracle() {
        let buf: [u8; 11] = [
            0xA5, 0x3C, 0xF0, 0x0F, 0x96, 0x5A, 0xC3, 0x12, 0x7E, 0x81, 0xD4,
        ];
        let total = buf.len() * 8;
        for start in 0..=total {
            for width in 0..=64 {
                let fast = read_bits(&buf, start, width);
                let slow = read_bits_scalar(&buf, start, width);
                assert_eq!(fast, slow, "mismatch at start={start} width={width}");
            }
        }
    }

    #[test]
    fn zero_width_reads_and_writes_at_boundaries() {
        let mut buf = [0xFFu8; 4];
        assert_eq!(read_bits(&buf, 0, 0), Ok(0));
        assert_eq!(read_bits(&buf, 32, 0), Ok(0));
        assert!(read_bits(&buf, 33, 0).is_err());

        assert_eq!(write_bits(&mut buf, 0, 0, 0), Ok(()));
        assert_eq!(write_bits(&mut buf, 32, 0, 0), Ok(()));
        assert_eq!(
            write_bits(&mut buf, 0, 0, 1),
            Err(BitWriteError::ValueOverflow { value: 1, width: 0 })
        );
        assert_eq!(write_masked_bits(&mut buf, 0, 0, 0xDEAD), Ok(()));
        assert_eq!(buf, [0xFFu8; 4]);
    }

    #[test]
    fn write_and_read_roundtrip_across_byte_and_word_boundaries() {
        let mut buf = [0u8; 16];
        // Test 48-bit vertex tuple crossing word boundaries and 64-bit write at unaligned offsets
        for start in 0..64 {
            for &width in &[1usize, 3, 4, 7, 8, 9, 15, 16, 24, 25, 32, 48, 63, 64] {
                if start + width > buf.len() * 8 {
                    continue;
                }
                buf.fill(0xAA);
                let mask = if width == 64 {
                    u64::MAX
                } else {
                    (1u64 << width) - 1
                };
                let val = 0x1234_5678_9ABC_DEF0u64 & mask;
                write_bits(&mut buf, start, width, val).unwrap();
                assert_eq!(read_bits(&buf, start, width), Ok(val));
                assert_eq!(read_bits_scalar(&buf, start, width), Ok(val));
                // Check neighbor bits untouched
                if start > 0 {
                    let expected_prev = ((0xAAu8 >> ((start - 1) % 8)) & 1) as u64;
                    assert_eq!(read_bits(&buf, start - 1, 1), Ok(expected_prev));
                }
                if start + width < buf.len() * 8 {
                    let after = start + width;
                    let expected_next = ((0xAAu8 >> (after % 8)) & 1) as u64;
                    assert_eq!(read_bits(&buf, after, 1), Ok(expected_next));
                }
            }
        }
    }

    #[test]
    fn write_rejects_overflow_without_truncating() {
        let mut buf = [0u8; 8];
        assert_eq!(
            write_bits(&mut buf, 5, 3, 8),
            Err(BitWriteError::ValueOverflow { value: 8, width: 3 })
        );
        assert_eq!(buf, [0u8; 8]);
        assert_eq!(write_masked_bits(&mut buf, 5, 3, 0xFF), Ok(()));
        assert_eq!(read_bits(&buf, 5, 3), Ok(7));
    }

    #[test]
    fn read_at_end_of_128_byte_block_does_not_overread() {
        let mut block = [0u8; BLOCK_BYTES];
        block[127] = 0b1101_0010;
        assert_eq!(read_bits(&block, 1022, 2), Ok(0b11));
        assert_eq!(read_bits(&block, 1016, 8), Ok(0b1101_0010));
        assert!(read_bits(&block, 1023, 2).is_err());
        assert!(read_bits(&block, 1024, 1).is_err());
    }
}
