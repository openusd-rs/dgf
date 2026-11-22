//! DGF1 triangle control stream, compressed strip index stream, and strip-to-triangle expansion.

use crate::bits::{BLOCK_BITS, BLOCK_BYTES, read_bits};
use crate::error::DecodeError;
use crate::header::{Header, MAX_TRIANGLES};
use crate::layout::{BlockLayout, MAX_STRIP_INDICES};

/// Per-triangle strip topology control code (2 bits in the backward control stream).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriControl {
    /// Start a new triangle from the next 3 strip indices `[n0, n1, n2]`.
    Restart = 0,
    /// Reuse reversed second edge `[c, b]` of previous triangle `[a, b, c]` with next index `n`.
    Edge1 = 1,
    /// Reuse reversed third edge `[a, c]` of previous triangle `[a, b, c]` with next index `n`.
    Edge2 = 2,
    /// Backtrack to the unused edge of `prev_prev` (legal only immediately after `Edge1` or `Edge2`).
    Backtrack = 3,
}

impl TriControl {
    /// Converts a raw 2-bit value (`0..=3`) into a [`TriControl`].
    pub const fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Restart),
            1 => Some(Self::Edge1),
            2 => Some(Self::Edge2),
            3 => Some(Self::Backtrack),
            _ => None,
        }
    }

    /// Returns the number of strip indices consumed by this control (`3` for `Restart`, `1` otherwise).
    pub const fn index_count(self) -> usize {
        match self {
            Self::Restart => 3,
            Self::Edge1 | Self::Edge2 | Self::Backtrack => 1,
        }
    }
}

/// Decodes and validates the triangle control stream from the end of `bytes`.
///
/// Returns `(controls, strip_index_count)`. `controls[0]` is always [`TriControl::Restart`].
pub fn decode_controls(
    bytes: &[u8; BLOCK_BYTES],
    num_triangles: u8,
) -> Result<([TriControl; MAX_TRIANGLES as usize], usize), DecodeError> {
    if num_triangles == 0 || num_triangles > MAX_TRIANGLES {
        return Err(DecodeError::ArithmeticOverflow {
            context: "num_triangles out of 1..=64",
        });
    }

    let mut controls = [TriControl::Restart; MAX_TRIANGLES as usize];
    let mut strip_index_count = 3usize;

    for i in 1..usize::from(num_triangles) {
        let bit_pos = BLOCK_BITS - 2 * i;
        let raw = read_bits(bytes, bit_pos, 2)? as u8;
        let ctrl = TriControl::from_u8(raw).ok_or(DecodeError::BitOutOfBounds {
            start: bit_pos,
            width: 2,
        })?;
        if ctrl == TriControl::Backtrack {
            let prev = controls[i - 1];
            if i < 2 || !matches!(prev, TriControl::Edge1 | TriControl::Edge2) {
                return Err(DecodeError::IllegalBacktrack {
                    triangle: i as u8,
                    prev_control: Some(prev as u8),
                });
            }
        }
        controls[i] = ctrl;
        strip_index_count += ctrl.index_count();
    }

    Ok((controls, strip_index_count))
}

/// Scans the backward first-use flag stream immediately preceding the control stream.
///
/// Returns `(is_first_flags, new_vertex_count, repeated_index_count)`.
/// The first three strip indices (`0, 1, 2`) are implicitly first uses.
pub fn scan_first_use_flags(
    bytes: &[u8; BLOCK_BYTES],
    num_triangles: u8,
    strip_index_count: usize,
) -> Result<([bool; MAX_STRIP_INDICES], usize, usize), DecodeError> {
    if !(3..=MAX_STRIP_INDICES).contains(&strip_index_count) {
        return Err(DecodeError::ArithmeticOverflow {
            context: "strip_index_count out of 3..=192",
        });
    }

    let control_bits = 2 * (usize::from(num_triangles) - 1);
    let mut is_first = [false; MAX_STRIP_INDICES];
    is_first[0] = true;
    is_first[1] = true;
    is_first[2] = true;

    let mut new_vertex_count = 3usize;
    let mut repeated_count = 0usize;

    for (j, flag_slot) in is_first
        .iter_mut()
        .enumerate()
        .take(strip_index_count)
        .skip(3)
    {
        // Bit offset = 1024 - 2*(T - 1) - 1 - (j - 3)
        let bit_pos = BLOCK_BITS.checked_sub(control_bits + 1 + (j - 3)).ok_or(
            DecodeError::SectionOverlap {
                repeat_end: 0,
                tail_start: 0,
            },
        )?;
        let bit = read_bits(bytes, bit_pos, 1)?;
        if bit != 0 {
            *flag_slot = true;
            new_vertex_count += 1;
        } else {
            repeated_count += 1;
        }
    }

    Ok((is_first, new_vertex_count, repeated_count))
}

/// Decodes and validates the full strip vertex index stream (`0..layout.strip_index_count`).
///
/// Enforces:
/// - The implicit initial vertices `0, 1, 2` are `< header.num_vertices`.
/// - Every newly introduced vertex is `< header.num_vertices`.
/// - Every repeated vertex index is `< next_new_vertex` (already introduced) and `< header.num_vertices`.
pub fn decode_strip_indices(
    bytes: &[u8; BLOCK_BYTES],
    header: &Header,
    layout: &BlockLayout,
    is_first: &[bool; MAX_STRIP_INDICES],
) -> Result<[u8; MAX_STRIP_INDICES], DecodeError> {
    let num_vertices = header.num_vertices;
    for init_idx in 0u8..3 {
        if init_idx >= num_vertices {
            return Err(DecodeError::VertexIndexOutOfRange {
                strip_position: init_idx,
                index: init_idx,
                num_vertices,
            });
        }
    }

    let mut indices = [0u8; MAX_STRIP_INDICES];
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;

    let mut next_new: u8 = 3;
    let mut repeat_bit = layout.front.repeat_start;
    let bpi = usize::from(header.bits_per_index);

    for j in 3..layout.strip_index_count {
        if is_first[j] {
            if next_new >= num_vertices {
                return Err(DecodeError::VertexIndexOutOfRange {
                    strip_position: j as u8,
                    index: next_new,
                    num_vertices,
                });
            }
            indices[j] = next_new;
            next_new += 1;
        } else {
            let idx = read_bits(bytes, repeat_bit, bpi)? as u8;
            repeat_bit += bpi;
            if idx >= num_vertices {
                return Err(DecodeError::VertexIndexOutOfRange {
                    strip_position: j as u8,
                    index: idx,
                    num_vertices,
                });
            }
            if idx >= next_new {
                return Err(DecodeError::RepeatedIndexNotYetIntroduced {
                    strip_position: j as u8,
                    index: idx,
                    next_new_vertex: next_new,
                });
            }
            indices[j] = idx;
        }
    }

    debug_assert_eq!(repeat_bit, layout.repeat_end);
    Ok(indices)
}

/// Expands a validated control stream and strip index stream into a triangle index list.
///
/// Winding rules (where `prev = [a, b, c]`, `prev_prev = [d, e, f]`, next index = `n`):
/// - `Restart`: `[s[p], s[p + 1], s[p + 2]]`
/// - `Edge1`: `[c, b, n]`
/// - `Edge2`: `[a, c, n]`
/// - `Backtrack` after `Edge1`: `[d, f, n]`
/// - `Backtrack` after `Edge2`: `[f, e, n]`
pub fn expand_strip_to_triangles(
    controls: &[TriControl],
    num_triangles: u8,
    strip_indices: &[u8],
    strip_index_count: usize,
) -> Result<[[u8; 3]; MAX_TRIANGLES as usize], DecodeError> {
    if num_triangles == 0 || usize::from(num_triangles) > controls.len() {
        return Err(DecodeError::ArithmeticOverflow {
            context: "num_triangles exceeds controls slice",
        });
    }
    if strip_index_count < 3 || strip_index_count > strip_indices.len() {
        return Err(DecodeError::ArithmeticOverflow {
            context: "strip_index_count exceeds strip_indices slice",
        });
    }

    let mut tris = [[0u8; 3]; MAX_TRIANGLES as usize];
    tris[0] = [strip_indices[0], strip_indices[1], strip_indices[2]];
    let mut pos = 3usize;

    for i in 1..usize::from(num_triangles) {
        let prev = tris[i - 1];
        let tri = match controls[i] {
            TriControl::Restart => {
                if pos + 3 > strip_index_count {
                    return Err(DecodeError::ArithmeticOverflow {
                        context: "strip index underflow on RESTART",
                    });
                }
                let t = [
                    strip_indices[pos],
                    strip_indices[pos + 1],
                    strip_indices[pos + 2],
                ];
                pos += 3;
                t
            }
            TriControl::Edge1 => {
                if pos + 1 > strip_index_count {
                    return Err(DecodeError::ArithmeticOverflow {
                        context: "strip index underflow on EDGE1",
                    });
                }
                let n = strip_indices[pos];
                pos += 1;
                [prev[2], prev[1], n]
            }
            TriControl::Edge2 => {
                if pos + 1 > strip_index_count {
                    return Err(DecodeError::ArithmeticOverflow {
                        context: "strip index underflow on EDGE2",
                    });
                }
                let n = strip_indices[pos];
                pos += 1;
                [prev[0], prev[2], n]
            }
            TriControl::Backtrack => {
                if i < 2 {
                    return Err(DecodeError::IllegalBacktrack {
                        triangle: i as u8,
                        prev_control: Some(controls[i - 1] as u8),
                    });
                }
                if pos + 1 > strip_index_count {
                    return Err(DecodeError::ArithmeticOverflow {
                        context: "strip index underflow on BACKTRACK",
                    });
                }
                let n = strip_indices[pos];
                pos += 1;
                let prev_prev = tris[i - 2];
                match controls[i - 1] {
                    TriControl::Edge1 => [prev_prev[0], prev_prev[2], n],
                    TriControl::Edge2 => [prev_prev[2], prev_prev[1], n],
                    other => {
                        return Err(DecodeError::IllegalBacktrack {
                            triangle: i as u8,
                            prev_control: Some(other as u8),
                        });
                    }
                }
            }
        };
        tris[i] = tri;
    }

    if pos != strip_index_count {
        return Err(DecodeError::ArithmeticOverflow {
            context: "unconsumed strip indices after expansion",
        });
    }

    Ok(tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_all_four_controls_and_both_backtrack_directions() {
        // Tri 0: RESTART -> [0, 1, 2]
        // Tri 1: EDGE1   -> [2, 1, 3]
        // Tri 2: BACKTRACK (after EDGE1, prev_prev=[0,1,2]) -> [0, 2, 4]
        // Tri 3: EDGE2   (from [0, 2, 4]) -> [0, 4, 5]
        // Tri 4: BACKTRACK (after EDGE2, prev_prev=[0,2,4]) -> [4, 2, 6]
        // Tri 5: RESTART -> [7, 8, 9]
        let controls = [
            TriControl::Restart,
            TriControl::Edge1,
            TriControl::Backtrack,
            TriControl::Edge2,
            TriControl::Backtrack,
            TriControl::Restart,
        ];
        let strip = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let tris = expand_strip_to_triangles(&controls, 6, &strip, strip.len()).unwrap();
        assert_eq!(tris[0], [0, 1, 2]);
        assert_eq!(tris[1], [2, 1, 3]);
        assert_eq!(tris[2], [0, 2, 4]);
        assert_eq!(tris[3], [0, 4, 5]);
        assert_eq!(tris[4], [4, 2, 6]);
        assert_eq!(tris[5], [7, 8, 9]);
    }

    #[test]
    fn illegal_backtrack_contexts_rejected() {
        // Illegal: triangle 1 BACKTRACK (prev is RESTART at 0)
        let mut bytes = [0u8; BLOCK_BYTES];
        // Tri 1 control at bit 1022..1024 = 0b11 (BACKTRACK)
        bytes[127] = 0b1100_0000;
        assert_eq!(
            decode_controls(&bytes, 2),
            Err(DecodeError::IllegalBacktrack {
                triangle: 1,
                prev_control: Some(0)
            })
        );

        // Illegal: Tri 1 = EDGE1 (1), Tri 2 = BACKTRACK (3), Tri 3 = BACKTRACK (3)
        let mut bytes2 = [0u8; BLOCK_BYTES];
        // i=1 at 1022: 01, i=2 at 1020: 11, i=3 at 1018: 11 -> byte 127 = 0b01_11_11_00
        bytes2[127] = (1 << 6) | (3 << 4) | (3 << 2);
        assert_eq!(
            decode_controls(&bytes2, 4),
            Err(DecodeError::IllegalBacktrack {
                triangle: 3,
                prev_control: Some(3)
            })
        );

        // Illegal: Tri 1 = RESTART (0), Tri 2 = BACKTRACK (3)
        let mut bytes3 = [0u8; BLOCK_BYTES];
        bytes3[127] = 3 << 4;
        assert_eq!(
            decode_controls(&bytes3, 3),
            Err(DecodeError::IllegalBacktrack {
                triangle: 2,
                prev_control: Some(0)
            })
        );
    }
}
