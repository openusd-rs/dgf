//! Deterministic mesh-wide quantization, error-bounded exponent selection, and error reporting.

use crate::error::EncodeError;
use crate::header::{
    ANCHOR_MAX, ANCHOR_MIN, BiasedExponent, MAX_UNBIASED_EXPONENT, MIN_UNBIASED_EXPONENT,
    QuantizationExponent,
};

/// Quantization precision policy for [`crate::encode::encode_mesh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationMode {
    /// Use an explicit unbiased exponent `e` in `-126..=105` (grid step `2^e`).
    ExplicitExponent(i16),
    /// Select a supported exponent guaranteeing maximum absolute coordinate error `<= tolerance`
    /// in object-space units. Fails if unachievable rather than silently coarsening.
    MaxCoordError(f32),
    /// Named baker heuristic choosing the finest exponent that fits coordinates within `1..=24`
    /// signed integer bits (bounded by the DGF1 signed 24-bit limit).
    TargetBitWidth(u8),
}

/// Summary of quantization accuracy and geometry diagnostics for an encoded mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationReport {
    /// Selected mesh-wide unbiased quantization exponent (`-126..=105`).
    pub exponent: QuantizationExponent,
    /// Corresponding biased exponent stored in DGF1 headers (`1..=232`).
    pub biased_exponent: BiasedExponent,
    /// Exact grid step `2^e`.
    pub grid_step: f64,
    /// Maximum measured `|source_axis - decoded_axis|` across all retained vertices.
    pub max_coord_error: f64,
    /// Maximum measured Euclidean vertex displacement across all retained vertices.
    pub max_euclidean_error: f64,
    /// Root-mean-square Euclidean vertex displacement across all retained vertices.
    pub rms_error: f64,
    /// Number of retained triangles whose quantized vertices became coincident (collapsed).
    pub collapsed_triangles: usize,
    /// Number of supplied source vertices not referenced by any retained triangle.
    pub unused_vertices: usize,
}

/// Mesh-wide quantized integer positions and reconstructed `f32` coordinates shared by all blocks.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedMesh {
    /// Quantized signed 24-bit integer positions per source vertex.
    pub positions: Vec<[i32; 3]>,
    /// Reconstructed `f32` positions `q * 2^e` per source vertex.
    pub reconstructed: Vec<[f32; 3]>,
    /// Selected mesh-wide exponent.
    pub exponent: QuantizationExponent,
    /// Measured accuracy and diagnostic report.
    pub report: QuantizationReport,
}

/// Quantizes all source vertices once using deterministic IEEE-754 round-ties-to-even in `f64`.
///
/// Every supplied vertex position is validated to be finite before quantization.
/// `retained_triangles` is used to measure error over active geometry, count unused vertices,
/// and count triangles that collapse after quantization.
pub fn quantize_mesh(
    positions: &[[f32; 3]],
    retained_triangles: &[[u32; 3]],
    mode: QuantizationMode,
    reject_collapses: bool,
) -> Result<QuantizedMesh, EncodeError> {
    // Validate all supplied input vertices are finite.
    for (v, pos) in positions.iter().enumerate() {
        for (axis, &val) in pos.iter().enumerate() {
            if !val.is_finite() {
                return Err(EncodeError::NonFiniteCoordinate {
                    vertex: v,
                    axis: axis as u8,
                    value: val,
                });
            }
        }
    }

    let mut used = vec![false; positions.len()];
    for tri in retained_triangles {
        for &idx in tri {
            if let Some(slot) = used.get_mut(idx as usize) {
                *slot = true;
            }
        }
    }
    let unused_vertices = used.iter().filter(|&&u| !u).count();

    if positions.is_empty() || retained_triangles.is_empty() {
        let exp = match mode {
            QuantizationMode::ExplicitExponent(e) => {
                QuantizationExponent::new(e).ok_or(EncodeError::InvalidExponent { exponent: e })?
            }
            QuantizationMode::MaxCoordError(tol) => {
                if !tol.is_finite() || tol < 0.0 {
                    return Err(EncodeError::InvalidTolerance { tolerance: tol });
                }
                QuantizationExponent::new(0).expect("exponent 0 is valid")
            }
            QuantizationMode::TargetBitWidth(bits) => {
                if bits == 0 || bits > 24 {
                    return Err(EncodeError::InvalidTargetBitWidth { bits });
                }
                QuantizationExponent::new(0).expect("exponent 0 is valid")
            }
        };
        let mut q_pos = Vec::with_capacity(positions.len());
        let mut r_pos = Vec::with_capacity(positions.len());
        if !positions.is_empty() {
            let attempted = try_quantize_at_exponent(positions, &used, exp)?;
            q_pos = attempted.0;
            r_pos = attempted.1;
        }
        return Ok(QuantizedMesh {
            positions: q_pos,
            reconstructed: r_pos,
            exponent: exp,
            report: QuantizationReport {
                exponent: exp,
                biased_exponent: exp.biased(),
                grid_step: exp.step_f64(),
                max_coord_error: 0.0,
                max_euclidean_error: 0.0,
                rms_error: 0.0,
                collapsed_triangles: 0,
                unused_vertices,
            },
        });
    }

    let (exp, q_positions, reconstructed, max_coord_err, max_euclid_err, rms_err) = match mode {
        QuantizationMode::ExplicitExponent(e) => {
            let exp =
                QuantizationExponent::new(e).ok_or(EncodeError::InvalidExponent { exponent: e })?;
            let (q, r, mc, me, rms) = try_quantize_at_exponent(positions, &used, exp)?;
            (exp, q, r, mc, me, rms)
        }
        QuantizationMode::MaxCoordError(tolerance) => {
            select_error_bounded(positions, &used, tolerance)?
        }
        QuantizationMode::TargetBitWidth(bits) => select_target_bit_width(positions, &used, bits)?,
    };

    let mut collapsed_triangles = 0usize;
    for &[a, b, c] in retained_triangles {
        let qa = q_positions[a as usize];
        let qb = q_positions[b as usize];
        let qc = q_positions[c as usize];
        if qa == qb || qb == qc || qa == qc {
            collapsed_triangles += 1;
        }
    }

    if reject_collapses && collapsed_triangles > 0 {
        return Err(EncodeError::QuantizationCollapsedTriangles {
            count: collapsed_triangles,
        });
    }

    Ok(QuantizedMesh {
        positions: q_positions,
        reconstructed,
        exponent: exp,
        report: QuantizationReport {
            exponent: exp,
            biased_exponent: exp.biased(),
            grid_step: exp.step_f64(),
            max_coord_error: max_coord_err,
            max_euclidean_error: max_euclid_err,
            rms_error: rms_err,
            collapsed_triangles,
            unused_vertices,
        },
    })
}

#[allow(clippy::type_complexity)]
fn try_quantize_at_exponent(
    positions: &[[f32; 3]],
    used: &[bool],
    exponent: QuantizationExponent,
) -> Result<(Vec<[i32; 3]>, Vec<[f32; 3]>, f64, f64, f64), EncodeError> {
    let step_f64 = exponent.step_f64();
    let scale_f32 = exponent.step_f32();
    let min_f64 = f64::from(ANCHOR_MIN);
    let max_f64 = f64::from(ANCHOR_MAX);

    let mut q_positions = Vec::with_capacity(positions.len());
    let mut reconstructed = Vec::with_capacity(positions.len());

    let mut max_coord_err = 0.0f64;
    let mut max_euclid_sq = 0.0f64;
    let mut sum_euclid_sq = 0.0f64;
    let mut used_count = 0usize;

    for (v, pos) in positions.iter().enumerate() {
        let mut qv = [0i32; 3];
        let mut rv = [0.0f32; 3];
        let mut dist_sq = 0.0f64;

        for a in 0..3 {
            let src = pos[a];
            // Canonicalize -0.0 to 0.
            let src_f64 = if src == 0.0 { 0.0f64 } else { f64::from(src) };
            let scaled = src_f64 / step_f64;
            let rounded = scaled.round_ties_even();
            if !rounded.is_finite() || rounded < min_f64 || rounded > max_f64 {
                return Err(EncodeError::QuantizedRangeExceeded {
                    vertex: v,
                    axis: a as u8,
                    value: src,
                    exponent: exponent.value(),
                });
            }
            let q = rounded as i32;
            let rec = (q as f32) * scale_f32;
            if !rec.is_finite() {
                return Err(EncodeError::QuantizedNonFinite {
                    vertex: v,
                    axis: a as u8,
                    exponent: exponent.value(),
                });
            }
            qv[a] = q;
            rv[a] = rec;

            if used[v] {
                let err = (src_f64 - f64::from(rec)).abs();
                if err > max_coord_err {
                    max_coord_err = err;
                }
                dist_sq += err * err;
            }
        }

        if used[v] {
            if dist_sq > max_euclid_sq {
                max_euclid_sq = dist_sq;
            }
            sum_euclid_sq += dist_sq;
            used_count += 1;
        }

        q_positions.push(qv);
        reconstructed.push(rv);
    }

    let rms = if used_count > 0 {
        (sum_euclid_sq / (used_count as f64)).sqrt()
    } else {
        0.0
    };

    Ok((
        q_positions,
        reconstructed,
        max_coord_err,
        max_euclid_sq.sqrt(),
        rms,
    ))
}

#[allow(clippy::type_complexity)]
fn select_error_bounded(
    positions: &[[f32; 3]],
    used: &[bool],
    tolerance: f32,
) -> Result<
    (
        QuantizationExponent,
        Vec<[i32; 3]>,
        Vec<[f32; 3]>,
        f64,
        f64,
        f64,
    ),
    EncodeError,
> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(EncodeError::InvalidTolerance { tolerance });
    }
    let tol_f64 = f64::from(tolerance);

    if tolerance == 0.0 {
        // Exact representability path: find coarsest supported exponent achieving exact zero error.
        for e in (MIN_UNBIASED_EXPONENT..=MAX_UNBIASED_EXPONENT).rev() {
            let exp = QuantizationExponent::new(e).expect("loop within valid range");
            if let Ok((q, r, mc, me, rms)) = try_quantize_at_exponent(positions, used, exp)
                && mc == 0.0
            {
                return Ok((exp, q, r, mc, me, rms));
            }
        }
        return Err(EncodeError::ToleranceUnachievable {
            requested: tolerance,
            reason: "mesh coordinates are not exactly representable on any DGF1 signed 24-bit power-of-two grid; supply a positive tolerance or transform to local coordinates",
        });
    }

    // 1. Identify the largest (coarsest) exponent whose worst-case rounding bound delta/2 <= tolerance.
    let mut best_step_exp: Option<i16> = None;
    for e in (MIN_UNBIASED_EXPONENT..=MAX_UNBIASED_EXPONENT).rev() {
        let exp = QuantizationExponent::new(e).expect("loop within valid range");
        if exp.step_f64() * 0.5 <= tol_f64 {
            best_step_exp = Some(e);
            break;
        }
    }

    let start_e = match best_step_exp {
        Some(e) => e,
        None => {
            // Even the finest DGF1 step (2^-126) has delta/2 > tolerance; still check if actual data fits exact/within tol at e = -126!
            let finest = QuantizationExponent::new(MIN_UNBIASED_EXPONENT).expect("valid");
            if let Ok((q, r, mc, me, rms)) = try_quantize_at_exponent(positions, used, finest)
                && mc <= tol_f64
            {
                return Ok((finest, q, r, mc, me, rms));
            }
            return Err(EncodeError::ToleranceUnachievable {
                requested: tolerance,
                reason: "requested tolerance is smaller than half the finest DGF1 step (2^-127) and input coordinates are not exactly representable",
            });
        }
    };

    // 2. Try start_e first; if it fits range and meets tolerance, use it.
    let start_exp = QuantizationExponent::new(start_e).expect("valid");
    match try_quantize_at_exponent(positions, used, start_exp) {
        Ok((q, r, mc, me, rms)) if mc <= tol_f64 => return Ok((start_exp, q, r, mc, me, rms)),
        Ok(_) => {
            // Try finer exponents in case a boundary rounding case pushed measured error slightly above tol.
            for e in (MIN_UNBIASED_EXPONENT..start_e).rev() {
                let exp = QuantizationExponent::new(e).expect("valid");
                match try_quantize_at_exponent(positions, used, exp) {
                    Ok((q, r, mc, me, rms)) if mc <= tol_f64 => {
                        return Ok((exp, q, r, mc, me, rms));
                    }
                    Err(EncodeError::QuantizedRangeExceeded { .. }) => break,
                    _ => {}
                }
            }
        }
        Err(EncodeError::QuantizedRangeExceeded { .. }) => {
            // Coordinates overflow signed 24-bit at start_e; check if coordinates happen to lie
            // within tolerance of a coarser grid (never coarsen past tolerance!).
            for e in (start_e + 1)..=MAX_UNBIASED_EXPONENT {
                let exp = QuantizationExponent::new(e).expect("valid");
                if let Ok((q, r, mc, me, rms)) = try_quantize_at_exponent(positions, used, exp)
                    && mc <= tol_f64
                {
                    return Ok((exp, q, r, mc, me, rms));
                }
            }
            return Err(EncodeError::ToleranceUnachievable {
                requested: tolerance,
                reason: "world-space coordinate magnitudes exceed signed 24-bit quantized range at the required precision; translate to local object coordinates or relax tolerance",
            });
        }
        Err(other) => return Err(other),
    }

    Err(EncodeError::ToleranceUnachievable {
        requested: tolerance,
        reason: "no supported DGF1 exponent satisfies both signed 24-bit representability and the requested tolerance",
    })
}

#[allow(clippy::type_complexity)]
fn select_target_bit_width(
    positions: &[[f32; 3]],
    used: &[bool],
    bits: u8,
) -> Result<
    (
        QuantizationExponent,
        Vec<[i32; 3]>,
        Vec<[f32; 3]>,
        f64,
        f64,
        f64,
    ),
    EncodeError,
> {
    if bits == 0 || bits > 24 {
        return Err(EncodeError::InvalidTargetBitWidth { bits });
    }

    // Target signed range [-2^(bits-1), 2^(bits-1) - 1] (for bits=1, [-1, 0] / [-1, 1]).
    let (target_min, target_max) = if bits == 1 {
        (-1i32, 1i32)
    } else {
        (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1)
    };

    // Find the finest exponent where all quantized coordinates stay within [target_min, target_max].
    for e in MIN_UNBIASED_EXPONENT..=MAX_UNBIASED_EXPONENT {
        let exp = QuantizationExponent::new(e).expect("valid");
        if let Ok((q, r, mc, me, rms)) = try_quantize_at_exponent(positions, used, exp) {
            let fits_target = q
                .iter()
                .all(|v| v.iter().all(|&c| c >= target_min && c <= target_max));
            if fits_target {
                return Ok((exp, q, r, mc, me, rms));
            }
        }
    }

    // Fallback to coarsest exponent if even target_min/max exceeded (still enforces 24-bit range).
    let coarsest = QuantizationExponent::new(MAX_UNBIASED_EXPONENT).expect("valid");
    let (q, r, mc, me, rms) = try_quantize_at_exponent(positions, used, coarsest)?;
    Ok((coarsest, q, r, mc, me, rms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_ties_to_even_is_deterministic() {
        // At exponent 0 (step = 1.0): 0.5 -> 0, 1.5 -> 2, 2.5 -> 2, -0.5 -> 0, -1.5 -> -2, -2.5 -> -2
        let positions = [[0.5, 1.5, 2.5], [-0.5, -1.5, -2.5], [-0.0, 3.5, -3.5]];
        let tris = [[0, 1, 2]];
        let qm = quantize_mesh(
            &positions,
            &tris,
            QuantizationMode::ExplicitExponent(0),
            false,
        )
        .unwrap();
        assert_eq!(qm.positions[0], [0, 2, 2]);
        assert_eq!(qm.positions[1], [0, -2, -2]);
        assert_eq!(qm.positions[2], [0, 4, -4]);
        assert_eq!(qm.report.max_coord_error, 0.5);
    }

    #[test]
    fn error_bounded_mode_respects_tolerance_and_rejects_impossible_world_coords() {
        let positions = [[0.0, 0.0, 0.0], [1.2345, 0.0, 0.0], [0.0, 2.3456, 0.0]];
        let tris = [[0, 1, 2]];
        let tol = 1e-4f32;
        let qm = quantize_mesh(
            &positions,
            &tris,
            QuantizationMode::MaxCoordError(tol),
            false,
        )
        .unwrap();
        assert!(qm.report.max_coord_error <= f64::from(tol));

        // Exact zero tolerance works for dyadic coordinates and fails for non-dyadic (0.1)
        let dyadic = [[0.0, 0.0, 0.0], [0.25, 0.5, 1.0], [-0.125, 2.0, 0.0]];
        let qm0 =
            quantize_mesh(&dyadic, &tris, QuantizationMode::MaxCoordError(0.0), false).unwrap();
        assert_eq!(qm0.report.max_coord_error, 0.0);

        let non_dyadic = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(matches!(
            quantize_mesh(
                &non_dyadic,
                &tris,
                QuantizationMode::MaxCoordError(0.0),
                false
            ),
            Err(EncodeError::ToleranceUnachievable { .. })
        ));

        // Huge world coordinates with tiny tolerance must fail instead of silently coarsening
        let huge = [
            [1.0e8, 0.0, 0.0],
            [1.0e8 + 0.001, 0.0, 0.0],
            [1.0e8, 0.001, 0.0],
        ];
        assert!(matches!(
            quantize_mesh(&huge, &tris, QuantizationMode::MaxCoordError(1e-4), false),
            Err(EncodeError::ToleranceUnachievable { .. })
        ));
    }

    #[test]
    fn subnormal_input_and_signed_range_boundaries() {
        // Subnormal input quantizes cleanly to 0 at exponent -126
        let sub = f32::from_bits(1);
        let positions = [
            [0.0, 0.0, 0.0],
            [sub, 0.0, 0.0],
            [0.0, f32::MIN_POSITIVE, 0.0],
        ];
        let tris = [[0, 1, 2]];
        let qm = quantize_mesh(
            &positions,
            &tris,
            QuantizationMode::ExplicitExponent(-126),
            false,
        )
        .unwrap();
        assert_eq!(qm.positions[1][0], 0);
        assert_eq!(qm.positions[2][1], 1);

        // Signed 24-bit asymmetry: -8_388_608 fits, +8_388_608 overflows
        let asym_ok = [
            [-8_388_608.0, 8_388_607.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ];
        assert!(
            quantize_mesh(
                &asym_ok,
                &tris,
                QuantizationMode::ExplicitExponent(0),
                false
            )
            .is_ok()
        );
        let asym_overflow = [[8_388_608.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(matches!(
            quantize_mesh(
                &asym_overflow,
                &tris,
                QuantizationMode::ExplicitExponent(0),
                false
            ),
            Err(EncodeError::QuantizedRangeExceeded { .. })
        ));
    }

    #[test]
    fn collapse_reporting_and_strict_rejection() {
        let positions = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let tris = [[0, 1, 2]];
        let qm = quantize_mesh(
            &positions,
            &tris,
            QuantizationMode::ExplicitExponent(0),
            false,
        )
        .unwrap();
        assert_eq!(qm.report.collapsed_triangles, 1);

        assert_eq!(
            quantize_mesh(
                &positions,
                &tris,
                QuantizationMode::ExplicitExponent(0),
                true
            ),
            Err(EncodeError::QuantizationCollapsedTriangles { count: 1 })
        );
    }
}
