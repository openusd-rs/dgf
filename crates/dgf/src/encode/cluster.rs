//! Coarse spatial clustering using binned SAH with explicit work stack and deterministic fallbacks.

/// Configuration for coarse spatial clustering prior to exact-fit DGF1 block packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterConfig {
    /// Target maximum triangles per coarse cluster (default `128`).
    pub max_triangles: usize,
    /// Target maximum unique source vertices per coarse cluster (default `256`).
    pub max_vertices: usize,
    /// Number of SAH histogram bins per axis for large clusters (default `16`).
    pub num_bins: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            max_triangles: 128,
            max_vertices: 256,
            num_bins: 16,
        }
    }
}

/// Robust `f64` bounding box used for spatial clustering and SAH block packing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb64 {
    /// Minimum corner `[x, y, z]`.
    pub min: [f64; 3],
    /// Maximum corner `[x, y, z]`.
    pub max: [f64; 3],
}

impl Aabb64 {
    /// Empty inverted bounding box identity for union operations.
    pub const EMPTY: Self = Self {
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
    };

    /// Computes the bounding box of a single triangle from 3 vertex positions.
    pub fn from_triangle(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Self {
        let mut out = Self::EMPTY;
        out.grow_point([f64::from(a[0]), f64::from(a[1]), f64::from(a[2])]);
        out.grow_point([f64::from(b[0]), f64::from(b[1]), f64::from(b[2])]);
        out.grow_point([f64::from(c[0]), f64::from(c[1]), f64::from(c[2])]);
        out
    }

    /// Expands this bounding box to include `p`.
    pub fn grow_point(&mut self, p: [f64; 3]) {
        for (a, &val) in p.iter().enumerate() {
            if val < self.min[a] {
                self.min[a] = val;
            }
            if val > self.max[a] {
                self.max[a] = val;
            }
        }
    }

    /// Expands this bounding box to include `other`.
    pub fn union(&mut self, other: &Self) {
        for a in 0..3 {
            if other.min[a] < self.min[a] {
                self.min[a] = other.min[a];
            }
            if other.max[a] > self.max[a] {
                self.max[a] = other.max[a];
            }
        }
    }

    /// Returns the total surface area of this box (`0.0` if empty or degenerate).
    pub fn surface_area(&self) -> f64 {
        if self.min[0] > self.max[0] {
            return 0.0;
        }
        let dx = (self.max[0] - self.min[0]).max(0.0);
        let dy = (self.max[1] - self.min[1]).max(0.0);
        let dz = (self.max[2] - self.min[2]).max(0.0);
        2.0 * (dx * dy + dy * dz + dz * dx)
    }

    /// Returns the sum of box extents `dx + dy + dz` (used to prefer compact tiles on planar geometry).
    pub fn half_perimeter(&self) -> f64 {
        if self.min[0] > self.max[0] {
            return 0.0;
        }
        let dx = (self.max[0] - self.min[0]).max(0.0);
        let dy = (self.max[1] - self.min[1]).max(0.0);
        let dz = (self.max[2] - self.min[2]).max(0.0);
        dx + dy + dz
    }

    /// Converts bounds to `[f32; 3]` min/max pair for reporting.
    pub fn to_f32_pair(&self) -> ([f32; 3], [f32; 3]) {
        (
            [self.min[0] as f32, self.min[1] as f32, self.min[2] as f32],
            [self.max[0] as f32, self.max[1] as f32, self.max[2] as f32],
        )
    }
}

/// Precomputed spatial geometry for one retained triangle.
#[derive(Debug, Clone, Copy)]
pub struct TriSpatial {
    /// Original source triangle index (used for deterministic tie breaking).
    pub input_triangle: u32,
    /// Triangle AABB in `f64`.
    pub aabb: Aabb64,
    /// Triangle centroid in `f64`.
    pub centroid: [f64; 3],
}

impl TriSpatial {
    /// Computes bounding box and centroid for a triangle in `f64` (avoids `f32` overflow).
    pub fn new(input_triangle: u32, a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Self {
        let aabb = Aabb64::from_triangle(a, b, c);
        let centroid = [
            (f64::from(a[0]) + f64::from(b[0]) + f64::from(c[0])) / 3.0,
            (f64::from(a[1]) + f64::from(b[1]) + f64::from(c[1])) / 3.0,
            (f64::from(a[2]) + f64::from(b[2]) + f64::from(c[2])) / 3.0,
        ];
        Self {
            input_triangle,
            aabb,
            centroid,
        }
    }
}

/// Partitions `indices` (indices into `triangles` and `spatials`) into coarse spatial clusters
/// using binned SAH and an explicit work stack.
///
/// Always terminates: every split produces two strictly nonempty subsets.
pub fn build_coarse_clusters(
    indices: Vec<usize>,
    triangles: &[[u32; 3]],
    spatials: &[TriSpatial],
    config: ClusterConfig,
) -> Vec<Vec<usize>> {
    if indices.is_empty() {
        return Vec::new();
    }

    let max_tris = config.max_triangles.max(1);
    let max_verts = config.max_vertices.max(3);
    let num_bins = config.num_bins.clamp(2, 64);

    let mut result = Vec::new();
    let mut stack: Vec<Vec<usize>> = Vec::new();
    stack.push(indices);

    let mut vert_scratch: Vec<u32> = Vec::new();

    while let Some(cluster) = stack.pop() {
        let fits_tris = cluster.len() <= max_tris;
        let fits_verts = if fits_tris {
            count_unique_vertices(&cluster, triangles, &mut vert_scratch) <= max_verts
        } else {
            false
        };

        if cluster.len() <= 1 || (fits_tris && fits_verts) {
            result.push(cluster);
            continue;
        }

        let (left, right) = split_binned_sah(cluster, spatials, num_bins);
        // Push right first so left is popped and emitted first (deterministic spatial order).
        stack.push(right);
        stack.push(left);
    }

    result
}

fn count_unique_vertices(
    cluster: &[usize],
    triangles: &[[u32; 3]],
    scratch: &mut Vec<u32>,
) -> usize {
    scratch.clear();
    scratch.reserve(cluster.len() * 3);
    for &idx in cluster {
        scratch.extend_from_slice(&triangles[idx]);
    }
    scratch.sort_unstable();
    scratch.dedup();
    scratch.len()
}

fn split_binned_sah(
    mut cluster: Vec<usize>,
    spatials: &[TriSpatial],
    num_bins: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = cluster.len();
    debug_assert!(n >= 2);

    // Compute centroid bounds across the cluster.
    let mut c_bounds = Aabb64::EMPTY;
    for &idx in &cluster {
        c_bounds.grow_point(spatials[idx].centroid);
    }

    let mut best_cost = f64::INFINITY;
    let mut best_perim = f64::INFINITY;
    let mut best_balance = usize::MAX;
    let mut best_axis: Option<usize> = None;
    let mut best_split_bin = 0usize;

    let mut bin_counts = vec![0usize; num_bins];
    let mut bin_bounds = vec![Aabb64::EMPTY; num_bins];
    let mut prefix_bounds = vec![Aabb64::EMPTY; num_bins];
    let mut prefix_counts = vec![0usize; num_bins];

    for axis in 0..3 {
        let extent = c_bounds.max[axis] - c_bounds.min[axis];
        if extent <= 0.0 || !extent.is_finite() {
            continue;
        }

        bin_counts.fill(0);
        bin_bounds.fill(Aabb64::EMPTY);

        let inv_extent = (num_bins as f64) / extent;
        for &idx in &cluster {
            let rel = (spatials[idx].centroid[axis] - c_bounds.min[axis]) * inv_extent;
            let bin = (rel as usize).min(num_bins - 1);
            bin_counts[bin] += 1;
            bin_bounds[bin].union(&spatials[idx].aabb);
        }

        // Prefix pass
        let mut running_box = Aabb64::EMPTY;
        let mut running_count = 0usize;
        for b in 0..num_bins {
            running_count += bin_counts[b];
            running_box.union(&bin_bounds[b]);
            prefix_counts[b] = running_count;
            prefix_bounds[b] = running_box;
        }

        // Suffix pass and SAH evaluation for split between bin `b` and `b + 1`
        let mut right_box = Aabb64::EMPTY;
        let mut right_count = 0usize;
        for b in (0..num_bins - 1).rev() {
            right_count += bin_counts[b + 1];
            right_box.union(&bin_bounds[b + 1]);
            let left_count = prefix_counts[b];
            if left_count == 0 || right_count == 0 {
                continue;
            }
            let cost = prefix_bounds[b].surface_area() * (left_count as f64)
                + right_box.surface_area() * (right_count as f64);
            if !cost.is_finite() {
                continue;
            }
            let perim = prefix_bounds[b].half_perimeter() * (left_count as f64)
                + right_box.half_perimeter() * (right_count as f64);
            let balance = left_count.abs_diff(right_count);
            let better = match cost.total_cmp(&best_cost) {
                core::cmp::Ordering::Less => true,
                core::cmp::Ordering::Equal => match perim.total_cmp(&best_perim) {
                    core::cmp::Ordering::Less => true,
                    core::cmp::Ordering::Equal => {
                        balance < best_balance
                            || (balance == best_balance && best_axis.is_none_or(|a| axis < a))
                    }
                    core::cmp::Ordering::Greater => false,
                },
                core::cmp::Ordering::Greater => false,
            };
            if better {
                best_cost = cost;
                best_perim = perim;
                best_balance = balance;
                best_axis = Some(axis);
                best_split_bin = b;
            }
        }
    }

    if let Some(axis) = best_axis {
        let extent = c_bounds.max[axis] - c_bounds.min[axis];
        let inv_extent = (num_bins as f64) / extent;
        let mut left = Vec::with_capacity(n / 2 + 1);
        let mut right = Vec::with_capacity(n / 2 + 1);
        for idx in cluster {
            let rel = (spatials[idx].centroid[axis] - c_bounds.min[axis]) * inv_extent;
            let bin = (rel as usize).min(num_bins - 1);
            if bin <= best_split_bin {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }
        if !left.is_empty() && !right.is_empty() {
            return (left, right);
        }
        // Fallback if numerical rounding produced an empty side.
        cluster = if left.is_empty() { right } else { left };
    }

    // Deterministic median fallback for zero-area / equal-centroid clusters.
    cluster.sort_unstable_by_key(|&idx| spatials[idx].input_triangle);
    let mid = cluster.len() / 2;
    let right = cluster.split_off(mid);
    (cluster, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_centroids_and_zero_area_terminate_deterministically() {
        let tris = vec![[0, 1, 2]; 300];
        let spatials: Vec<_> = (0..300)
            .map(|i| TriSpatial::new(i as u32, [0.0; 3], [0.0; 3], [0.0; 3]))
            .collect();
        let indices: Vec<usize> = (0..300).collect();
        let clusters = build_coarse_clusters(indices, &tris, &spatials, ClusterConfig::default());
        let total: usize = clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total, 300);
        assert!(clusters.iter().all(|c| !c.is_empty() && c.len() <= 128));
    }
}
