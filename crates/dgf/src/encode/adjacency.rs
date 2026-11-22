//! Deterministic directed-edge adjacency built from source vertex IDs.
//!
//! Safely handles boundary edges, non-manifold edges, duplicate faces, disconnected
//! components, and mixed winding without assuming a manifold surface.

/// A directed edge belonging to a candidate triangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DirectedEdgeRecord {
    u: u32,
    v: u32,
    tri: usize,
    edge: u8,
}

/// Deterministic directed-edge adjacency structure over a slice of candidate triangles.
#[derive(Debug, Clone)]
pub struct AdjacencyGraph {
    edges: Vec<DirectedEdgeRecord>,
}

impl AdjacencyGraph {
    /// Builds deterministic directed-edge adjacency from candidate triangles `[v0, v1, v2]`.
    ///
    /// Edge indices follow DGF conventions:
    /// - Edge 0: `(v0, v1)`
    /// - Edge 1: `(v1, v2)`
    /// - Edge 2: `(v2, v0)`
    pub fn build(triangles: &[[u32; 3]]) -> Self {
        let mut edges = Vec::with_capacity(triangles.len() * 3);
        for (tri, &[a, b, c]) in triangles.iter().enumerate() {
            edges.push(DirectedEdgeRecord {
                u: a,
                v: b,
                tri,
                edge: 0,
            });
            edges.push(DirectedEdgeRecord {
                u: b,
                v: c,
                tri,
                edge: 1,
            });
            edges.push(DirectedEdgeRecord {
                u: c,
                v: a,
                tri,
                edge: 2,
            });
        }
        edges.sort_unstable();
        Self { edges }
    }

    /// Iterates over all triangles containing the directed edge `(u, v)` in deterministic
    /// `(tri_index, edge_index)` order.
    ///
    /// To find winding-preserving neighbors across directed edge `(a, b)`, query `(b, a)`.
    pub fn triangles_with_directed_edge(
        &self,
        u: u32,
        v: u32,
    ) -> impl Iterator<Item = (usize, u8)> + '_ {
        let start = self.edges.partition_point(|r| (r.u, r.v) < (u, v));
        self.edges[start..]
            .iter()
            .take_while(move |r| r.u == u && r.v == v)
            .map(|r| (r.tri, r.edge))
    }

    /// Counts remaining unvisited winding-compatible neighbors of `tri` across all 3 edges.
    pub fn unvisited_valence(&self, triangles: &[[u32; 3]], visited: &[bool], tri: usize) -> usize {
        let [a, b, c] = triangles[tri];
        let directed = [(a, b), (b, c), (c, a)];
        let mut count = 0usize;
        for (u, v) in directed {
            for (nbr, _) in self.triangles_with_directed_edge(v, u) {
                if nbr != tri && !visited[nbr] {
                    count += 1;
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_manifold_non_manifold_and_mixed_winding() {
        // Tri 0: [0, 1, 2] -> edge1 (1, 2)
        // Tri 1: [2, 1, 3] -> has (2, 1): winding-compatible with Tri 0!
        // Tri 2: [2, 1, 4] -> non-manifold second fan wing sharing (2, 1)
        // Tri 3: [1, 2, 5] -> mixed winding sharing (1, 2): NOT compatible with Tri 0's (1, 2)
        let tris = [[0, 1, 2], [2, 1, 3], [2, 1, 4], [1, 2, 5]];
        let graph = AdjacencyGraph::build(&tris);

        let compatible: Vec<(usize, u8)> = graph.triangles_with_directed_edge(2, 1).collect();
        assert_eq!(compatible, vec![(1, 0), (2, 0)]);

        // Same-orientation edge (1, 2) is not returned when querying reversed (2, 1)
        assert!(!compatible.iter().any(|&(t, _)| t == 3));
    }
}
