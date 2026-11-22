//! Pure Rust DGF1 block planner, quantizer, strip builder, spatial clusterer, and mesh baker.

pub mod adjacency;
pub mod block;
pub mod cluster;
pub mod mesh;
pub mod pack;
pub mod quantize;
pub mod strip;

pub use block::{
    PlannedBlock, PreparedBlock, PreparedOmm, PreparedTopology, encode_block, plan_block,
};
pub use cluster::ClusterConfig;
pub use mesh::{
    DegeneratePolicy, EncodeOptions, EncodeStatistics, EncodedMesh, FitFailureCounts, MeshView,
    PackingPolicy, TriangleMetadata, encode_mesh, encode_mesh_with_user_data,
    encode_preclustered_mesh,
};
pub use quantize::{QuantizationMode, QuantizationReport, QuantizedMesh, quantize_mesh};
