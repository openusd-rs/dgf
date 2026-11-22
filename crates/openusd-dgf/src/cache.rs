//! Content-addressed cache for baked DGF1 geometry, keyed by resolved mesh content and options.
//!
//! Transforms and prim paths are intentionally excluded from the cache key so moving a prim
//! or placing multiple instances reuses the same `Arc<BakedUsdMesh>` without rebaking.

use std::collections::HashMap;
use std::sync::Arc;

use dgf::{DegeneratePolicy, EncodeOptions, PackingPolicy, QuantizationMode};
use openusd::usd;
use openusd_schemas::geom::{self, Orientation, UpAxis, XformCache};

use crate::bake::{BakedUsdMesh, UsdBakeOptions, UsdMeshInstance, bake_extracted_mesh};
use crate::error::UsdDgfError;
use crate::extract::{ExtractedAttribute, ExtractedUsdMesh, UsdAttributeValues, extract_usd_mesh};

/// Deterministic 256-bit SHA-256 content key for a resolved USD mesh and encoding options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

/// Whether a cache query reused an existing baked entry or baked a new one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStatus {
    /// Existing [`BakedUsdMesh`] reused without rebaking.
    Hit,
    /// Mesh was freshly baked and inserted into the cache.
    Miss,
}

/// Content-addressed cache storing shared `Arc<BakedUsdMesh>` entries.
#[derive(Debug, Default)]
pub struct UsdDgfCache {
    entries: HashMap<ContentHash, Arc<BakedUsdMesh>>,
    hits: usize,
    misses: usize,
}

impl UsdDgfCache {
    /// Creates an empty geometry cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of distinct cached geometries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no geometries are cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns total cache hits recorded.
    pub fn hits(&self) -> usize {
        self.hits
    }

    /// Returns total cache misses (fresh bakes) recorded.
    pub fn misses(&self) -> usize {
        self.misses
    }

    /// Clears all cached entries and resets counters.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Extracts `mesh` at `time`, checks the content cache, and returns a placed
    /// [`UsdMeshInstance`] sharing `Arc<BakedUsdMesh>` across identical content.
    pub fn bake_or_get(
        &mut self,
        mesh: &geom::Mesh,
        time: impl Into<Option<usd::TimeCode>>,
        xform_cache: &mut XformCache,
        options: &UsdBakeOptions,
    ) -> Result<(UsdMeshInstance, CacheStatus), UsdDgfError> {
        let extracted = extract_usd_mesh(mesh, time, xform_cache, &options.extract)?;
        let key = compute_content_hash(&extracted, &options.encode);

        if let Some(existing) = self.entries.get(&key) {
            self.hits += 1;
            return Ok((
                UsdMeshInstance {
                    prim_path: extracted.prim_path,
                    world_transform: extracted.world_transform,
                    geometry: Arc::clone(existing),
                },
                CacheStatus::Hit,
            ));
        }

        let baked = Arc::new(bake_extracted_mesh(&extracted, &options.encode)?);
        self.entries.insert(key, Arc::clone(&baked));
        self.misses += 1;

        Ok((
            UsdMeshInstance {
                prim_path: extracted.prim_path,
                world_transform: extracted.world_transform,
                geometry: baked,
            },
            CacheStatus::Miss,
        ))
    }

    /// Traverses all active `UsdGeomMesh` prims on `stage` at `time` and bakes or reuses
    /// cached DGF1 geometry for each instance.
    pub fn bake_stage(
        &mut self,
        stage: &usd::Stage,
        time: impl Into<Option<usd::TimeCode>>,
        options: &UsdBakeOptions,
    ) -> Result<Vec<UsdMeshInstance>, UsdDgfError> {
        let time = time.into();
        let mut xform_cache = XformCache::default();
        xform_cache.set_time(time);
        let mut instances = Vec::new();
        for name in stage.root_prims()? {
            let root = stage
                .prim(format!("/{name}"))
                .map_err(|e| UsdDgfError::Usd(e.into()))?;
            self.visit_prim(&root, time, &mut xform_cache, options, &mut instances)?;
        }
        Ok(instances)
    }

    fn visit_prim(
        &mut self,
        prim: &usd::Prim,
        time: Option<usd::TimeCode>,
        xform_cache: &mut XformCache,
        options: &UsdBakeOptions,
        out: &mut Vec<UsdMeshInstance>,
    ) -> Result<(), UsdDgfError> {
        if let Some(mesh) = geom::Mesh::from_prim(prim.clone())? {
            let (instance, _status) = self.bake_or_get(&mesh, time, xform_cache, options)?;
            out.push(instance);
        }
        for child in prim.children()? {
            self.visit_prim(&child, time, xform_cache, options, out)?;
        }
        Ok(())
    }
}

/// Computes a deterministic SHA-256 content hash over resolved geometry, attributes, subsets,
/// scene conventions, and DGF1 encode options — deliberately excluding prim path and transform.
pub fn compute_content_hash(extracted: &ExtractedUsdMesh, encode: &EncodeOptions) -> ContentHash {
    let mut h = Sha256::new();
    h.update(b"openusd-dgf-v1");
    h.update(dgf::CANONICAL_ENCODER_POLICY.as_bytes());

    // Positions
    h.update_u64(extracted.positions.len() as u64);
    for p in &extracted.positions {
        for c in p {
            h.update(&c.to_bits().to_le_bytes());
        }
    }

    // Topology
    h.update_u64(extracted.face_vertex_counts.len() as u64);
    for c in &extracted.face_vertex_counts {
        h.update(&c.to_le_bytes());
    }
    h.update_u64(extracted.face_vertex_indices.len() as u64);
    for i in &extracted.face_vertex_indices {
        h.update(&i.to_le_bytes());
    }

    // Metadata & subsets
    h.update_u64(extracted.triangle_metadata.len() as u64);
    for m in &extracted.triangle_metadata {
        h.update(&m.geometry_id.to_le_bytes());
        h.update(&[u8::from(m.opaque)]);
    }
    hash_opt_str(&mut h, extracted.default_material_path.as_deref());
    h.update_u64(extracted.material_subsets.len() as u64);
    for s in &extracted.material_subsets {
        h.update(&s.geometry_id.to_le_bytes());
        hash_str(&mut h, &s.name);
        hash_str(&mut h, &s.family_name);
        hash_opt_str(&mut h, s.material_path.as_deref());
    }

    // Scene metadata
    h.update(&[match extracted.scene_metadata.orientation {
        Orientation::RightHanded => 0,
        Orientation::LeftHanded => 1,
    }]);
    h.update(&[u8::from(extracted.scene_metadata.double_sided)]);
    h.update(&[match extracted.scene_metadata.up_axis {
        UpAxis::Y => 0,
        UpAxis::Z => 1,
    }]);
    h.update(
        &extracted
            .scene_metadata
            .meters_per_unit
            .to_bits()
            .to_le_bytes(),
    );

    // Normals & primvars
    match &extracted.normals {
        None => h.update(&[0]),
        Some(n) => {
            h.update(&[1]);
            hash_attribute(&mut h, n);
        }
    }
    h.update_u64(extracted.primvars.len() as u64);
    for attr in extracted.primvars.values() {
        hash_attribute(&mut h, attr);
    }

    // DGF1 encode options (excluding `parallel`, which guarantees identical output!)
    match encode.quantization {
        QuantizationMode::ExplicitExponent(e) => {
            h.update(&[0]);
            h.update(&e.to_le_bytes());
        }
        QuantizationMode::MaxCoordError(tol) => {
            h.update(&[1]);
            h.update(&tol.to_bits().to_le_bytes());
        }
        QuantizationMode::TargetBitWidth(bits) => {
            h.update(&[2, bits]);
        }
    }
    h.update(&[match encode.packing {
        PackingPolicy::BaselineRestartOnly => 0,
        PackingPolicy::BaselineWithStrips => 1,
        PackingPolicy::Sah => 2,
    }]);
    h.update(&[match encode.degenerate_policy {
        DegeneratePolicy::Reject => 0,
        DegeneratePolicy::DropAndReport => 1,
    }]);
    h.update(&[u8::from(encode.reject_quantization_collapses)]);
    h.update(&encode.prim_id_base.to_le_bytes());
    match encode.user_data {
        None => h.update(&[0]),
        Some(w) => {
            h.update(&[1]);
            h.update(&w.to_le_bytes());
        }
    }
    h.update(&[u8::from(encode.enable_exponent_adjustment)]);
    h.update_u64(encode.cluster_config.max_triangles as u64);
    h.update_u64(encode.cluster_config.max_vertices as u64);
    h.update_u64(encode.cluster_config.num_bins as u64);

    ContentHash(h.finalize())
}

fn hash_str(h: &mut Sha256, s: &str) {
    h.update_u64(s.len() as u64);
    h.update(s.as_bytes());
}

fn hash_opt_str(h: &mut Sha256, s: Option<&str>) {
    match s {
        None => h.update(&[0]),
        Some(text) => {
            h.update(&[1]);
            hash_str(h, text);
        }
    }
}

fn hash_attribute(h: &mut Sha256, attr: &ExtractedAttribute) {
    hash_str(h, &attr.name);
    hash_str(h, attr.interpolation.as_token());
    match &attr.values {
        UsdAttributeValues::Vec2f(v) => {
            h.update(&[0]);
            h.update_u64(v.len() as u64);
            for a in v {
                h.update(&a[0].to_bits().to_le_bytes());
                h.update(&a[1].to_bits().to_le_bytes());
            }
        }
        UsdAttributeValues::Vec3f(v) => {
            h.update(&[1]);
            h.update_u64(v.len() as u64);
            for a in v {
                h.update(&a[0].to_bits().to_le_bytes());
                h.update(&a[1].to_bits().to_le_bytes());
                h.update(&a[2].to_bits().to_le_bytes());
            }
        }
        UsdAttributeValues::Vec4f(v) => {
            h.update(&[2]);
            h.update_u64(v.len() as u64);
            for a in v {
                for c in a {
                    h.update(&c.to_bits().to_le_bytes());
                }
            }
        }
        UsdAttributeValues::Float(v) => {
            h.update(&[3]);
            h.update_u64(v.len() as u64);
            for x in v {
                h.update(&x.to_bits().to_le_bytes());
            }
        }
        UsdAttributeValues::Int(v) => {
            h.update(&[4]);
            h.update_u64(v.len() as u64);
            for x in v {
                h.update(&x.to_le_bytes());
            }
        }
    }
}

/// Minimal streaming pure-Rust SHA-256 (FIPS 180-4) for zero-dependency content hashing.
struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_bytes: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_bytes: 0,
        }
    }

    fn update_u64(&mut self, v: u64) {
        self.update(&v.to_le_bytes());
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_bytes = self.total_bytes.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = data.len().min(need);
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                Self::compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let block: &[u8; 64] = data[..64].try_into().expect("64-byte slice");
            Self::compress(&mut self.state, block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_bytes.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            self.buf[self.buf_len..64].fill(0);
            let block = self.buf;
            Self::compress(&mut self.state, &block);
            self.buf_len = 0;
        }
        self.buf[self.buf_len..56].fill(0);
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        Self::compress(&mut self.state, &block);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(h: &mut [u32; 8], chunk: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let mut w = [0u32; 64];
        for (i, &word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
}
