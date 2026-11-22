//! Independent minimal per-bit Rust DGF1 reference interpreter and pure-Rust SHA-256.
//!
//! Deliberately shares NO layout, header, bit-extraction, topology, or palette code
//! with the production `dgf` crate.

#![forbid(unsafe_code)]
#![allow(dead_code)]

/// Independently decoded representation of a 128-byte DGF1 block.
#[derive(Debug, Clone, PartialEq)]
pub struct RefDecodedBlock {
    pub bits_per_index: u8,
    pub num_vertices: u8,
    pub num_triangles: u8,
    pub biased_exponent: u8,
    pub anchor: [i32; 3],
    pub offset_bits: [u8; 3],
    pub omm_descriptor_count: u8,
    pub geom_id_mode: u8,
    pub prim_id_base: u32,
    pub user_data: Option<u32>,
    pub quantized_positions: Vec<[i32; 3]>,
    pub positions: Vec<[f32; 3]>,
    pub controls: Vec<u8>,
    pub strip_indices: Vec<u8>,
    pub triangles: Vec<[u8; 3]>,
    pub geometry_ids: Vec<u32>,
    pub opaque: Vec<bool>,
    pub omm_hot_patch: Vec<u8>,
    pub omm_selectors: Vec<u8>,
}

/// Slow, dead-simple per-bit reader over a 128-byte block.
pub fn ref_read_bits(block: &[u8; 128], start: usize, width: usize) -> Result<u64, String> {
    if width > 64 || start.saturating_add(width) > 1024 {
        return Err(format!(
            "bit read out of range: start={start} width={width}"
        ));
    }
    let mut val = 0u64;
    for i in 0..width {
        let p = start + i;
        let b = (block[p / 8] >> (p % 8)) & 1;
        val |= u64::from(b) << i;
    }
    Ok(val)
}

/// Slow, dead-simple per-bit writer for hand-assembling independent binary fixtures.
pub fn ref_write_bits(block: &mut [u8; 128], start: usize, width: usize, value: u64) {
    assert!(width <= 64 && start + width <= 1024);
    if width < 64 {
        assert_eq!(
            value >> width,
            0,
            "value 0x{value:x} overflows width {width}"
        );
    }
    for i in 0..width {
        let p = start + i;
        let bit = ((value >> i) & 1) as u8;
        if bit != 0 {
            block[p / 8] |= 1 << (p % 8);
        } else {
            block[p / 8] &= !(1 << (p % 8));
        }
    }
}

fn ref_align8(bits: usize) -> usize {
    (bits + 7) & !7
}

fn ref_ceil_log2(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut k = 0usize;
    let mut p = 1usize;
    while p < n {
        p <<= 1;
        k += 1;
    }
    k
}

fn ref_check_zero_bits(block: &[u8; 128], start: usize, width: usize) -> Result<(), String> {
    for p in start..(start + width) {
        if ((block[p / 8] >> (p % 8)) & 1) != 0 {
            return Err(format!("nonzero padding bit at {p}"));
        }
    }
    Ok(())
}

/// Completely independent per-bit DGF1 reference decoder following `docs/format.md`.
pub fn decode_block_independent(
    block: &[u8; 128],
    require_zero_omm_hot_patch: bool,
) -> Result<RefDecodedBlock, String> {
    // 1. Header words read bit-by-bit
    let magic = ref_read_bits(block, 0, 8)? as u8;
    if magic != 0x06 {
        return Err(format!("bad magic 0x{magic:02x}"));
    }
    let bits_per_index = (ref_read_bits(block, 8, 2)? as u8) + 3;
    let num_vertices = (ref_read_bits(block, 10, 6)? as u8) + 1;
    let num_triangles = (ref_read_bits(block, 16, 6)? as u8) + 1;
    let geom_id_meta = ref_read_bits(block, 22, 10)? as u16;

    let biased_exponent = ref_read_bits(block, 32, 8)? as u8;
    if biased_exponent == 0 || biased_exponent > 232 {
        return Err(format!("bad exponent {biased_exponent}"));
    }
    let raw_x_anc = ref_read_bits(block, 40, 24)? as u32;
    let x_bits = (ref_read_bits(block, 64, 4)? as u8) + 1;
    let y_bits = (ref_read_bits(block, 68, 4)? as u8) + 1;
    let raw_y_anc = ref_read_bits(block, 72, 24)? as u32;
    let z_bits = (ref_read_bits(block, 96, 4)? as u8) + 1;
    let omm_descriptor_count = ref_read_bits(block, 100, 3)? as u8;
    let geom_id_mode = ref_read_bits(block, 103, 1)? as u8;
    let raw_z_anc = ref_read_bits(block, 104, 24)? as u32;

    let prim_id_base = ref_read_bits(block, 128, 29)? as u32;
    let have_user_data = ref_read_bits(block, 157, 1)? != 0;
    let reserved = ref_read_bits(block, 158, 2)? as u8;
    if reserved != 0 {
        return Err(format!("reserved nonzero: {reserved}"));
    }

    if !(x_bits + y_bits + z_bits).is_multiple_of(4) {
        return Err("vertex width sum not multiple of 4".into());
    }
    if u64::from(prim_id_base) + u64::from(num_triangles - 1) >= (1u64 << 29) {
        return Err("primitive ID overflow".into());
    }

    let sign_ext = |raw: u32| -> i32 {
        if (raw & 0x80_0000) != 0 {
            (raw | 0xFF00_0000) as i32
        } else {
            raw as i32
        }
    };
    let anchor = [
        sign_ext(raw_x_anc),
        sign_ext(raw_y_anc),
        sign_ext(raw_z_anc),
    ];

    // 2. Optional user data
    let mut cursor = 160usize;
    let user_data = if have_user_data {
        let ud = ref_read_bits(block, cursor, 32)? as u32;
        cursor += 32;
        Some(ud)
    } else {
        None
    };

    let front_start = cursor;

    // 3. Vertex offsets
    let bx = usize::from(x_bits);
    let by = usize::from(y_bits);
    let bz = usize::from(z_bits);
    let v_count = usize::from(num_vertices);
    let t_count = usize::from(num_triangles);
    let v_data_bits = v_count * (bx + by + bz);
    let v_aligned_bits = ref_align8(v_data_bits);
    if v_aligned_bits > 768 || cursor + v_aligned_bits > 1024 {
        return Err("vertex section exceeds block/front buffer".into());
    }

    let scale = f32::from_bits(u32::from(biased_exponent) << 23);
    let mut quantized_positions = Vec::with_capacity(v_count);
    let mut positions = Vec::with_capacity(v_count);
    for v in 0..v_count {
        let base = cursor + v * (bx + by + bz);
        let ox = ref_read_bits(block, base, bx)? as i32;
        let oy = ref_read_bits(block, base + bx, by)? as i32;
        let oz = ref_read_bits(block, base + bx + by, bz)? as i32;
        let q = [anchor[0] + ox, anchor[1] + oy, anchor[2] + oz];
        let p = [
            q[0] as f32 * scale,
            q[1] as f32 * scale,
            q[2] as f32 * scale,
        ];
        if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
            return Err(format!("non-finite position at v={v}"));
        }
        quantized_positions.push(q);
        positions.push(p);
    }
    ref_check_zero_bits(block, cursor + v_data_bits, v_aligned_bits - v_data_bits)?;
    cursor += v_aligned_bits;

    // 4. Optional OMM palette
    let mut omm_hot_patch = Vec::new();
    let mut omm_selectors = Vec::new();
    let d_count = usize::from(omm_descriptor_count);
    if d_count > 0 {
        let hp_bytes = 4 * (2 + d_count);
        let hp_start_byte = cursor / 8;
        assert_eq!(cursor % 8, 0);
        if hp_start_byte + hp_bytes > 128 {
            return Err("OMM hot-patch exceeds block bounds".into());
        }
        omm_hot_patch.extend_from_slice(&block[hp_start_byte..hp_start_byte + hp_bytes]);
        if require_zero_omm_hot_patch && omm_hot_patch.iter().any(|&b| b != 0) {
            return Err("nonzero OMM hot-patch byte in asset mode".into());
        }
        cursor += hp_bytes * 8;

        let sel_w = ref_ceil_log2(d_count);
        let sel_data = t_count * sel_w;
        let sel_aligned = ref_align8(sel_data);
        if cursor + sel_aligned > 1024 {
            return Err("OMM selector section exceeds block bounds".into());
        }
        for t in 0..t_count {
            let sel = ref_read_bits(block, cursor + t * sel_w, sel_w)? as u8;
            if usize::from(sel) >= d_count {
                return Err(format!("OMM selector {sel} >= {d_count}"));
            }
            omm_selectors.push(sel);
        }
        ref_check_zero_bits(block, cursor + sel_data, sel_aligned - sel_data)?;
        cursor += sel_aligned;
    }

    // 5. Optional Geometry ID palette
    let mut geometry_ids = Vec::with_capacity(t_count);
    let mut opaque = Vec::with_capacity(t_count);
    if geom_id_mode == 0 {
        let gid = u32::from(geom_id_meta >> 1);
        let op = (geom_id_meta & 1) != 0;
        geometry_ids.resize(t_count, gid);
        opaque.resize(t_count, op);
    } else {
        let prefix_bits = usize::from(geom_id_meta & 31);
        if prefix_bits > 25 {
            return Err(format!("geom prefix_bits {prefix_bits} > 25"));
        }
        let entry_count = usize::from(geom_id_meta >> 5) + 1;
        let payload_bits = 25 - prefix_bits;
        let sel_bits = ref_ceil_log2(entry_count);

        let geom_data_bits = prefix_bits + t_count * sel_bits + entry_count * payload_bits;
        let geom_aligned_bits = ref_align8(geom_data_bits);
        if cursor + geom_aligned_bits > 1024 {
            return Err("geometry palette exceeds block bounds".into());
        }

        let prefix = ref_read_bits(block, cursor, prefix_bits)? as u32;
        let sel_start = cursor + prefix_bits;
        let payload_start = sel_start + t_count * sel_bits;

        let mut palette = Vec::with_capacity(entry_count);
        for e in 0..entry_count {
            let payload =
                ref_read_bits(block, payload_start + e * payload_bits, payload_bits)? as u32;
            let combined = if payload_bits == 0 {
                prefix
            } else {
                (prefix << payload_bits) | payload
            };
            palette.push(combined);
        }
        for t in 0..t_count {
            let sel = ref_read_bits(block, sel_start + t * sel_bits, sel_bits)? as usize;
            if sel >= entry_count {
                return Err(format!("geom selector {sel} >= {entry_count}"));
            }
            let combined = palette[sel];
            geometry_ids.push(combined >> 1);
            opaque.push((combined & 1) != 0);
        }
        ref_check_zero_bits(
            block,
            cursor + geom_data_bits,
            geom_aligned_bits - geom_data_bits,
        )?;
        cursor += geom_aligned_bits;
    }

    let front_bits = cursor - front_start;
    if front_bits > 768 {
        return Err(format!("front_bits {front_bits} > 768"));
    }
    let repeat_start = cursor;

    // 6. Backward triangle controls
    let mut controls = vec![0u8; t_count];
    let mut strip_index_count = 3usize;
    for i in 1..t_count {
        let c = ref_read_bits(block, 1024 - 2 * i, 2)? as u8;
        if c == 3 && (i < 2 || (controls[i - 1] != 1 && controls[i - 1] != 2)) {
            return Err(format!("illegal backtrack at tri {i}"));
        }
        controls[i] = c;
        strip_index_count += if c == 0 { 3 } else { 1 };
    }

    // 7. Backward first-use flags
    let control_bits = 2 * (t_count - 1);
    let first_bits = strip_index_count - 3;
    let tail_start = 1024usize
        .checked_sub(control_bits + first_bits)
        .ok_or("tail underflow")?;

    let mut is_first = vec![true; strip_index_count];
    let mut repeated_count = 0usize;
    for (j, slot) in is_first
        .iter_mut()
        .enumerate()
        .take(strip_index_count)
        .skip(3)
    {
        let f = ref_read_bits(block, 1024 - control_bits - 1 - (j - 3), 1)? != 0;
        *slot = f;
        if !f {
            repeated_count += 1;
        }
    }

    let repeat_bits = repeated_count * usize::from(bits_per_index);
    if repeat_bits > 192 {
        return Err(format!("repeat_bits {repeat_bits} > 192"));
    }
    let repeat_end = repeat_start + repeat_bits;
    if repeat_end > tail_start {
        return Err(format!(
            "overlap: repeat_end={repeat_end} > tail_start={tail_start}"
        ));
    }

    // 8. Strip indices
    if num_vertices < 3 {
        return Err(format!("num_vertices {num_vertices} < 3"));
    }
    let mut strip_indices = vec![0u8; strip_index_count];
    strip_indices[0] = 0;
    strip_indices[1] = 1;
    strip_indices[2] = 2;
    let mut next_new = 3u8;
    let mut rep_cursor = repeat_start;
    for j in 3..strip_index_count {
        if is_first[j] {
            if next_new >= num_vertices {
                return Err(format!("new vertex {next_new} >= {num_vertices}"));
            }
            strip_indices[j] = next_new;
            next_new += 1;
        } else {
            let idx = ref_read_bits(block, rep_cursor, usize::from(bits_per_index))? as u8;
            rep_cursor += usize::from(bits_per_index);
            if idx >= num_vertices || idx >= next_new {
                return Err(format!(
                    "bad repeated index {idx} (next_new={next_new}, V={num_vertices})"
                ));
            }
            strip_indices[j] = idx;
        }
    }

    // 9. Strip-to-triangle expansion
    let mut triangles = Vec::with_capacity(t_count);
    triangles.push([strip_indices[0], strip_indices[1], strip_indices[2]]);
    let mut pos = 3usize;
    for i in 1..t_count {
        let prev = triangles[i - 1];
        let tri = match controls[i] {
            0 => {
                let t = [
                    strip_indices[pos],
                    strip_indices[pos + 1],
                    strip_indices[pos + 2],
                ];
                pos += 3;
                t
            }
            1 => {
                let n = strip_indices[pos];
                pos += 1;
                [prev[2], prev[1], n]
            }
            2 => {
                let n = strip_indices[pos];
                pos += 1;
                [prev[0], prev[2], n]
            }
            3 => {
                let n = strip_indices[pos];
                pos += 1;
                let pp = triangles[i - 2];
                if controls[i - 1] == 1 {
                    [pp[0], pp[2], n]
                } else {
                    [pp[2], pp[1], n]
                }
            }
            _ => unreachable!(),
        };
        triangles.push(tri);
    }

    Ok(RefDecodedBlock {
        bits_per_index,
        num_vertices,
        num_triangles,
        biased_exponent,
        anchor,
        offset_bits: [x_bits, y_bits, z_bits],
        omm_descriptor_count,
        geom_id_mode,
        prim_id_base,
        user_data,
        quantized_positions,
        positions,
        controls,
        strip_indices,
        triangles,
        geometry_ids,
        opaque,
        omm_hot_patch,
        omm_selectors,
    })
}

/// Pure-Rust SHA-256 (FIPS 180-4) for zero-dependency fixture digest verification.
pub fn sha256(data: &[u8]) -> [u8; 32] {
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

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.as_chunks::<64>().0 {
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
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
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

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

pub fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256(data);
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

/// Asserts that the production decoder and the independent reference interpreter agree completely.
pub fn assert_matches_reference(block: &[u8; 128], mode: dgf::ValidationMode) {
    let require_zero_hp = mode == dgf::ValidationMode::Asset;
    let prod = dgf::validate_block_with_mode(block, mode).and_then(|v| dgf::decode_block(&v));
    let ref_res = decode_block_independent(block, require_zero_hp);

    match (prod, ref_res) {
        (Ok(p), Ok(r)) => {
            assert_eq!(p.header.bits_per_index, r.bits_per_index);
            assert_eq!(p.num_vertices, r.num_vertices);
            assert_eq!(p.num_triangles, r.num_triangles);
            assert_eq!(p.exponent.raw(), r.biased_exponent);
            assert_eq!(p.anchor, r.anchor);
            assert_eq!(p.header.offset_bits, r.offset_bits);
            assert_eq!(p.omm.descriptor_count, r.omm_descriptor_count);
            assert_eq!(p.prim_id_base, r.prim_id_base);
            assert_eq!(p.user_data, r.user_data);
            assert_eq!(p.quantized_positions(), r.quantized_positions.as_slice());
            for (pp, rp) in p.positions().iter().zip(r.positions.iter()) {
                assert_eq!(pp[0].to_bits(), rp[0].to_bits());
                assert_eq!(pp[1].to_bits(), rp[1].to_bits());
                assert_eq!(pp[2].to_bits(), rp[2].to_bits());
            }
            assert_eq!(p.triangles(), r.triangles.as_slice());
            assert_eq!(p.geometry_ids(), r.geometry_ids.as_slice());
            assert_eq!(p.opaque_flags(), r.opaque.as_slice());
            assert_eq!(p.omm.hot_patch(), r.omm_hot_patch.as_slice());
            assert_eq!(p.omm_triangle_descriptors(), r.omm_selectors.as_slice());
        }
        (Err(_), Err(_)) => {}
        (Ok(p), Err(re)) => {
            panic!("Production accepted block that reference rejected: ref_err={re}, prod={p:?}");
        }
        (Err(pe), Ok(r)) => {
            panic!("Production rejected block that reference accepted: prod_err={pe}, ref={r:?}");
        }
    }
}
