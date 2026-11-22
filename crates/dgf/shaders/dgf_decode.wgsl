// Pure WGSL scalar Dense Geometry Format (DGF1) block decoder.
//
// Spec-faithful 32-bit implementation (no 64-bit integers, no shifts >= 32):
// - Fixed 128-byte DGF1 blocks (32 little-endian u32 words)
// - Signed 24-bit integer anchors, 1..=16-bit vertex offsets, biased exponents 1..=232
// - Optional 4-byte user-data word
// - Optional OMM palette (0..=7 descriptors, hot-patch offset + per-triangle selectors)
// - Constant (9-bit) and Palette (24-bit + opaque, 0..=25 prefix bits, 1..=32 entries) geometry IDs
// - Backward 2-bit triangle controls (RESTART, EDGE1, EDGE2, BACKTRACK)
// - Backward 1-bit first-use flags (up to 192 strip indices)
// - Forward repeated-index buffer (3..=6 bits per index)
// - 29-bit per-triangle primitive IDs (prim_id_base + local_triangle_index)

const DGF_MAGIC: u32 = 0x06u;
const DGF_BLOCK_BITS: u32 = 1024u;
const DGF_HEADER_BITS: u32 = 160u;

const DGF_CTRL_RESTART: u32 = 0u;
const DGF_CTRL_EDGE1: u32 = 1u;
const DGF_CTRL_EDGE2: u32 = 2u;
const DGF_CTRL_BACKTRACK: u32 = 3u;

const DGF_TRI_FLAG_OPAQUE: u32 = 1u;
const DGF_TRI_FLAG_HAVE_OMM: u32 = 16u;
const DGF_BLOCK_FLAG_HAVE_USER_DATA: u32 = 1u;

struct DgfBlock {
    words: array<u32, 32>,
}

struct DgfBlockDispatch {
    vertex_offset: u32,
    triangle_offset: u32,
    add_vertex_offset: u32,
    _pad: u32,
}

struct DgfDecodedVertex {
    px: f32,
    py: f32,
    pz: f32,
    _pad: u32,
}

struct DgfDecodedTriangle {
    i0: u32,
    i1: u32,
    i2: u32,
    geometry_id: u32,
    prim_id: u32,
    // bit 0: opaque, bits 1..=3: omm_descriptor (0..7), bit 4: have_omm
    flags: u32,
}

struct DgfBlockInfo {
    num_vertices: u32,
    num_triangles: u32,
    user_data: u32,
    // bit 0: have_user_data, bits 1..=3: omm_descriptor_count,
    // bits 4..=7: bits_per_index, bits 8..=15: exponent
    flags: u32,
}

struct DgfHeader {
    bits_per_index: u32,
    num_vertices: u32,
    num_triangles: u32,
    geom_id_meta: u32,
    geom_id_mode: u32,
    exponent: u32,
    anchor: vec3<i32>,
    offset_bits: vec3<u32>,
    omm_descriptor_count: u32,
    prim_id_base: u32,
    have_user_data: u32,
    user_data: u32,
}

struct DgfFrontLayout {
    vertex_start: u32,
    vertex_stride: u32,
    omm_start: u32,
    omm_selector_start: u32,
    omm_sel_w: u32,
    geom_start: u32,
    geom_prefix_bits: u32,
    geom_entry_count: u32,
    geom_sel_w: u32,
    geom_payload_w: u32,
    repeat_start: u32,
}

fn dgf_align8(bits: u32) -> u32 {
    return (bits + 7u) & (~7u);
}

fn dgf_ceil_log2(n: u32) -> u32 {
    if (n <= 1u) {
        return 0u;
    }
    return 32u - countLeadingZeros(n - 1u);
}

fn dgf_read_bits(blk: ptr<function, DgfBlock>, bit_pos: u32, width: u32) -> u32 {
    if (width == 0u) {
        return 0u;
    }
    let word_idx = bit_pos >> 5u;
    let bit_off = bit_pos & 31u;
    var val = (*blk).words[word_idx] >> bit_off;
    if (bit_off + width > 32u) {
        val = val | ((*blk).words[word_idx + 1u] << (32u - bit_off));
    }
    if (width >= 32u) {
        return val;
    }
    return val & ((1u << width) - 1u);
}

fn dgf_parse_header(blk: ptr<function, DgfBlock>) -> DgfHeader {
    let w0 = (*blk).words[0];
    let w1 = (*blk).words[1];
    let w2 = (*blk).words[2];
    let w3 = (*blk).words[3];
    let w4 = (*blk).words[4];

    let have_user_data = (w4 >> 29u) & 0x1u;
    var user_data = 0u;
    if (have_user_data != 0u) {
        user_data = (*blk).words[5];
    }

    var hdr: DgfHeader;
    hdr.bits_per_index = ((w0 >> 8u) & 0x3u) + 3u;
    hdr.num_vertices = ((w0 >> 10u) & 0x3Fu) + 1u;
    hdr.num_triangles = ((w0 >> 16u) & 0x3Fu) + 1u;
    hdr.geom_id_meta = (w0 >> 22u) & 0x3FFu;
    hdr.geom_id_mode = (w3 >> 7u) & 0x1u;
    hdr.exponent = w1 & 0xFFu;
    hdr.anchor = vec3<i32>(
        bitcast<i32>(w1 & 0xFFFFFF00u) >> 8u,
        bitcast<i32>(w2 & 0xFFFFFF00u) >> 8u,
        bitcast<i32>(w3 & 0xFFFFFF00u) >> 8u,
    );
    hdr.offset_bits = vec3<u32>(
        (w2 & 0xFu) + 1u,
        ((w2 >> 4u) & 0xFu) + 1u,
        (w3 & 0xFu) + 1u,
    );
    hdr.omm_descriptor_count = (w3 >> 4u) & 0x7u;
    hdr.prim_id_base = w4 & 0x1FFFFFFFu;
    hdr.have_user_data = have_user_data;
    hdr.user_data = user_data;
    return hdr;
}

fn dgf_compute_front_layout(hdr: DgfHeader) -> DgfFrontLayout {
    var front: DgfFrontLayout;
    let user_data_bits = select(0u, 32u, hdr.have_user_data != 0u);
    front.vertex_start = DGF_HEADER_BITS + user_data_bits;
    front.vertex_stride = hdr.offset_bits.x + hdr.offset_bits.y + hdr.offset_bits.z;
    let vertex_bits = dgf_align8(hdr.num_vertices * front.vertex_stride);

    front.omm_start = front.vertex_start + vertex_bits;
    var omm_bits = 0u;
    if (hdr.omm_descriptor_count == 0u) {
        front.omm_selector_start = front.omm_start;
        front.omm_sel_w = 0u;
    } else {
        let hot_patch_bits = 32u * (2u + hdr.omm_descriptor_count);
        front.omm_selector_start = front.omm_start + hot_patch_bits;
        front.omm_sel_w = dgf_ceil_log2(hdr.omm_descriptor_count);
        omm_bits = hot_patch_bits + dgf_align8(hdr.num_triangles * front.omm_sel_w);
    }

    front.geom_start = front.omm_start + omm_bits;
    var geom_bits = 0u;
    if (hdr.geom_id_mode == 0u) {
        front.geom_prefix_bits = 0u;
        front.geom_entry_count = 0u;
        front.geom_sel_w = 0u;
        front.geom_payload_w = 0u;
    } else {
        front.geom_prefix_bits = hdr.geom_id_meta & 31u;
        front.geom_entry_count = (hdr.geom_id_meta >> 5u) + 1u;
        front.geom_sel_w = dgf_ceil_log2(front.geom_entry_count);
        front.geom_payload_w = 25u - front.geom_prefix_bits;
        let geom_data_bits = front.geom_prefix_bits
            + hdr.num_triangles * front.geom_sel_w
            + front.geom_entry_count * front.geom_payload_w;
        geom_bits = dgf_align8(geom_data_bits);
    }

    front.repeat_start = front.geom_start + geom_bits;
    return front;
}

fn dgf_decode_vertex(
    blk: ptr<function, DgfBlock>,
    hdr: DgfHeader,
    front: DgfFrontLayout,
    vert_idx: u32,
) -> vec3<f32> {
    let base = front.vertex_start + vert_idx * front.vertex_stride;
    let ox = dgf_read_bits(blk, base, hdr.offset_bits.x);
    let oy = dgf_read_bits(blk, base + hdr.offset_bits.x, hdr.offset_bits.y);
    let oz = dgf_read_bits(blk, base + hdr.offset_bits.x + hdr.offset_bits.y, hdr.offset_bits.z);

    let qx = hdr.anchor.x + i32(ox);
    let qy = hdr.anchor.y + i32(oy);
    let qz = hdr.anchor.z + i32(oz);
    let scale = bitcast<f32>(hdr.exponent << 23u);
    return vec3<f32>(f32(qx) * scale, f32(qy) * scale, f32(qz) * scale);
}

fn dgf_decode_geom_entry(
    blk: ptr<function, DgfBlock>,
    hdr: DgfHeader,
    front: DgfFrontLayout,
    tri_idx: u32,
) -> vec2<u32> {
    if (hdr.geom_id_mode == 0u) {
        return vec2<u32>(hdr.geom_id_meta >> 1u, hdr.geom_id_meta & 1u);
    }
    let prefix = dgf_read_bits(blk, front.geom_start, front.geom_prefix_bits);
    let selectors_start = front.geom_start + front.geom_prefix_bits;
    let payloads_start = selectors_start + hdr.num_triangles * front.geom_sel_w;
    let sel = dgf_read_bits(blk, selectors_start + tri_idx * front.geom_sel_w, front.geom_sel_w);
    let payload = dgf_read_bits(
        blk,
        payloads_start + sel * front.geom_payload_w,
        front.geom_payload_w,
    );
    let combined = (prefix << front.geom_payload_w) | payload;
    return vec2<u32>(combined >> 1u, combined & 1u);
}

fn dgf_decode_omm_selector(
    blk: ptr<function, DgfBlock>,
    hdr: DgfHeader,
    front: DgfFrontLayout,
    tri_idx: u32,
) -> u32 {
    if (hdr.omm_descriptor_count == 0u) {
        return 0u;
    }
    return dgf_read_bits(
        blk,
        front.omm_selector_start + tri_idx * front.omm_sel_w,
        front.omm_sel_w,
    );
}

fn dgf_next_strip_index(
    blk: ptr<function, DgfBlock>,
    bits_per_index: u32,
    first_use_base: u32,
    strip_pos: ptr<function, u32>,
    next_new: ptr<function, u32>,
    repeat_bit: ptr<function, u32>,
) -> u32 {
    let j = *strip_pos;
    *strip_pos = j + 1u;
    let is_first = dgf_read_bits(blk, first_use_base - (j - 3u), 1u);
    if (is_first != 0u) {
        let v = *next_new;
        *next_new = v + 1u;
        return v;
    }
    let v = dgf_read_bits(blk, *repeat_bit, bits_per_index);
    *repeat_bit = *repeat_bit + bits_per_index;
    return v;
}

@group(0) @binding(0) var<storage, read> dgf_blocks: array<DgfBlock>;
@group(0) @binding(1) var<storage, read> dgf_dispatch: array<DgfBlockDispatch>;
@group(0) @binding(2) var<storage, read_write> out_vertices: array<DgfDecodedVertex>;
@group(0) @binding(3) var<storage, read_write> out_triangles: array<DgfDecodedTriangle>;
@group(0) @binding(4) var<storage, read_write> out_block_info: array<DgfBlockInfo>;

@compute @workgroup_size(64, 1, 1)
fn decode_blocks(@builtin(global_invocation_id) gid: vec3<u32>) {
    let block_idx = gid.x;
    if (block_idx >= arrayLength(&dgf_blocks)) {
        return;
    }

    var blk = dgf_blocks[block_idx];
    let disp = dgf_dispatch[block_idx];
    let hdr = dgf_parse_header(&blk);
    let front = dgf_compute_front_layout(hdr);

    let block_flags = (hdr.have_user_data & 1u)
        | ((hdr.omm_descriptor_count & 7u) << 1u)
        | ((hdr.bits_per_index & 15u) << 4u)
        | ((hdr.exponent & 255u) << 8u);
    out_block_info[block_idx] = DgfBlockInfo(
        hdr.num_vertices,
        hdr.num_triangles,
        hdr.user_data,
        block_flags,
    );

    // 1. Decode vertex positions.
    for (var v = 0u; v < hdr.num_vertices; v = v + 1u) {
        let p = dgf_decode_vertex(&blk, hdr, front, v);
        out_vertices[disp.vertex_offset + v] = DgfDecodedVertex(p.x, p.y, p.z, 0u);
    }

    // 2. Decode topology, geometry IDs, opacity, primitive IDs, and OMM selectors.
    let vbase = select(0u, disp.vertex_offset, disp.add_vertex_offset != 0u);
    let have_omm_flag = select(0u, DGF_TRI_FLAG_HAVE_OMM, hdr.omm_descriptor_count > 0u);
    let control_bits = 2u * (hdr.num_triangles - 1u);
    let first_use_base = DGF_BLOCK_BITS - control_bits - 1u;

    var strip_pos = 3u;
    var next_new = 3u;
    var repeat_bit = front.repeat_start;

    var prev_tri = vec3<u32>(0u, 1u, 2u);
    var prev_prev_tri = prev_tri;
    var prev_ctrl = DGF_CTRL_RESTART;

    for (var i = 0u; i < hdr.num_triangles; i = i + 1u) {
        var tri = vec3<u32>(0u, 1u, 2u);
        var ctrl = DGF_CTRL_RESTART;

        if (i > 0u) {
            ctrl = dgf_read_bits(&blk, DGF_BLOCK_BITS - 2u * i, 2u);
            if (ctrl == DGF_CTRL_RESTART) {
                let n0 = dgf_next_strip_index(
                    &blk, hdr.bits_per_index, first_use_base, &strip_pos, &next_new, &repeat_bit,
                );
                let n1 = dgf_next_strip_index(
                    &blk, hdr.bits_per_index, first_use_base, &strip_pos, &next_new, &repeat_bit,
                );
                let n2 = dgf_next_strip_index(
                    &blk, hdr.bits_per_index, first_use_base, &strip_pos, &next_new, &repeat_bit,
                );
                tri = vec3<u32>(n0, n1, n2);
            } else {
                let n = dgf_next_strip_index(
                    &blk, hdr.bits_per_index, first_use_base, &strip_pos, &next_new, &repeat_bit,
                );
                if (ctrl == DGF_CTRL_EDGE1) {
                    tri = vec3<u32>(prev_tri.z, prev_tri.y, n);
                } else if (ctrl == DGF_CTRL_EDGE2) {
                    tri = vec3<u32>(prev_tri.x, prev_tri.z, n);
                } else {
                    // DGF_CTRL_BACKTRACK
                    if (prev_ctrl == DGF_CTRL_EDGE1) {
                        tri = vec3<u32>(prev_prev_tri.x, prev_prev_tri.z, n);
                    } else {
                        tri = vec3<u32>(prev_prev_tri.z, prev_prev_tri.y, n);
                    }
                }
            }
            prev_prev_tri = prev_tri;
            prev_tri = tri;
            prev_ctrl = ctrl;
        }

        let geom = dgf_decode_geom_entry(&blk, hdr, front, i);
        let omm_sel = dgf_decode_omm_selector(&blk, hdr, front, i);
        let tri_flags = (geom.y & DGF_TRI_FLAG_OPAQUE) | ((omm_sel & 7u) << 1u) | have_omm_flag;

        out_triangles[disp.triangle_offset + i] = DgfDecodedTriangle(
            tri.x + vbase,
            tri.y + vbase,
            tri.z + vbase,
            geom.x,
            hdr.prim_id_base + i,
            tri_flags,
        );
    }
}
