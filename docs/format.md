# DGF1 Block Format Reference

This document summarizes the Dense Geometry Format version 1 (DGF1) 128-byte block layout implemented by this crate. It is a working engineering reference derived from the pinned upstream sources listed in [`compatibility.md`](compatibility.md) and does not supersede the upstream Khronos Vulkan specification (`VK_AMDX_dense_geometry_format`, CC-BY-4.0) or AMD DGF-SDK.

## Global Limits

| Parameter | Limit / Encoding |
|---|---|
| Block size | Exactly 128 bytes (1024 bits), little-endian LSB-first bit numbering |
| Fixed header | 20 bytes (5 little-endian `u32` words, bits `0..160`) |
| Magic | `0x06` |
| Vertex count `V` | `1..=64` (stored as `V - 1` in 6 bits; a valid block requires `V >= 3` because triangle 0 introduces vertices `0, 1, 2`) |
| Triangle count `T` | `1..=64` (stored as `T - 1` in 6 bits) |
| Position anchor | Three signed 24-bit two's-complement integers in `[-8_388_608, 8_388_607]` |
| Offset widths `bx, by, bz` | `1..=16` bits each (stored as `width - 1`), with `(bx + by + bz) % 4 == 0` |
| Biased exponent `E` | `1..=232` (unbiased `e = E - 127` in `-126..=105`; scale `2^e`) |
| Geometry ID | 24 bits (`0..=16_777_215`); in constant mode 9 bits (`0..=511`) |
| Geometry/opacity palette | At most 32 distinct `(geometry_id, opaque)` pairs; prefix `0..=25` bits |
| Primitive ID base | 29 bits (`0..=536_870_911`); `prim_id_base + (T - 1)` must also fit in 29 bits |
| OMM descriptor count `D` | `0..=7` |
| User data | Optional single 4-byte little-endian `u32` word immediately after header |
| Front buffer budget | `<= 96` bytes (`768` bits), excluding header and optional user data |
| Repeated-index buffer budget | `<= 24` bytes (`192` bits) |
| Repeated-index width `B` | `3..=6` bits (stored as `B - 3` in 2 bits) |

## Fixed Header Layout (20 bytes = 5 x `u32` LE)

Bit ranges are inclusive within each little-endian 32-bit word.

| Word | Bits | Field | Semantic Interpretation |
|---|---|---|---|
| 0 | `0..=7` | `magic` | Must be `0x06` |
| 0 | `8..=9` | `bits_per_index` | `raw + 3` (`3..=6`) |
| 0 | `10..=15` | `num_vertices` | `raw + 1` (`1..=64`) |
| 0 | `16..=21` | `num_triangles` | `raw + 1` (`1..=64`) |
| 0 | `22..=31` | `geom_id_meta` | Constant mode: `(geom_id << 1) \| opaque`; Palette mode: `prefix_bits \| ((entry_count - 1) << 5)` |
| 1 | `0..=7` | `exponent` | Biased exponent `E` (`1..=232`) |
| 1 | `8..=31` | `x_anchor` | Signed 24-bit two's-complement |
| 2 | `0..=3` | `x_bits` | `raw + 1` (`1..=16`) |
| 2 | `4..=7` | `y_bits` | `raw + 1` (`1..=16`) |
| 2 | `8..=31` | `y_anchor` | Signed 24-bit two's-complement |
| 3 | `0..=3` | `z_bits` | `raw + 1` (`1..=16`) |
| 3 | `4..=6` | `omm_descriptor_count` | `0..=7` |
| 3 | `7` | `geom_id_mode` | `0` = constant, `1` = palette |
| 3 | `8..=31` | `z_anchor` | Signed 24-bit two's-complement |
| 4 | `0..=28` | `prim_id_base` | 29-bit unsigned primitive ID base |
| 4 | `29` | `have_user_data` | `1` inserts 4-byte user-data word at bytes `20..24` |
| 4 | `30..=31` | `reserved` | Must be `0` |

## Block Section Layout (1024 bits)

```text
Bit 0                                                                       Bit 1024
+--------+------+--------+-----+------+--------+--- gap ---+----------+-----------+
| Header | User | Vertex | OMM | Geom | Repeat | (unused)  | FirstUse | Controls  |
| 160 b  | 0/32 | byte-  |byte-|byte- | indices|           | (backwd) | (backwd)  |
|        |      | aligned|algnd|algnd | <=192b |           |          |           |
+--------+------+--------+-----+------+--------+-----------+----------+-----------+
                |<---- front_bits <= 768 ----->|           |<---- tail bits ----->|
```

## Exact Layout Equations

Let `align8(n) = (n + 7) & !7` and `ceil_log2(1) = 0`, `ceil_log2(n) = ceil(log2(n))` for `n >= 2`.

```text
U = 32 if have_user_data else 0

vertex_data_bits = V * (bx + by + bz)
vertex_bits      = align8(vertex_data_bits)

omm_bits = 0                                                   if D == 0
         = 32 * (2 + D) + align8(T * ceil_log2(D))             if D > 0

geom_bits = 0                                                  if geom_id_mode == 0
          = align8(L + T * ceil_log2(P) + P * (25 - L))        if geom_id_mode == 1

front_bits   = vertex_bits + omm_bits + geom_bits
repeat_start = 160 + U + front_bits

strip_index_count = 3 + sum_{i=1..T-1} (3 if control[i] == RESTART else 1)
control_bits      = 2 * (T - 1)
first_use_bits    = strip_index_count - 3
tail_start        = 1024 - control_bits - first_use_bits

repeat_bits = repeated_index_count * bits_per_index
repeat_end  = repeat_start + repeat_bits
```

### Mandatory Layout Invariants

1. `front_bits <= 768` (96 bytes)
2. `repeat_bits <= 192` (24 bytes)
3. `repeat_end <= tail_start`
4. Named section padding bits (vertex section, OMM selector section, geometry palette section) and header reserved bits must be zero.
5. Unused gap bits between `repeat_end` and `tail_start` are zero-initialized by the canonical encoder, but are not rejected if nonzero during decoding.

## Positions

For each vertex `v` in `0..V`:
- Read unsigned `x_off` (`bx` bits), `y_off` (`by` bits), `z_off` (`bz` bits) sequentially from `vertex_start`.
- `q_axis = anchor_axis + (off_axis as i32)`
- `pos_axis = (q_axis as f32) * f32::from_bits((E as u32) << 23)`
- Every decoded coordinate must be finite (`pos_axis.is_finite()`). At `E == 232`, `q_axis >= 2^23` overflows `f32` to `+inf` and is rejected.

## Topology Streams

### Triangle Control Stream (backward from bit 1024)
- `control[0]` is implicitly `RESTART (0)`.
- For `i` in `1..T`, read 2 bits at absolute bit offset `1024 - 2 * i`:
  - `0` = `RESTART` (consumes 3 strip indices)
  - `1` = `EDGE1` (consumes 1 strip index)
  - `2` = `EDGE2` (consumes 1 strip index)
  - `3` = `BACKTRACK` (consumes 1 strip index; legal only when `i >= 2` and `control[i - 1]` is `EDGE1` or `EDGE2`)

### First-Use Flags and Repeated Indices
- Strip indices `0, 1, 2` are implicitly `0, 1, 2` (`next_new_vertex = 3`).
- For strip position `j` in `3..strip_index_count`, read 1 bit at `1024 - 2*(T - 1) - 1 - (j - 3)`:
  - `1`: emit `next_new_vertex` and increment `next_new_vertex` (must be `< V`).
  - `0`: read `bits_per_index` bits from forward repeated-index stream starting at `repeat_start`; value must be `< next_new_vertex` and `< V`.

### Strip-to-Triangle Expansion
Let `prev = [a, b, c]`, `prev_prev = [d, e, f]`, and newly consumed index `n`:
- `RESTART`: next 3 strip indices `[n0, n1, n2]`
- `EDGE1`: `[c, b, n]`
- `EDGE2`: `[a, c, n]`
- `BACKTRACK` after `EDGE1`: `[d, f, n]`
- `BACKTRACK` after `EDGE2`: `[f, e, n]`

## Geometry ID / Opacity Palette

- Constant mode (`geom_id_mode == 0`): `geometry_id = geom_id_meta >> 1` (`0..=511`), `opaque = (geom_id_meta & 1) != 0`.
- Palette mode (`geom_id_mode == 1`):
  - `prefix_bits = geom_id_meta & 31` (`<= 25`)
  - `entry_count = (geom_id_meta >> 5) + 1` (`1..=32`)
  - `payload_bits = 25 - prefix_bits`
  - `selector_bits = ceil_log2(entry_count)`
  - Layout: `prefix` (`prefix_bits`), `T` selectors (`selector_bits` each, `< entry_count`), `entry_count` payloads (`payload_bits` each), zero pad to byte boundary.
  - Entry value: `val = (prefix << payload_bits) | payload[selector]`; `geometry_id = val >> 1`, `opaque = (val & 1) != 0`.

## OMM Palette

Present when `D = omm_descriptor_count > 0` (`1..=7`):
1. Hot-patch region: `4 * (2 + D)` bytes (`8 + 4*D` bytes). Zero-initialized by the encoder for new assets; preserved verbatim as opaque bytes by the decoder.
2. Pre-computed selector region: `T` per-triangle descriptor indices of `ceil_log2(D)` bits each (zero bits when `D == 1`), each `< D`, followed by zero pad bits to byte boundary.
