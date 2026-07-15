# 03 — The `NGPT` model blob format

One file format carries every model this project will ever ship — from
M1's fake canned-text "model" to the final quantized GRU. The loader in
`core/ngpt.cpp` is the single entry point; the `model type` field decides
what the payload means.

## Layout (format version 1)

All multi-byte integers are **big-endian** ("network order").

| offset | size | field | notes |
|---|---|---|---|
| 0 | 4 | magic | ASCII `NGPT` (`0x4E475054`) |
| 4 | 2 | format version | currently `1`; bump on breaking layout change |
| 6 | 2 | model type | `0` = canned text, `1` = GRU (M2+) |
| 8 | 4 | payload length | bytes following the header |
| 12 | n | payload | model-type-specific |

**Model type 0 (canned text):** payload is the raw ASCII bytes of the
line to "generate". The engine replays them one per `ngpt_step`, then
returns `NGPT_EOS`.

**Model type 1 (GRU):** payload layout is specified at M2 — dims, vocab
table, quantization scales, sigmoid/tanh LUTs, then weights. Same
header, same loader, new branch.

Example — the M1 blob, all 33 bytes (`od -A d -t x1z tests/vectors/m1_canned.bin`):

```
0000000 4e 47 50 54 00 01 00 00 00 00 00 15 48 41 4c 54  >NGPT........HALT<
0000016 21 20 57 48 4f 20 47 4f 45 53 20 54 48 45 52 45  >! WHO GOES THERE<
0000032 3f                                               >?<
```

## Why big-endian, parsed byte-by-byte

The N64 is big-endian, development machines are little-endian. If either
side ever did `*(uint32_t*)p`, the same file would parse differently on
each. Instead the parser only ever reads single bytes and combines them
arithmetically:

```cpp
uint32_t ngpt_read_u32be(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}
```

Shifts and ORs are defined by the C++ standard, not by the CPU's byte
order, so this returns the same value everywhere. Choosing big-endian
for the wire format is then just a convention (it matches the target and
is the traditional network order). This one discipline is what makes
"host tests prove console behavior" true — see `docs/00-n64-primer.md`.

Alignment bonus: because nothing is ever cast to a wider pointer, the
payload needs no alignment guarantees, and the MIPS CPU (which faults on
misaligned loads) can't be tripped up by a weight table at an odd offset.

## Versioning rules

- **format version** guards the header/layout itself. Readers reject
  versions they don't know (`NGPT_ERR_VERSION`) rather than misparse.
- **model type** selects the payload interpretation. Unknown types are
  rejected (`NGPT_ERR_MODEL_TYPE`), so an M2 GRU blob fed to an M1 ROM
  fails loudly, not weirdly.
- Every field is bounds-checked against the actual buffer length before
  use (`NGPT_ERR_TRUNCATED`), including the overflow case where a huge
  payload length would wrap — see `tests/test_blob_parser.cpp` for the
  full rejection matrix.

## Producers and consumers

| role | code | notes |
|---|---|---|
| producer (M1) | `trainer/make_canned_blob.py` | stdlib-only Python; also emits the golden vectors + self-test header |
| producer (M2+) | `trainer/export.py` (future) | quantized GRU → type-1 payload |
| consumer | `core/ngpt.cpp` (`ngpt_load`) | the only parser; host tests and the ROM share it |

The blob, the expected output, and the on-ROM self-test golden are all
emitted by one script in one run — they cannot drift apart silently.
