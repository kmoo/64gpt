# Spike: GRU matvec on the RSP — VERDICT: ALL GATES PASS, 2.8× unoptimized

**Branch:** `spike/rsp-matvec` (record only, not merged). **Question:**
can the N64's RSP (the graphics coprocessor's 8-lane × 16-bit vector
unit) run the GRU's hot matvec bit-exactly, and how much faster?

## Verdict (2026-07-16, Ares, on-screen artifacts in talk/)

| gate | result |
|---|---|
| G1 — toolchain: overlay assembles, registers, dispatches, DMAs | **PASS** |
| G2 — exactness: 128-wide int8×int16 dot == CPU, bit-for-bit | **PASS** |
| G3 — full 384×128 matvec (a GRU step's 3H rows), DMA-tiled | **PASS — all 384 rows exact; RSP 3,022µs vs CPU 8,596µs (2.8×)** |

That 2.8× is a *floor*: the ucode re-unpacks every weight row from
int8, does no DMA/compute overlap, and the stopwatch includes rspq
dispatch. Known headroom, in order of payoff:

1. **Pre-shuffled weights in the blob** — the trainer emits W in the
   even/odd int16 layout the VU wants; the entire unpack phase (~half
   the inner loop) disappears. Zero runtime cost, one exporter change.
2. **Double-buffered tile DMA** — overlap the next 2KB tile's DMA with
   this tile's MACs (the DMA engine queues one transfer for free).
3. Unroll RowLoop / keep two rows in flight in the vector regs.

Realistic target after 1–2: **5×+ on the matvec**, i.e. a ~4.2ms →
~2.5ms step at H=128 (≈ 240–400 raw chars/sec), or H=256 (~250K params
— the M7 "magic zone") at today's speed. The CPU is also *freed* during
inference, which is what the M7 world-simulation vision needs.

## How it works

- rspq overlay (`game/src/user/rsp_ngpt.S`), three commands: echo /
  one-row dot / tiled matvec. Hooked into the build by one
  `Makefile.custom` line (n64.mk assembles any `rsp*.S`); `core/` and
  the frozen API untouched — all RSP code is game-side.
- Exactness scheme: `vmudh`/`vmadh` accumulate (S16×S16 product) << 16
  into the 48-bit per-lane accumulators; 16 products of ≤2^22 peak
  below 2^42 — no clamp, no rounding, ever. Per-lane int32 =
  `(ACC_HI << 16) | (u16)ACC_MD` via `vsar`, folded on the scalar unit.
- Weights DMA'd packed (int8); unpacked in-register from `lqv`
  byte-pairs — even bytes by arithmetic `>>8`, odd bytes by
  `<<8` then `>>8` (both exact). The CPU pre-shuffles h once per step
  into even/odd streams; dot products are order-invariant.
- Tiling: h resident in DMEM (256B); W streamed as 24 × 2KB tiles
  (16 rows each); 16 int32 sums DMA'd out per tile.
- CPU harness: hang-proof lazy dispatch from update() with non-blocking
  syncpoints and a frame budget — every outcome (PASS/FAIL/HANG) is a
  line on screen, because there is no debugger on a ROM.

## The five bugs (chronological — each cost a build-boot-capture cycle)

1. **`rspq_wait` in initDelete hung the boot** (black screen, no
   diagnosis surface). Never block before the first frame; dispatch
   from update() and poll.
2. **`#include <rsp_dma.inc>` double-defined every DMA symbol** —
   rsp_queue.inc already includes it.
3. **Vector shifts are multiplies and CLOBBER the accumulator.** The
   rsp.inc `vsra`/`vsll8` macros expand to vmudm/vmudn — interleaving
   unpack shifts with vmadh zeroed the whole dot product. All unpacking
   must complete before the first vmudh.
4. **`lpv` rotates lanes by the address's low 4 bits** — packed loads
   from `addr%16==8` land rotated by 4 lanes. Eliminated lpv entirely
   in favor of lqv byte-pair splitting.
5. **The one that ate the night: CPU cache-line eviction over the DMA
   result.** The 8-byte result buffer shared a 16-byte cache line with
   CPU-written statics; dirty-line writebacks overwrote the RSP's DMA'd
   bytes in RDRAM with stale values. The wrong sum was *deterministic
   and invariant across two ucode rewrites* — which looked exactly like
   a VU math bug and was diagnosable only by noticing that no ucode
   change moved the number. Every CPU/RSP-shared buffer must own its
   cache lines: `aligned(16)`, size a multiple of 16, and one side
   accesses uncached.

Also learned: `mfc2`'s `.eN` goes through `byteVectorElements`, which
already maps lane N to byte offset 2N — lane indices 0..7 are correct
as written (a wrong "fix" to byte offsets 0,2,..,14 doesn't even
assemble).

## What adoption would look like (post-spike, NOT scheduled)

- Trainer: emit W_hh/W_out pre-shuffled (blob layout change → the
  sanctioned model-type-2 path, or a v2 payload flag).
- Engine: `ngpt_step` gains an RSP backend behind the same frozen API;
  greedy/sampled goldens stay the referee (they already pass against
  this kernel's math).
- The M5 CPU path stays as fallback and as the bit-exactness
  cross-check — the self-test can run both and compare.
