# Spike: K-dimension tiling for the RSP matvec kernel — decoupling DMEM from H

**Branch:** `worktree-rsp-spike-ktile`. **Status:** design-only, no kernel
code written yet — this doc exists to pick a design before touching
`rsp_ngpt.S`, per the project's "detailed files before implementation"
request. **Question:** can the RSP matvec kernel's DMEM footprint be made
independent of H, and how much wall-clock does it cost?

## Why this exists

`docs/milestones/m9.md` §6 derives the current kernel's DMEM formula from
the real `.bss` layout (`game/src/user/rsp_ngpt.S:38-46`):

```
DMEM(H) = 2600 + 4H + 8192/H       budget: 4096B
```

| H | DMEM used | headroom |
|---|---|---|
| 256 (M8, shipped) | 3,656B | 440B |
| 320 (M9 target) | ~3,906B | ~190B |
| 368 | 4,096B | **0B — unsafe** |
| 473 (800K "magic zone" floor) | ~4,664B | **impossible** |

The `4H` term is `H_BUF` (512B at H=256) + `W16_BUF` (512B) — the entire
hidden state, held resident for the whole matvec call because every one
of the 3H output rows dots against all of it
(`rsp_ngpt.S:40-46`). That's the wall. M9 §6 explicitly parks a "genuinely
different DMEM strategy" as a follow-up rather than attempting it, so
M9's own plan is to ship H≈320 against the *existing*, non-tiled,
`4H`-bound kernel.

This spike asks: what if we tile the reduction dimension (K = H, the
column axis of `W_hh`'s 3H×H) the same way the kernel already tiles the
output dimension (M = 3H, the row axis)? That's genuinely 2D blocking —
the standard trick when neither matrix dimension fits in fast memory —
applied to a case (matvec, N=1) simple enough that it may not need the
full generality real BLAS kernels use.

## Baseline: what stays true regardless of design

- Weight DMA traffic is `3H²` packed int8 bytes, fixed by the math
  problem itself — no tiling strategy changes this. It's the dominant
  cost at any H worth testing (H=320: ~300KB; dwarfs anything below).
- `OUT_TILE` (partial sums for the row-tile currently being processed)
  stays small and untouched by any of these designs — `4 × tile_rows`
  bytes, already the smallest resident structure in the kernel.
- All three designs below preserve `DotRow`'s proven per-lane math
  (`vmudh`/`vmadh` accumulate, exact under 48-bit ACC) untouched — only
  the *tiling/looping* structure around it changes. This matters because
  every bug in the H=128 spike (`docs/spikes/rsp-matvec.md`'s five) was a
  DMA/addressing/lane bug, never a math bug — so the risk in any new
  design is concentrated in loop restructuring and buffer addressing,
  not in `DotRow` itself.

## Design A — row-tile outer, K-chunk inner (recommended first attempt)

Keep today's outer loop (iterate row-tiles of `W_hh`, same as now). Within
each row-tile, don't load all of H up front — load it in fixed-size
chunks (e.g. 64 elements), and for each chunk: DMA the corresponding
chunk-width slice of every row in the current row-tile, dot-product
against the resident H-chunk, and accumulate into `OUT_TILE` (already
row-tile-sized, unchanged). Move to the next H-chunk; when a row-tile's
full K range is exhausted, move to the next row-tile — and reload H from
chunk 0 again, since it's no longer resident.

**DMEM footprint:** `H_BUF`/`W16_BUF` shrink from `2H` bytes each to
`2×chunk` bytes each — a **fixed constant regardless of H** (e.g. 128B
each at chunk=64, vs 640B each at H=320 today). `W_TILE`'s tile depth
also decouples from H: `tile_rows ≈ 2048 / chunk_bytes`, a constant, not
`2048/H` shrinking toward 1 as H grows. New formula:

```
DMEM(chunk) = const + tile_rows×chunk_bytes(W_TILE) + 2×chunk×2(H_BUF+W16_BUF) + 4×tile_rows(OUT_TILE)
```

— independent of H entirely. Pick `chunk` once, and it works for H=320,
512, or 4096 without a new hardcode.

**Extra DMA traffic vs. today:** H gets re-read once per row-tile instead
of once total. At H=320, chunk=64: `tile_rows = 2048/64 = 32`,
`num_row_tiles = 3×320/32 = 30`, H-reread traffic ≈
`30 × 2×320(bytes, shuffled int16 form) = 19,200B ≈ 19KB` — about **6%
on top of the 300KB weight-DMA floor**. Grows linearly with H (more
row-tiles), but stays a fraction of the weight traffic, which also grows
with H (`3H²`) — so the *ratio* stays roughly flat, not worsening as H
climbs. This is the "graceful degradation, not a cliff" property.

**Risk:** low. No new buffer lifetime spans a loop boundary that didn't
exist before (H_BUF's lifetime shrinks from "whole matvec" to "one
row-tile's one K-chunk" — strictly simpler, not more complex). No RDRAM
round-trip, no new synchronization between DMA and compute beyond what
today's kernel already does per row-tile.

## Design B — K-chunk outer, row-tile inner, RDRAM-backed partial sums

Flip the loop order: iterate H-chunks in the outer loop, row-tiles in the
inner loop. Each H-chunk is DMA'd once and reused across *all* row-tiles
(avoiding Design A's re-read), but since a row's partial sum must persist
across the outer K-chunk loop and there are 3H rows total (too many to
keep resident — `4×3H` bytes, which is *worse* than the `4H` term we're
trying to eliminate), each row-tile's partial-sum accumulator must be
DMA'd out to RDRAM after its contribution for this K-chunk and DMA'd back
in when its turn comes again next K-chunk.

**DMEM footprint:** same as Design A — H-chunk buffers and `OUT_TILE`
are both fixed-size regardless of H.

**Extra DMA traffic vs. today:** H is read once total (no re-read), but
partial sums round-trip twice (out + in) per row-tile per K-chunk beyond
the first. At H=320, chunk=64 (5 chunks), 30 row-tiles: partial-sum
traffic ≈ `3×320 rows × 5 chunks × 4 bytes × 2(out+in) ≈ 38,400B ≈ 38KB`
— **about 13% on top of the weight-DMA floor, roughly 2x Design A's
overhead**, because a 32-bit accumulator round-trips more bytes than a
re-read of H's compact int16 form does, and every row-tile pays it, not
just H.

**Risk:** meaningfully higher than A. New failure mode: a row-tile's
accumulator must be correctly zeroed on its *first* K-chunk pass and
correctly loaded (not zeroed) on every subsequent pass — an off-by-one
here produces a wrong-but-plausible number (accumulator either double-
counts or silently drops a chunk), exactly the kind of bug the H=128
spike's bug #5 (cache-line eviction over a DMA result) warns is hard to
diagnose on a ROM with no debugger: the wrong sum is deterministic and
looks like a math bug, not a lifecycle bug.

**Verdict on B:** for this specific problem shape (matvec, N=1), B is
strictly worse than A on both traffic and risk in the estimate above —
the "reuse H-chunk across row-tiles" benefit it's built around doesn't
outweigh the round-trip cost it requires, because H's compact form is
cheaper to re-read than a wide accumulator is to round-trip. B is the
design that generalizes to real GEMM (N>1, where H-chunk reuse pays off
across many output columns, not just row-tiles of one column) — not
obviously a win here. Recording it because it's the "textbook" answer
and worth ruling out explicitly rather than by assumption, and in case A
hits a real-hardware surprise Design A's estimate doesn't capture.

## Design C — shrink the constant, no tiling (fallback, not the goal)

Don't tile K at all; instead shrink `H_BUF`/`W16_BUF`'s per-element cost.
They're already near-minimal (`H_BUF` is one 2-byte-per-element
CPU-shuffled stream, not a doubled representation), so the realistic
squeeze is small — folding `W16_BUF`'s per-row unpack scratch into a
narrower reuse pattern might claw back some headroom, plausibly pushing
the wall from H≈368 to somewhere in the H≈420-450 range, not further.

**Verdict on C:** doesn't reach H=512, let alone the 500-800K magic zone
requiring H≈368-473+. Rejected as the primary design — recorded because
it's the lowest-risk option if A hits an unexpected hardware wall, and
because "shrink the constant" is worth having ruled out on the record
before claiming tiling is *necessary* rather than just the best option.

## Decision

**Implement Design A.** Target progression, hardware-verified at each
step before extending (mirrors the H=128→H=256 spike's own methodology
rather than jumping straight to the interesting number):

1. **H=320**, chunk size TBD by prototyping (64 is the working estimate
   above; the real constraint is `tile_rows = 2048/chunk_bytes` landing
   on a clean power-of-2 row count, same reasoning as the H=256 spike's
   "8 rows/tile" choice). Gate: DMEM fits (trivially, by construction),
   SELFTEST PASS, XCHK bit-exact against the CPU path, and a **direct
   speed comparison against the existing non-tiled H=320 kernel path**
   from M9 §6 — this is the number that matters, since Design A adds
   ~6% DMA overhead in exchange for removing a hard capacity ceiling, and
   that trade should be visible, not assumed.
2. If (1) is clean: **push to H=512**, past the current 368 hard wall —
   this is the step that actually proves the decoupling claim (H=320
   alone doesn't need Design A at all; the untiled kernel already
   reaches it per M9 §6). Same gate sequence.

Not attempting Design B or C in the kernel unless Design A hits a real
hardware surprise this doc's estimates didn't predict — precedent
(`rsp-matvec.md`'s five bugs, all DMA/addressing, none predicted in
advance) says that's a real possibility worth planning for, not just a
hedge.

## Open question: milestone placement

Not resolved here. M9 §6 already claims H≈320 against the *existing*
kernel and is still "planning — not started," so no conflict yet — if
step 1 above lands clean, it most likely just becomes M9's actual §6
implementation (a better kernel choice within the same milestone, same
pattern as how the H=128→H=256 generalization was adopted as a follow-up
commit rather than its own milestone). If step 2 also lands clean, H=512
is capability beyond anything M9/M10/M11 currently scope, and a new
milestone number (M12 would be next free) becomes a real question — but
that's a call for after both gates report a verdict, not before.

## Status log

- 2026-07-17: design doc written. `game/src/user/rsp_ngpt.S` rewritten
  for Design A at H=320, chunk=64 (`DotRowChunk` + a 2D-tiled
  `NgptMatvec`: 30 row-tiles x 5 K-chunks, using the RSP DMA engine's
  native pitched-transfer support to pull each row-tile's K-chunk
  column-slice straight out of the H-wide RDRAM matrix — no RDRAM
  pre-restructuring needed). **Not yet hardware-verified — blocked on
  a trained H=320 model.** No H=320 blob/goldens exist yet (M9 hasn't
  trained anything); game-side wiring (`DialogueDemo.cpp`'s
  `rspWhh`/`rspHShuf`/`rspMvOut` sizes and `rspBackendInit`'s `H!=256`
  guard) is still H=256-shaped and untouched. Deliberately not pushed
  further tonight — `.claude/worktrees/rsp-spike-h256/trainer` has an
  unrelated retrain in progress, and producing a real H=320 training
  artifact isn't something to rush alongside it.
  Remaining before a verdict: (1) an H=320 trained blob + goldens,
  (2) game-side buffer/guard updates to match, (3) build + Ares boot,
  (4) SELFTEST + XCHK (CPU vs RSP, bit-exact) + speed number vs CPU-only
  H=320. Then, if clean, the mechanical H=320->H=512 constant swap this
  doc's Decision section describes.
