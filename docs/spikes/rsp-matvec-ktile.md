# Spike: K-dimension tiling for the RSP matvec kernel — decoupling DMEM from H

**Branch:** `worktree-rsp-spike-ktile`. **Status:** kernel code written
(`DotRowChunk` + 2D-tiled `NgptMatvec` at H=320), not yet
hardware-verified — see Status log. **Question:** can the RSP matvec
kernel's DMEM footprint be made independent of H, and how much
wall-clock does it cost?

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

## Beyond the DMEM wall: what K-tiling also unlocks

The DMEM formula in "Why this exists" is the headline result, but it's
not the only thing this design changes. Checked against the actual
dispatch code (`game/src/user/DialogueDemo.cpp:94-96`) rather than
assumed, three further threads are worth recording now so they don't
get rediscovered later.

### 1. The CPU isn't actually freed today — that's a deliberate
simplification the K-tiled kernel makes worth revisiting

```cpp
rspq_write(rspOvlId, 0, 0, PhysicalAddr(rspWhh), PhysicalAddr(rspHShuf), PhysicalAddr(rspMvOut));
rspq_wait(); // never reached from initDelete — see boot sequence below
```

The CPU dispatches the matvec and then **blocks** until the RSP
finishes — the surrounding comment says so explicitly ("the RSP path
blocks, so it can never run in `initDelete`"). So every one of the
"~4-6ms per step" the RSP spends on a matvec today (H=256; grows as
`3H²`, so materially longer at H=320/512) is currently spent with the
CPU idle on `rspq_wait()`, not doing anything else. The closing line of
the original spike — "the CPU is also freed during inference, which is
what the M7 world-simulation vision needs" (`docs/spikes/rsp-matvec.md`)
— describes the architecture's *potential*, not something the code
currently exploits.

The fix is a swap, not new invention: the self-test harness already
proved a non-blocking pattern works on this hardware ("hang-proof lazy
dispatch from `update()` with non-blocking syncpoints and a frame
budget," same doc). Replacing `rspq_wait()` with an `rspq_syncpoint`
+ poll lets the CPU do other work during the matvec window instead of
blocking on it. This is **orthogonal to K-tiling** — it applies just as
well to the existing H=256 kernel on `main`, needs no new trained model,
and is a reasonable thing to prototype separately while this spike is
blocked on an H=320 training artifact.

It matters more *because of* this spike, though: `3H²` scaling means
the matvec window (and thus the CPU idle time being wasted) grows
quadratically as H climbs toward the capacity this spike is chasing —
the bigger the model gets, the more there is to gain from not blocking
on it.

### 2. K-tiling's own DMEM headroom pays for a real RSP-side speedup

At H=320 the K-tiled kernel uses 2,432B of the 4,096B budget — 1,664B
free, versus the ~190B the non-tiled M9 §6 plan would have had at the
same H. That's enough room for the **double-buffered tile DMA** the
very first spike named as unclaimed headroom ("overlap the next 2KB
tile's DMA with this tile's MACs... realistic target 5×+ [vs. today's
2.8x floor]," `docs/spikes/rsp-matvec.md`) — previously unaffordable
because H-scaling ate the budget before double-buffering could get a
second `W_TILE`-sized buffer.

K-tiling also changes the shape of the opportunity, not just its
existence: this design runs 150 DMA/compute cycles per matvec (5
K-chunks × 30 row-tiles) instead of the non-tiled kernel's ~6-30 — more,
smaller transfers, meaning more chances to overlap DMA latency with the
previous chunk's vector compute. Not implemented in this spike (scope
is the DMEM decoupling itself), but the headroom to attempt it now
exists where it didn't before, and should be the next optimization pass
once H=320 clears its gates.

### 3. The kernel generalized into a reusable tiled-matvec primitive, not a GRU-only accelerator

`NgptMatvec` is now parametrized entirely by tile geometry (H, chunk,
row-tile size) rather than baked around one fixed problem shape. M9's
own roadmap already has other small-linear-algebra-shaped needs — item
3, "relationship state: persistent, and randomized for testing," and
M10's procedural-cast scoring are both candidates. Once H=320 is
hardware-verified, the same RSP overlay is a plausible fit for those
too, instead of each subsystem growing its own bespoke CPU-only path.
Recorded as a direction, not a commitment — worth revisiting once those
milestones have real shape.

### The honest caveat

(1) and (3) only pay off once there's other CPU work actually queued to
run during the freed window. Today's demo is one NPC, turn-based,
nothing competing for the CPU — so async dispatch alone won't show a
measurable win yet. The real payoff arrives with M10/M11's world-sim
work. This section is "unlocks the option," not "doubles perf today" —
worth being precise about that distinction before it turns into an
overclaim later.

## Hardware verdict at H=256 (2026-07-17): correct, DMEM-independent, NOT yet faster

Retargeted the kernel to H=256 (chunk=64 unchanged, only the H-derived
outer-loop constants) to A/B directly against the shipped kernel on the
exact same model — no `core/` change needed for this test (see the
milestone-placement section's blocker). Both booted in Ares:

| | CPU (µs/step) | RSP (µs/step) | speedup | DMEM (.bss) |
|---|---|---|---|---|
| Baseline (`main`, non-tiled) | 37,667 | 15,710 | 2.40x | 3,104B |
| K-tiled (this spike) | 37,507 | 19,763 | 1.90x | **2,432B** |

XCHK bit-exact PASS on both. Two real, separable results:

- **DMEM independence: hardware-confirmed and structural**, not just
  measured — 2,432B is fixed by `.bss` declarations that never
  reference H, so it holds at H=320/512/etc. by construction, the same
  way it held here.
- **Speed: the Design A section's ~6% estimate was wrong** — real
  overhead is ~26% (19,763µs vs 15,710µs). Root cause, counted exactly:
  K-tiling does **312 DMA operations vs the old kernel's 193**, and
  **3,072 `DotRowChunk` calls vs 768 `DotRow` calls** (each 1/4-width) —
  the byte-traffic estimate accounted for bytes moved but not the fixed
  per-transfer/per-call overhead (DMA busy-wait, `jal`/`jr`, loop
  setup), which chunk=64 pays 4x as often for the same total work.
  Still beats CPU-only by 1.9x, just not the old kernel's 2.4x at this H.

**Update, same day, after hardware-verifying the `H_BUF` merge:** applied,
compiled clean, booted — XCHK still bit-exact PASS, but **RSP time
barely moved: 19,719µs vs 19,763–19,772µs pre-merge**, a ~0.2%
difference indistinguishable from run-to-run noise (compare CPU-side
numbers across all three runs tonight: 37,507 / 37,522 / 37,523 —
±16µs of jitter on a number nothing here should be changing). This is
a real result, not a null test: DMA op count dropped a genuine ~31%
(312→216, from merging the even/odd `H_BUF` transfer), and it bought
essentially nothing.

**That redirects the diagnosis.** DMA operation count is not the
dominant cost here after all. What the merge *didn't* touch is
`DotRowChunk` call count — still 3,072 calls vs. the old kernel's 768,
each paying `UnpackLoop`/`EvenLoop`/`OddLoop`'s fixed loop-entry and
branch overhead against only 1/4 as much productive vector work per
call. That's the more likely dominant term, and it's exactly what a
coarser chunk would cut directly (chunk=128 halves call count to
1,536; chunk=256 — back to the old kernel's granularity — would drop
it to 768, matching the old kernel's call count exactly, which is a
useful sanity-check target in itself). The "two cheap fixes" framing
from earlier tonight put the DMA-merge and the chunk-size change on
equal footing; tonight's measurement says they weren't equal — the
chunk-size experiment is now the higher-confidence next step, not a
parallel option.

## Further optimization ideas (recorded, not applied — see verdicts below)

Three more levers came up discussing tonight's result. One is real and
should be tried once there's hardware time; the other two looked free
on paper but turn out to compete with each other and with the K-tiling
mechanism itself for the same scarce 4,096B DMEM budget — worth being
precise about that before implementing either blind.

### Async double-buffered DMA — the real lever, still unclaimed by ANY version of this kernel

`rsp_dma.inc` provides `DMAInAsync`/`DMAWaitReady`: the DMA engine and
the vector unit are separate hardware, so today's kernel — this one AND
the shipped one — burns the *entire* transfer latency idle before doing
any math. Overlapping next-tile's DMA with this-tile's compute attacks
a different bottleneck than tonight's finding (latency-hiding, not
op-count) and was named as headroom toward "5x+" all the way back in
the original H=128 spike, never implemented once.

**The real cost, not glossed over this time:** double-buffering needs a
second `W_TILE`-sized buffer so the next transfer can land somewhere
other than where the current one is still being read. At today's
2,048B tile size, two of them is 4,096B — the *entire* DMEM budget,
leaving nothing for `W16_BUF`/`H_BUF`/`OUT_TILE`. Making room means
shrinking the tile (e.g. `tile_rows` 32->16), which — per the op-count
math this whole spike turned on tonight — roughly *doubles* the
W_TILE/OUT_TILE DMA op count. So this isn't a free stack-on-top: it
trades a known op-count cost against an unmeasured latency-hiding gain.
Plausibly still a net win (latency-hiding on a shrunk tile could easily
outweigh double the op count), but "plausibly" is exactly the word —
this needs to be measured, not assumed, especially after tonight's
~6%-estimated/~26%-actual miss on the same kind of arithmetic.

### Pre-shuffled weights in the blob — corrected: NOT free, actively competes with tile size

Originally described (by me, last message, quoting the H=128 spike's
optimistic framing) as "zero runtime cost." That framing predates
tonight's lesson and doesn't survive contact with it. Pre-shuffling
means `W_TILE` holds already-unpacked int16 instead of packed int8 —
**doubling the bytes per element**:

```
tile_rows=32, chunk=64: W_TILE at int8  (today)     = 2,048B
                        W_TILE at int16 (shuffled)   = 4,096B  <- the ENTIRE DMEM budget
to fit back in 2,048B, tile_rows must drop to 16      <- doubles row-tile count
```

Saves real work per `DotRowChunk` call (the ~24-instruction unpack
phase disappears entirely), but halving `tile_rows` to make room
doubles `NUM_ROW_TILES`, which doubles the W_TILE/OUT_TILE DMA op
count — the exact quantity tonight's finding says is expensive. Net
effect is genuinely unclear without measuring both terms; not
implementing this blind.

### NPC batching — the one actually novel idea, not a spike-backlog item

Design B (K-chunk-outer, RDRAM-backed partial sums) was rejected above
specifically because "matvec has N=1, so reusing an H-chunk across many
outputs doesn't apply." That reasoning inverts the moment N stops being
1. If M10/M11's living-NPC world ever needs several NPCs' dialogue
advancing in the same tick, batching them into one wider call turns
this into a small-N GEMM: today's per-call/per-DMA overhead — the same
26% — gets amortized across N NPCs instead of paid once each, and
Design B's rejected-for-N=1 advantage becomes real for the first time.
Not appropriate to build ahead of that use case existing, but worth
remembering this spike's own Design B section stops being the final
word the moment the problem shape changes.

### Why not just do two of these tonight

All three trade against the *same* scarce budget the DMEM-independence
result depends on — double-buffering wants tile-space, pre-shuffling
wants per-element space, and both compete with simply growing
`tile_rows`/`chunk` (item 2 in "Next," above) for the identical 2,048B
`W_TILE` allowance. Stacking two of these blind, on top of an
already-uncertain 26%-miss night, means compounding unverified risk
instead of measuring one change at a time. The `H_BUF` merge above was
worth doing now because it's unambiguous — same bytes, half the ops,
no tradeoff against anything else. These three aren't that; they need
the merged-DMA number first, then a real decision informed by data
instead of a second round of arithmetic that might also be wrong.

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

- 2026-07-17 (same day, later): retargeted to H=256 for a same-model
  A/B against the shipped kernel (no `core/` change needed — see the
  milestone-placement section). Hardware-verified, twice: baseline
  CPU 37,667µs/RSP 15,710µs (2.40x, .bss 3,104B) vs. K-tiled CPU
  37,507µs/RSP 19,763µs (1.90x, .bss **2,432B**). Correct (XCHK
  bit-exact both times) and DMEM-independent (structural, not just
  measured) — the two headline claims hold. Speed does not yet beat
  the kernel it would replace: ~26% slower at equal H, and the design
  doc's own ~6% overhead estimate was wrong before hardware corrected
  it. Applied the `H_BUF` DMA merge (two linear transfers → one pitched
  `DMA_SIZE(64,2)`) and re-verified: RSP 19,719µs — **no meaningful
  change** (~0.2%, within run-to-run noise), despite a real ~31% drop
  in DMA op count. Diagnosis redirected: `DotRowChunk` call count
  (3,072 vs. the old kernel's 768), not DMA op count, is the more
  likely dominant cost — the coarser-chunk experiment (not yet applied)
  is now the higher-confidence next step. Screenshots and captions in
  `talk/` (`2026-07-17-ktile-spike-h256-*.png`).

- 2026-07-17 (same day, later still): tested the coarser-chunk
  hypothesis directly — CHUNK 64→128 at the same H=256 (`tile_rows`
  correspondingly 32→16 to hold `W_TILE` at 2,048B; `DotRowChunk` call
  count halves, 3,072→1,536; `.bss` grows slightly to 2,624B, still
  well under budget). Hardware-verified: XCHK bit-exact PASS, **RSP
  19,719µs → 17,118µs, a real 13.2% speedup** — unlike the DMA merge,
  this one worked, confirming call count (not DMA op count) was the
  dominant cost. Closes the gap to the old non-tiled kernel from 26%
  slower to **~9% slower** (17,118 vs. 15,710µs) at equal H, with less
  DMEM used either way (2,624B vs. 3,656B). Hit one real bug along the
  way: an on-screen debug label ("K128-H256 ON") pushed the XCHK
  results line past the N64 debug font's fixed-width screen edge and
  clipped the RSP number itself off a screenshot — a genuine draw-width
  overflow, not a capture artifact, fixed by shortening the label
  ("K128 ON") and re-verifying. Every per-build on-screen tag from here
  on needs to leave the full result line visible, not just look right
  at a glance.

- 2026-07-17 (same day, the headline test): **H=512, chunk=64** — the
  size the old non-tiled kernel's own DMEM formula (`m9.md` §6)
  physically cannot reach at all (hard wall ~H=368), not just one it's
  slower at. There is no old-kernel number to compare against here;
  that absence is the point. Required two things resolved first, both
  disclosed rather than assumed away: (1) a worktree-local `core/
  ngpt.h` bump (`NGPT_GRU_MAX_HIDDEN` 256→512, not on `main`) — see
  that file's updated comment for the int32 accumulator overflow
  analysis this rests on (proceeding on a realistic-bound argument,
  not a full proof; flagged for whoever's driving the H=320 training
  session too, since they'll hit the same question with real weights);
  (2) a throwaway H=512 model — right shape, gibberish content,
  converged on a single 21-char line in 10.6s on CPU (`trainer/
  make_ktile_h512_spike_blob.py`), specifically converged rather than
  left randomly-initialized so it reliably hits EOS well inside the
  ROM self-test's 2000-step runaway guard, which matters more at
  H=512's ~4x per-step cost.

  Hit one real bug getting there: first boot showed `SELFTEST FAIL`
  with no XCHK line at all (the pattern for "model blob failed to
  load," not "ran and failed") — caused by exactly the incremental-
  build gotcha this project's own `CLAUDE.md` documents: changing the
  shared `core/ngpt.h` header without a clean rebuild left a stale
  `.o` with the old struct layout. `rm -rf game/build` fixed it
  immediately.

  Clean-rebuilt result: **`SELFTEST PASS`, `H512 ON`, `XCHK PASS`
  (bit-exact), CPU 136,419µs / RSP 64,962µs — a real 2.10x speedup**,
  slightly *better* than H=256's 1.90x (RSP scales 3.29x for a 2x H
  increase vs. CPU's 3.64x — sub-quadratic relative to CPU, not just
  in absolute terms). DMEM: **2,432B — byte-identical to the H=256
  build's footprint.** That equality, not just "under budget," is the
  actual proof: nothing in this build's DMEM changed when H doubled.
  Both headline claims of this spike are now hardware-verified, not
  just formula-proven: correctness holds, and the wall is genuinely
  gone. Screenshot and caption in `talk/`
  (`2026-07-17-ktile-spike-h512-xchk.png`).
