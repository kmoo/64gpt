# Spike: generalizing the RSP matvec kernel to H=256 — VERDICT: ALL GATES PASS, 2.49×

**Branch:** `worktree-rsp-spike-h256`. **Question:** does the M6.1 RSP
matvec kernel (proven bit-exact and 2.8× at H=128,
`docs/spikes/rsp-matvec.md`) generalize to H=256 — the hidden size M7
shipped with — and how much of the lost speedup comes back?

## Why this existed

M7 doubled the model to H=256 for the "magic zone" (`docs/milestones/
m7.md`), but the RSP kernel adopted in M6.1 was hardcoded for 128
columns (`rspBackendInit()` checked `H != 128` and silently fell back to
CPU). M7 shipped correct but slow: ~208 ch/s (M6.1, RSP, H=128) dropped
to ~25 ch/s (M7, CPU-only, H=256) — fully explained by H² scaling +
losing RSP's 2.1×, not a regression, but real performance left on the
table. Tracked in `docs/plan.md`'s "Known follow-ups" list specifically
so it wouldn't quietly stay unsolved.

## Verdict (2026-07-17, Ares, on-screen)

| gate | result |
|---|---|
| DMEM budget | **FITS** — 3104B .bss + 552B data = 3656B / 4096B, confirmed via the built `.elf`'s actual section sizes, not hand math alone |
| SELFTEST (18 curated goldens, replayed through the new kernel) | **PASS** |
| XCHK (CPU-only generation vs RSP-enabled generation, same seed) | **PASS** — byte-for-byte text AND hidden-state identical |
| Speed | **CPU 38,417us vs RSP 15,428us per step — 2.49×** (beats the H=128 kernel's own 2.1×) |

First real boot, zero bugs — unlike the original H=128 spike's five
(cache-line eviction, lpv rotation, accumulator-clobbering shifts, etc.,
all still respected here since this generalizes rather than rewrites
that kernel's proven primitives). The only genuinely new engineering
decision was the DMEM retiling below.

## What changed from the H=128 kernel

The core per-row math (`DotRow`'s unpack-then-MAC structure, the ACC
folding via `vsar`/`mfc2`) is untouched — H=256 just means more of it.
Three concrete changes:

1. **Row width doubled** (128B -> 256B packed int8 per row), so
   `NgptMatvec`'s row-address arithmetic shifts by 8 instead of 7
   (`row * 256` not `row * 128`).
2. **Row count doubled** (384 -> 768, since `W_hh` is `3H x H` and H
   itself doubled — this is 4x the total weight bytes, not 2x).
3. **Tile depth halved** (16 rows/tile -> 8 rows/tile) to keep
   `W_TILE` at the same 2048B — 16 rows at the new 256B row width would
   be 4096B, the *entire* DMEM budget with nothing left for code. This
   was the one real design decision: more, smaller DMA tiles (96 instead
   of 24) rather than fewer, larger ones. `W16_BUF` and `H_BUF` both
   doubled in step with H (512B and 512B respectively, still with no
   wasted padding — the original kernel's "odd half starts at +256"
   offset convention turned out to already match H=256's exact even-half
   size, so several offset constants didn't need to change at all, only
   the loop counts feeding them).

Game-side (`DialogueDemo.cpp`): `rspWhh`/`rspHShuf`/`rspMvOut` buffer
sizes and the CPU-side h-shuffle loop scale with H (`3*256*256`,
`256`, `768`); `rspBackendInit()`'s guard moved from `H != 128` to
`H != 256` — same shape, not a redesign. Deliberately **not** a
variable/dual-H kernel: nothing in M7 or the planned M8-M10 roadmap
needs H=128 anymore (H=256 is the established floor going forward), and
runtime branching on H would cost real DMEM/IMEM and add exactly the
kind of surface the original spike's bug log warns about, for a case
with no forward use.

## Adoption

Unlike the original spike (recorded only, adopted a full milestone
later at M6.1), this one was adopted immediately in a follow-up commit
on `main` — no open risk left to resolve first, and it directly closes
a tracked regression rather than adding new capability. See
`docs/plan.md`'s Known follow-ups list (moved to Closed) and
`docs/milestones/m7.md`'s Performance section for the before/after.
