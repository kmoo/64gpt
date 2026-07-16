# 07 — Performance: making 68K parameters feel alive

*(M5. Audience: an engineer who has never profiled for a 1996 console.)*

## The question

M4 shipped a correct model; M5 asks whether it is *fast enough to be a
game feature*. The gate from `docs/plan.md`: **≥30 characters/second
sustained** while the demo holds 60 frames/second. The N64's CPU is an
NEC VR4300 (MIPS R4300i core) at 93.75 MHz — one step of our H=128 GRU
is ~61K multiply-accumulates, so the budget is real but not absurd.

## How we measured

Game-side only (`core/` stays libdragon-free): wrap the `ngpt_step` call
in libdragon `get_ticks()`, keep an exponential moving average, print
`STEP <n> US  RAW <n> CH/S` on screen. "RAW" is 1s ÷ step time — the
engine's ceiling, ignoring rendering. Ares is the bench; it emulates at
hardware-accurate speed, and ticks are *emulated* time, so numbers are
valid even when the emulator fast-forwards. Every optimization step had
to keep the full bit-exactness suite green (host goldens + trainer +
ROM self-test) — a fast engine that drifts one bit is a regression, not
a win.

## Results (H=128, V=35, sampled generation)

| build | µs/step | raw ch/s |
|---|---|---|
| baseline: project-default `-Os`, int64 accumulators | 16,595 | 60 |
| `-O3 -funroll-loops` on `core/` objects only | 12,754 | 78 |
| + int32 accumulators in the hot loops | **9,795** | **102** |

Sustained streaming: one char per frame at 60 fps = **60 ch/s with VPS
pinned at 60** — 2× the gate. (Two chars/frame would stream 100+ ch/s
but blows the 16.6ms frame budget and judders; not worth it.)

## What worked

- **`-Os` → `-O3` for the engine only** (`game/Makefile.custom`
  per-object flags; the rest of the ROM stays small). ~1.3×. Modest —
  the loops were already simple.
- **int64 → int32 accumulators** in the gate matvecs and logits. ~1.3×
  again, the single biggest *code* win. Why: C++ promoted every
  multiply-accumulate to 64-bit, and on the R4300i a 64-bit multiply
  (`DMULT`) costs roughly double a 32-bit `MULT` — in the innermost of
  61K MACs. The reference implementation had documented from day one
  that int32 suffices at these dims (|acc| < 2^30); the change is
  bit-identical, and the golden suite is the proof. Two spots genuinely
  need 64 bits and keep an explicit cast: the n-gate's `r × acc_h`
  product (2^14 × 2^30) and the sampler's temperature multiply
  (2^30 × 2^16) — both cold paths.

## Dead ends and non-starters (kept per the project rule)

- **Expecting `-O3` to be the big win.** It wasn't; the arithmetic
  *width* was. On a 90s RISC core the instruction you emit matters more
  than how cleverly it's scheduled.
- **Weight-layout tricks** (cache-line blocking of W_hh): unexplored —
  rows are already walked sequentially and the whole 71KB blob fits in
  RDRAM with the 16KB data cache streaming it; the profile after int32
  didn't justify the churn.
- **RSP matvec** (offloading to the signal processor): documented
  stretch goal, explicitly out of scope for M5 (plan.md). It remains
  the obvious next 5–10× if a future model needs it.

## The boot-time corollary

The ROM self-test replays ~1,000 characters before the first frame, so
step time is also boot time: 16.6ms/step ≈ 17s of black screen at
baseline, ~10s after M5. Worth remembering when the self-test grows.
