# Spike: GRU matvec on the RSP (idea #6 — the 5–10× headline)

**Branch:** `worktree-rsp-spike` (not for merge). **Question:** can the
N64's RSP (the graphics coprocessor's 8-lane × 16-bit vector unit) run
the GRU's hot matvec bit-exactly, and how much faster?

## Why it might work

- The step cost is ~61K MACs; the CPU (post-M5) does one step in 9.8ms.
  The RSP does 8 MACs/cycle at 62.5MHz with zero-wait DMEM — a naive
  ceiling of ~0.15ms/step for the matvec share.
- Toolchain: libdragon's rspq overlay framework ships in the SDK
  (`rsp_queue.inc`), and `n64.mk` auto-assembles `rsp*.S` sources. The
  ucode hooks into the generated Makefile via one `Makefile.custom`
  line — no generated files touched, `core/` untouched.

## Exactness scheme (the crux)

`acc[i] = Σ_j w[i·H+j](int8) · h[j](int16)` must equal the CPU's int32
exactly. Scheme: `vmudh`/`vmadh` multiply S16×S16 and accumulate
`product << 16` into the 48-bit per-lane accumulators. With |w| ≤ 127:
products ≤ 2^22, and 16 accumulations/lane peak below 2^42 — no clamp,
no rounding. The exact per-lane int32 is `(ACC_HI << 16) | (u16)ACC_MD`
(read via `vsar`), folded across 8 lanes on the scalar unit. Weights are
DMA'd as packed int8 and unpacked in-register: `lpv` (byte << 8 per
lane) + arithmetic `vsra 8` restores the sign.

## Gates

| gate | what it proves | status |
|---|---|---|
| G1 echo | assemble / register / dispatch / DMA out | **built; Ares verify pending (screen locked)** |
| G2 exact dot | one 128-wide int8×int16 row == CPU bit-for-bit | **built; Ares verify pending** |
| G3 tiled matvec + timing | full 3H×H via DMA tiles; µs vs 9.8ms CPU step | not started (gated on G2) |

## Findings so far

- `rsp_queue.inc` already includes the DMA routines — a second
  `#include <rsp_dma.inc>` double-defines every DMA symbol.
- `vsra` handles shift quantities 1–8; `vsra8` is for 9–15 (the
  assembler enforces it).
- Command dispatch passes the first 4 command words in a0–a3; `ra`
  arrives pointing at `RSPQ_Loop`, so tail-jumping into `DMAOut` returns
  to the queue engine for free — but any `jal` (e.g. the input DMAs)
  must save/restore `ra` first.
- Both CPU and RSP are big-endian: `lqv` of the int16 h-vector needs no
  byte swapping anywhere.
