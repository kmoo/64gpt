# 64GPT — a tiny neural NPC dialogue brain running on a real Nintendo 64

A character-level GRU (~100K parameters, int8-quantized) that generates
short NPC dialogue **on real N64 hardware**, built with the
[Pyrite64](https://github.com/HailToDodongo/pyrite64) engine on
[libdragon](https://libdragon.dev). The demo is a dialogue box that
streams AI-generated text; controller buttons cycle the conditioning
(`NPC=guard / Mood=angry / Event=stole_sword`), A regenerates.

## Method: the walking skeleton

The AI is never built in isolation and "ported at the end". A bootable
ROM ships at **every** milestone — starting with a fake model that says
one canned line — and each milestone swaps in the smallest real piece.
The on-console proof is re-verified continuously:

- Inference is pure integer math and the model blob is parsed
  byte-by-byte, so output is **bit-exact identical** on the Mac, in the
  Ares emulator, and on the console.
- Every ROM boots with a **self-test** that replays committed golden
  vectors and prints `SELFTEST PASS/FAIL` on screen.
- Per-milestone gate: host tests green + Ares boot with SELFTEST PASS.
  Real-hardware (EverDrive-64) runs are occasional spot-checks, and
  required for v1.0.

## Status

- [x] **M0** — repo skeleton, docs, verification loop: toolchain +
      Ares installed, stock example built headlessly and boots in Ares
      (`docs/milestones/m0.md`)
- [x] **M1** — ROM v0.1, the walking skeleton: canned-line "model" behind
      the final streaming API; parser + generation tests green; boots in
      Ares with SELFTEST PASS + streaming dialogue box
      (`docs/milestones/m1.md`)
- [x] **M2** — ROM v0.2: real GRU overfit on ONE line — first neural net
      on N64: int8 weights, integer-only inference, bit-identical to the
      Python trainer (`docs/milestones/m2.md`)
- [x] **M3** — ROM v0.3: conditioning on a dozen hand-written lines —
      prompt priming (`NPC=… MOOD=… EV=…|`) selects which line the GRU
      speaks; 12/12 bit-exact on host and in the ROM self-test
      (`docs/milestones/m3.md`)
- [x] **M4** — ROM v0.9: full generated corpus + temperature/top-k
      sampling — 1.5MB generated corpus, H=128 (~68K params) trained
      with a val split; seeded xorshift32 sampler bit-exact from trainer
      to silicon; every regenerate speaks a fresh in-character line
      (`docs/milestones/m4.md`)
- [x] **M5** — ROM v1.0-rc: performance pass — 16.6ms → 9.8ms per char
      (`-O3` core + int32 accumulators, bit-exactness preserved); raw
      102 chars/sec, sustained 60 at a held 60 VPS — 2× the ≥30 gate
      (`docs/milestones/m5.md`, `docs/07-performance.md`)
- [ ] **M6** — ROM v1.0: running on a real N64 via EverDrive

## Quickstart

Host tests (any OS, needs cmake + a C++20 compiler):

```sh
cmake -B build tests && cmake --build build
ctest --test-dir build --output-on-failure
```

Regenerate the model blob + golden vectors (stdlib-only Python 3):

```sh
python3 trainer/make_canned_blob.py
```

Build the ROM (macOS, after the one-time toolchain install —
`docs/01-toolchain-and-pyrite64.md`):

```sh
# ./pyrite64 is a local wrapper that execs the app binary by its REAL path —
# running it via a plain symlink breaks the app's resource lookup and the
# build fails with an empty Makefile (docs/01-toolchain-and-pyrite64.md).
./pyrite64 --cli --cmd build "$PWD/game/project.p64proj"   # → game/64gpt.z64
ares game/64gpt.z64
```

## Repo map

| dir | contents |
|---|---|
| `core/` | the inference engine — portable C-style C++, zero libdragon deps, compiled unchanged into host tests and the ROM |
| `tests/` | host test suite (CMake/CTest, ASan/UBSan); `tests/vectors/` holds committed goldens |
| `trainer/` | Python tooling: blob export now; corpus/train/quantize/ref-impl from M2 |
| `game/` | the Pyrite64 project (see `game/README.md`) |
| `docs/` | concept guides + per-milestone notes |
| `versions/` | built `.z64` ROMs, one per milestone (EverDrive-ready; see its README) |

## Docs

Written for a good software engineer who has never done game dev or
embedded ML — start at 00 and read in order, or jump in:

- [00 — N64 primer](docs/00-n64-primer.md)
- [01 — Toolchain & Pyrite64](docs/01-toolchain-and-pyrite64.md)
- [02 — Pyrite64 scripting: DialogueDemo explained](docs/02-pyrite64-scripting.md)
- [03 — The NGPT model blob format](docs/03-blob-format.md)
- [04 — Fixed-point inference](docs/04-fixed-point-inference.md)
- [05 — A GRU on a napkin](docs/05-gru-on-a-napkin.md)
- [06 — The training pipeline](docs/06-training-pipeline.md)
- [07 — Performance](docs/07-performance.md)
- Milestone notes: [m0](docs/milestones/m0.md) · [m1](docs/milestones/m1.md) · [m2](docs/milestones/m2.md) · [m3](docs/milestones/m3.md) · [m4](docs/milestones/m4.md) · [m5](docs/milestones/m5.md)

## Pinned versions

| what | version | why |
|---|---|---|
| [pyrite64-mac](https://github.com/proverbiallemon/pyrite64-mac) | **v0.4.0** (release-asset app at `~/GitHub/Pyrite64-v0.4.0`, source clone at `~/GitHub/pyrite64-mac`) | early-dev engine, breaking changes; upgrade deliberately |
| MIPS toolchain | GCC 14.4.0 in `~/pyrite64-sdk` | installed by the Toolchain Manager |
| Ares emulator | v147+ | hardware-accurate; installed by the Toolchain Manager |
| Python | 3.12 via `uv python pin 3.12` (from M2) | PyTorch wheel availability |
