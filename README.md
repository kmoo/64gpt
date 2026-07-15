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
- [x] **M1 (host side)** — walking skeleton: canned-line "model" behind
      the final streaming API; parser + generation tests green under
      ASan/UBSan *(ROM assembly pending M0's toolchain —
      `docs/milestones/m1.md`)*
- [ ] **M1 (on Ares)** — ROM v0.1: SELFTEST PASS + streaming dialogue box
- [ ] **M2** — ROM v0.2: real GRU overfit on ONE line (first neural net on N64)
- [ ] **M3** — ROM v0.3: conditioning on a dozen hand-written lines
- [ ] **M4** — ROM v0.9: full generated corpus + temperature/top-k sampling
- [ ] **M5** — ROM v1.0-rc: performance pass, ≥30 chars/sec
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
./pyrite64 --cli --cmd build game/project.p64proj   # → game/64gpt.z64
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

## Docs

Written for a good software engineer who has never done game dev or
embedded ML — start at 00 and read in order, or jump in:

- [00 — N64 primer](docs/00-n64-primer.md)
- [01 — Toolchain & Pyrite64](docs/01-toolchain-and-pyrite64.md)
- [02 — Pyrite64 scripting: DialogueDemo explained](docs/02-pyrite64-scripting.md)
- [03 — The NGPT model blob format](docs/03-blob-format.md)
- Milestone notes: [m0](docs/milestones/m0.md) · [m1](docs/milestones/m1.md)

## Pinned versions

| what | version | why |
|---|---|---|
| [pyrite64-mac](https://github.com/proverbiallemon/pyrite64-mac) | **v0.4.0** (release-asset app at `~/GitHub/Pyrite64-v0.4.0`, source clone at `~/GitHub/pyrite64-mac`) | early-dev engine, breaking changes; upgrade deliberately |
| MIPS toolchain | GCC 14.4.0 in `~/pyrite64-sdk` | installed by the Toolchain Manager |
| Ares emulator | v147+ | hardware-accurate; installed by the Toolchain Manager |
| Python | 3.12 via `uv python pin 3.12` (from M2) | PyTorch wheel availability |
