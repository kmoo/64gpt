# 64GPT — Full project plan (M0–M6)

This is the approved walking-skeleton plan the project runs on, including
engine facts verified against Pyrite64's actual source. Working conventions
live in the root `CLAUDE.md`; per-milestone status and remaining checklists
live in `docs/milestones/`.

## Status at handoff (2026-07-15)

- **M0** — done (2026-07-15): repo skeleton, docs, host verification
  loop, Mac toolchain (GCC 14.4.0 → `~/pyrite64-sdk`), pyrite64-mac
  v0.4.0, stock `empty` example built headlessly and boots in Ares.
  CLI path gotchas documented in `docs/milestones/m0.md`. Tagged `m0`.
- **M1** — done (2026-07-15): canned-line model behind the frozen
  streaming API. Host tests green on Linux/gcc (ASan/UBSan) and
  macOS/AppleClang (`-DNGPT_SANITIZE=OFF`; the ASan runtime deadlocks
  before `main` on macOS 26.5 + AppleClang 17). ROM v0.1 built headlessly
  and verified in Ares: SELFTEST PASS, line streams, A regenerates
  (screenshot in `docs/milestones/m1.md`). CLI invocation gotcha (must
  exec the app binary by real path, not a symlink) documented there and
  in `CLAUDE.md`. Tagged `m1`.
- **M2–M6** — not started; specified below.

## Context

Goal (non-negotiable): a character-level GRU (~100K params, int8) generating short NPC dialogue **on a real N64** at the end, integrated via the Pyrite64 engine. Text-only demo: a dialogue box streams AI-generated text; buttons cycle conditioning (`NPC=guard / Mood=angry / Event=stole_sword`).

**Method:** never build the AI in isolation and port at the end. Ship a bootable ROM at **every** milestone, starting with a fake "model" that says one canned line, then swap in the smallest real piece at a time (GRU overfit on ONE line → a dozen lines → full corpus). Each milestone = red tests → implement → green tests → **bootable ROM + README update + git tag**.

**Emulator-first workflow (per user):** Ares (v147+, hardware-accurate) is the per-milestone gate — every ROM must boot there with `SELFTEST PASS`. The user has an EverDrive-64; real-console runs are **occasional spot-checks**, recommended at M2 (first neural net on silicon) and required for final v1.0 (M6). Bit-exact integer math makes this safe: host-green ⟹ Ares-green ⟹ silicon-green.

Environment: repo `~/GitHub/64gpt` (git `main`, remote `git@github.com:kmoo/64gpt.git`, one empty commit). macOS Apple Silicon, no N64 toolchain yet. Python 3.14.6 → **pin 3.12 via uv** for PyTorch. Host-side milestones (core/, tests/, trainer/) are OS-agnostic; ROM builds require the Mac toolchain install (M0).

## Verified facts (checked against Pyrite64 source, not just README)

- **Headless build is real:** `./pyrite64 --cli --cmd build /path/to/project.p64proj` (upstream `docs/docs/manual/cli.md`); `--cmd clean` also exists. **The mac fork ships `src/cli.cpp` too** — CLI works there (fork latest release v0.4.0; upstream is at v0.7.0 — the fork lags, pin the fork release and note the version in README).
- **User code pickup:** generated Makefile compiles `$(wildcard src/user/*.cpp)` plus one `src += $(wildcard src/user/<dir>/*.cpp)` line per subdirectory found by `Utils::FS::scanDirs` (`src/build/projectBuilder.cpp:140-143`). Two hard constraints found in source:
  - Only **`.cpp`** files are globbed and only `.cpp` is mapped to objects (`$(src:%.cpp=$(BUILD_DIR)/%.o)`) ⇒ **core engine files must be `.cpp`** (written as C-style C++: stdint, no exceptions/RTTI/heap — build already uses `-fno-exceptions`).
  - `scanDirs` uses `fs::recursive_directory_iterator`, which **lists a symlinked dir but does not recurse into it** ⇒ symlink `game/src/user/n64gpt → core/` works **iff `core/` is flat** (no subdirectories). Keep it flat. Fallback if the editor ever misbehaves with the symlink: a 3-line pre-build rsync in `Makefile.custom`.
  - `-Isrc/user` is on the include path ⇒ game code does `#include "n64gpt/ngpt.h"`.
- **Custom binary asset (the model blob):** do **not** put `model.bin` in `assets/` (the editor's converter list decides `{{ASSET_LIST}}`; unknown types aren't in the pipeline). Instead: the generated Makefile `include`s **`Makefile.custom`**, and the DFS rule packs the entire `filesystem/` dir. So in `game/Makefile.custom`:
  ```make
  $(BUILD_DIR)/$(ROM_NAME).dfs: filesystem/model.bin
  filesystem/model.bin: rawfs/model.bin
  	@mkdir -p $(dir $@) && cp $< $@
  ```
  (`assets_conv +=` would NOT work — the template assigns `assets_conv =` *after* the include; an extra prerequisite on the `.dfs` target does.) Runtime load: libdragon `asset_load("rom:/model.bin", &size)`.
- **Dialogue box reference implementation exists:** upstream example `n64/examples/jam25/src/user/systems/dialog.cpp` + `systems/fonts.cpp` — a complete text box with `rdpq_font_load` + `rdpq_text_printf` from a script's `Draw`. Crib this pattern for M1 instead of inventing one.
- Object scripts: C++ in `src/user/`, UUID namespace `P64::Script::<UUID>`, `P64_DATA` params, lifecycle `Init/Update/FixedUpdate/Draw/Destroy/OnEvent` (docs `manual/script/objScript.md`).
- mac fork ([proverbiallemon/pyrite64-mac](https://github.com/proverbiallemon/pyrite64-mac)): Apple-Silicon-only, Toolchain Manager installs libdragon SDK to `$HOME/pyrite64-sdk` (~30–60 min, **user-attended**, no sudo), auto-installs Ares via Homebrew.
- **Endianness:** N64 is big-endian MIPS, Mac is little-endian. Inference is pure integer math and the blob parser is byte-oriented ⇒ bit-exact identical output on host tests, Ares, and real N64. This is what lets host TDD prove hardware correctness.

## Architecture (fixed from M1)

- **Streaming API (never changes after M1):** `ngpt_load(blob, len) → ngpt_reset(ctx, prompt) → ngpt_step(ctx) → next char (or EOS)`. Game loop calls `step` N times per frame. M1 implements it with a canned string; later milestones swap internals only.
- **Model "64GPT-100K" (final form):** char-level GRU, 1 layer, 128 hidden, vocab ≤ 96 (printable ASCII subset + `\n` + EOS), one-hot input. ≈99K params ≈100 KB int8. ~99K MACs/char → 30+ chars/sec on the 93.75 MHz R4300i is realistic.
- **Math:** int8 weights (+per-tensor scales), int32 accumulators, int16 Q4.11 hidden state, integer LUT sigmoid/tanh **emitted by the trainer** (single source of truth). Sampling: temperature + top-k on integer logits, xorshift32 seeded RNG. Zero floats at inference.
- **Blob format `NGPT`:** magic, version, model-type (v0 canned-text / v1 GRU), dims, vocab table, scales, LUTs, weights — big-endian, byte-oriented parser, versioned so M1's canned blob and M2+'s GRU blob ship through the same loader.
- **On-target test gate:** every ROM embeds a boot self-test replaying committed golden vectors, printing `SELFTEST PASS/FAIL` on screen before the demo starts.

## Data / verification flow

```mermaid
flowchart LR
  subgraph trainer [trainer/ Python 3.12+uv]
    T[train.py GRU] --> Q[quantize + LUT emit]
    Q --> R[ref_impl.py int NumPy]
    Q --> B[export model.bin NGPT blob]
    R --> G[goldens: hidden states, logits, output bytes]
  end
  subgraph core [core/ flat C-style .cpp, zero libdragon deps]
    E[ngpt engine: parser, matvec, GRU cell, sampler]
  end
  G -->|committed tests/vectors/| H[host CTest + ASan/UBSan\nbit-exact vs goldens]
  E --> H
  E -->|symlink game/src/user/n64gpt| ROM[pyrite64 --cli --cmd build]
  B -->|game/rawfs + Makefile.custom → DFS| ROM
  G -->|embedded selftest vectors| ROM
  ROM --> A[Ares: SELFTEST PASS\nevery milestone]
  A -.occasional: M2 + v1.0.-> N64[EverDrive on real N64]
```

Bit-exact integer math is the load-bearing arrow: the same bytes must fall out of `ref_impl.py`, the host tests, Ares, and silicon.

## Repo layout

```
64gpt/
  README.md          # living deliverable: goal, status checklist, build/run, pinned versions
  core/              # FLAT dir, C-style C++ (.cpp/.h), zero libdragon deps (canonical)
  tests/             # host C++ tests (CMake + CTest, ASan/UBSan); tests/vectors/ = goldens
  trainer/           # Python (uv, pin 3.12): corpus_gen, vocab, train, quantize, export, ref_impl (+ pytest)
  game/              # Pyrite64 project: src/user/n64gpt → symlink ../../core;
                     #   src/user/DialogueDemo.cpp; rawfs/model.bin; Makefile.custom
  docs/              # concept guides + per-milestone notes (see Documentation track)
```

## Documentation track (a first-class deliverable at every milestone)

Audience: **a good software engineer who has never done game dev and never touched this tech** — no N64, no libdragon/Pyrite64, no fixed-point/quantized inference background assumed. Every doc explains *why*, not just *what*; every code sample is copy-pasteable; every claim links to where it was verified. Written **while building the milestone that needs it** (not backfilled), and each milestone's definition of done includes its docs.

Concept guides (each ~1–2 pages, created at the milestone that first needs the concept):

- `docs/00-n64-primer.md` *(M0)* — what an N64 actually is to a programmer: 93.75 MHz MIPS R4300i, 4 MB RAM, big-endian, bare-metal (no OS), what a ROM/`.z64` is, why "hardware-accurate emulator" (Ares) is a trustworthy proxy, what an EverDrive does.
- `docs/01-toolchain-and-pyrite64.md` *(M0)* — the layer cake: libdragon (open-source SDK/kernel) → tiny3d → Pyrite64 engine+editor; project anatomy (`assets/`, `data/`, `src/p64` generated, `src/user` yours, `Makefile.custom` hook); the headless CLI build; annotated walkthrough of our exact toolchain-install steps with screenshots of the Toolchain Manager.
- `docs/02-pyrite64-scripting.md` *(M1)* — object scripts for someone who's never used a game engine: the scene/object model, lifecycle (`Init/Update/FixedUpdate/Draw`), `P64_DATA`, UUID namespaces, how a frame works (game loop ≠ your program's main), how our dialogue box draws text (`rdpq_text_printf`, cribbed from jam25 and explained line-by-line).
- `docs/03-blob-format.md` *(M1)* — the `NGPT` format spec: byte-level table (offsets, types, endianness), why big-endian on the wire, why a byte-oriented parser makes host/target bit-identical, how versioning lets a canned-text blob and a GRU blob share one loader.
- `docs/04-fixed-point-inference.md` *(M2)* — floats-to-integers for engineers who've only used floats: Q-format notation (what Q4.11 means), saturation, int8 quantization with scales, why LUT sigmoid/tanh, and the punchline: integer math is deterministic across architectures, which is what makes "test on Mac, trust on N64" sound.
- `docs/05-gru-on-a-napkin.md` *(M2)* — the GRU cell as ~6 equations and a diagram, char-level modeling, why 100K params fits the N64 budget (params × bytes, MACs/char vs CPU clock), greedy vs sampled decoding.
- `docs/06-training-pipeline.md` *(M2–M4, grows)* — corpus → vocab → train → quantize → export → goldens; how to re-run every stage; what the acceptance metrics mean.
- `docs/07-performance.md` *(M5)* — how we measured chars/sec on-target, what the R4300i cache is, which matvec optimizations paid off and which didn't (keep the failures — they're the education).
- `docs/hardware-checklist.md` *(M6)* — EverDrive from zero: formatting the SD card, where the `.z64` goes, boot steps, what SELFTEST PASS looks like, troubleshooting.

Per-milestone notes `docs/milestones/mN.md` (short, written at tag time): what was built, what it *proves*, the red→green test list, gotchas hit and how they were solved, screenshot/photo of the ROM running. These are the narrative thread a newcomer can read start-to-finish to retrace the whole project.

README.md stays the front door: goal, current-status checklist (☑ per milestone), 5-minute build/run quickstart, pinned versions, and a linked table of contents into `docs/`.

## Milestones (each ends: tests green → ROM boots in Ares with SELFTEST PASS → **docs written** → README updated → tag `mN`)

### M0 — Environment + skeleton repo *(deliverable: README v0 + verified toolchain)*
1. `brew install cmake ninja git-lfs`; scaffold dirs; README v0 (goal, milestone checklist, pinned pyrite64-mac release, build instructions).
2. Clone pyrite64-mac **at its latest tagged release (v0.4.0)**; run Toolchain Manager (30–60 min, the one user-attended step); Ares auto-installed.
3. Prove the loop headlessly: create/build a stock example via `pyrite64 --cli --cmd build`, boot the `.z64` in Ares. Commit + tag `m0`.
- **Docs:** `00-n64-primer.md`, `01-toolchain-and-pyrite64.md` (written during the install, while the gotchas are fresh), README v0, `milestones/m0.md`.

### M1 — Walking skeleton: "AI" says one canned line *(ROM v0.1)*
- Host TDD: NGPT blob parser (magic/version/big-endian fields on hand-crafted byte arrays), canned-text model (v0) behind the full `ngpt_*` API, EOS handling, per-frame step budget.
- Game: create `game/` project via the editor once (launcher), then commit; add symlink `src/user/n64gpt → ../../core`; `DialogueDemo.cpp` — `Init` = `asset_load("rom:/model.bin")` + `ngpt_load`, `Update` = step N chars/frame + A-button regenerate, `Draw` = dialogue box (port pattern from jam25 `systems/dialog.cpp`/`fonts.cpp`). `Makefile.custom` rule for `rawfs/model.bin` (above). Boot self-test v0: canned output vs embedded golden bytes.
- **Proves on Ares:** CLI build pipeline, symlinked core compiles, DFS blob loading, text rendering, streaming API, self-test harness.
- **Docs:** `02-pyrite64-scripting.md` (line-by-line walkthrough of `DialogueDemo.cpp`), `03-blob-format.md`, `milestones/m1.md`.

### M2 — Real GRU, ONE line of training data *(ROM v0.2 — first neural net on N64)*
Entire ML pipeline lands; verification is trivial: GRU overfit on one line must reproduce it byte-for-byte, greedy.
- Python TDD: vocab round-trip; overfit-to-~zero-loss; quantizer error bounds; `ref_impl.py` integer NumPy inference; blob export/re-import round-trip; commit goldens (per-step hidden states, logits, exact output bytes).
- C TDD (red→green order): saturating fixed-point ops → LUT sigmoid/tanh vs trainer-emitted tables → int8 matvec vs goldens → GRU cell single step bit-exact → full greedy generation byte-identical → swap in behind `ngpt_*` (blob v1).
- ROM: same demo, new blob; self-test replays GRU goldens. **Recommended EverDrive spot-check** — the "neural net on silicon" moment.
- **Docs:** `04-fixed-point-inference.md`, `05-gru-on-a-napkin.md`, `06-training-pipeline.md` v1, `milestones/m2.md`.

### M3 — Conditioning: a dozen hand-written lines *(ROM v0.3 — prompt-controlled)*
- ~12 hand-written `NPC=/Mood=/Event= → response` pairs; model memorizes all; prompt selects which.
- TDD: per-prompt golden strings (greedy); C prompt-priming path; ROM input — D-pad/C cycle NPC/mood/event, A regenerates, prompt shown on screen.
- **Proves** the conditioning mechanism end-to-end before scaling data.
- **Docs:** `milestones/m3.md` (incl. how prompt priming works); update `06-training-pipeline.md`.

### M4 — Full generated corpus + sampling *(ROM v0.9 — real generalizing AI)*
- Corpus generator TDD (deterministic seed, grammar/charset/length invariants, NPC×mood×event coverage, 1–2 MB) → train 64GPT-100K (acceptance: val-loss threshold + int-vs-float top-1 agreement ≥95% + human eyeball per condition).
- Sampler TDD: xorshift32 sequence, temperature/top-k bit-exact vs `ref_impl.py`; seeded goldens ⇒ self-test stays deterministic.
- ROM: varied in-character responses.
- **Docs:** finalize `06-training-pipeline.md` (corpus grammar, acceptance metrics explained), `milestones/m4.md`.

### M5 — Performance pass *(ROM v1.0-rc, ≥30 chars/sec)*
- On-screen chars/sec + frame time via libdragon timers. Optimize matvec (weight layout for cache lines, unrolling, int16×int8); optionally bump core to `-O2` via `Makefile.custom` per-object flags. Regression gate: goldens still byte-exact. RSP matvec = documented stretch goal, out of scope.
- **Docs:** `07-performance.md` (measurements, wins *and* dead ends), `milestones/m5.md`.

### M6 — Real hardware *(ROM v1.0 on the user's N64 via EverDrive)*
- EverDrive run: boot, `SELFTEST PASS` photo/video, demo run cycling conditions. v1.0 requires the real console; everything before gates on Ares only.
- **Docs:** `hardware-checklist.md` (EverDrive from zero), `milestones/m6.md`, README final pass (project retrospective + full doc index).

## Verification (every milestone, one loop)
1. `cd trainer && uv run pytest` (from M2 on)
2. `cmake -B build tests && ctest` — byte-identical golden generation, ASan/UBSan
3. `pyrite64 --cli --cmd build game/project.p64proj` → boot `.z64` in Ares → **SELFTEST PASS on screen**
4. Manual demo check.
5. Write/update the milestone's docs (concept guides + `milestones/mN.md`, screenshot of the running ROM); update README checklist; commit + tag `mN`. **A milestone without its docs is not done.**

## Risks / notes
- Pyrite64 is early-dev with breaking changes; the mac fork lags upstream (v0.4.0 vs v0.7.0). Pin the fork tag, vendor the version in README, don't chase upstream mid-project. If a needed upstream feature is missing from the fork, the jam25 dialog pattern is plain libdragon `rdpq_*` and works on any version.
- Symlink into `src/user` is verified against `scanDirs` semantics but not battle-tested in the editor GUI; fallback is a pre-build rsync in `Makefile.custom` (one-line change, no repo restructure).
- PyTorch wheels on Python 3.14: pin 3.12 via `uv python pin 3.12`.
- 100K-param quality: mitigated by rigid corpus format + low temperature; M3 proves conditioning before M4 scales it.
- Toolchain install (M0.2) is the one user-attended step; everything after is scriptable/headless.
