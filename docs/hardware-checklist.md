# Hardware checklist — running 64GPT on a real N64 (M6)

*(Audience: someone who owns an N64 and an EverDrive-64 but has never
put homebrew on either. Everything before M6 gates on the Ares emulator;
v1.0 requires this document to have been executed on real silicon.)*

## What you need

- An N64 console (any region; the ROM is NTSC-timed — on PAL consoles
  expect 50Hz letterboxing, the self-test still applies)
- An EverDrive-64 flash cartridge (X7 or X5; older v2.5+ also works)
- A microSD card, 32GB or smaller is simplest
- A controller in port 1 (the demo reads buttons from port 1 only)
- A TV with composite/S-Video in, or an upscaler

## SD card prep (one time)

1. Format the card **FAT32**. On macOS: Disk Utility → Erase →
   MS-DOS (FAT). exFAT also works on X7; FAT32 is the safe answer.
2. Download the EverDrive OS (`ED64` folder) from the krikzz site and
   copy it to the card root — the cart won't boot menus without it.
3. Make a folder for the ROMs, e.g. `/64gpt/`.

## Putting a build on the card

1. Build (or take a Desktop copy of) the ROM: `game/64gpt.z64`.
   Milestone builds are the tagged ones — `git checkout m5 && <build>` —
   and when switching tags **delete `game/build/` AND
   `game/filesystem/`** before building (stale-blob gotcha,
   `docs/milestones/m5.md`).
2. Copy the `.z64` into `/64gpt/` on the card. No conversion, no header
   fiddling — EverDrive takes big-endian `.z64` natively.
3. Eject cleanly.

## Boot steps

1. Card in cart, cart in console, controller in port 1, power on.
2. In the EverDrive menu: navigate to `/64gpt/`, select the ROM, press
   A (start on X7 menus: follow the prompt).
3. **Expect ~10 seconds of black screen.** The ROM runs its boot
   self-test first — ~1,000 GRU inference steps replaying the committed
   goldens — before the first frame. Do not power-cycle; the console is
   not frozen, it is thinking.

## What success looks like

- Top line: **`SELFTEST PASS`** — the on-console generation is
  byte-identical to what the Python trainer and the Mac test suite
  produced. This single line is the project's whole claim, verified on
  the real chip.
- A prompt line (`NPC=… MOOD=… EV=…`), a perf line (M5+ builds:
  `STEP <n> US  RAW <n> CH/S`), and text streaming into the dialogue
  box at ~60 chars/sec.
- Controls: D-pad up/down NPC, left/right MOOD, C-left/right EVENT,
  A regenerates (M4+ builds sample a fresh line each press). Idle for
  8s and it auto-cycles.
- **Record for the log:** photograph/film the SELFTEST PASS screen and
  the perf line — the M6 gate artifact — and note the real-hardware
  `STEP` µs next to Ares' 9,795 (they should be close; Ares is
  cycle-oriented but not cycle-perfect).

## If it fails

| symptom | likely cause | fix |
|---|---|---|
| `SELFTEST FAIL` | ROM built from mixed milestone state | clean rebuild: purge `game/build/` + `game/filesystem/`, rebuild, re-copy |
| Black screen forever (>60s) | corrupted copy or wrong file | re-copy the `.z64`; verify its size matches the built one exactly |
| EverDrive menu doesn't list it | missing `ED64` OS folder or exFAT on old cart | re-do SD prep with FAT32 |
| Garbled text, PASS still shown | TV/upscaler mangling 240p | try composite direct to a CRT first, then blame the upscaler |
| No response to buttons | controller in port 2 | move to port 1 |

## Milestone ROM lineup (for A/B demos)

Keep one card folder per milestone (`/64gpt/m2/` … `/64gpt/m7/`): the
same demo getting a progressively more real brain — canned line → one
memorized line → 12 prompted lines → fresh sampled lines → fast fresh
sampled lines → H=256 RSP-accelerated Selena — is the walking-skeleton
story told in cartridge form. **Outstanding as of 2026-07-17: nothing
past M5 has actually been run on real hardware yet** — M6/v1.0 requires
it, M7's own docs note it as "occasional spot-check... not required for
this milestone's DoD," which is why it kept sliding. The pressure-test
table below is the next real-hardware session's actual job, not a
future nice-to-have.

## Pressure-test table: emulator vs. physical (fill in the right column)

Two different questions, both worth real silicon numbers: does Ares'
timing match reality closely enough to trust ("hardware-accurate" is a
claim, not a given), and where do the architecture's actual limits sit
(DMEM, RDRAM, ROM size) versus where we've only pushed it so far.

| metric | Ares (measured 2026-07-17) | Physical N64 (fill in) |
|---|---|---|
| Steady-state gen speed, H=256 RSP ON | 63 ch/s raw | |
| Per-step latency (`STEP` line) | ~15,700us | |
| RSP vs CPU speedup ratio | 2.49x (CPU 38,417us / RSP 15,428us) | |
| Boot + SELFTEST time (18 goldens through RSP) | not precisely timed yet — ballpark <60s, worth a stopwatch pass on either platform | |
| ROM ↔ real-console load/boot latency | n/a (Ares loads instantly) | first real EverDrive-specific number |

**Fixed specs (same on any platform, not an Ares-vs-physical question —
included here as the actual limits, not yet pressure-tested past current
usage):**

| metric | current (H=256) | headroom / limit |
|---|---|---|
| RSP DMEM (data+bss) | 3,656 / 4,096 B | 440B slack at H=256's 8-row tiling — untested whether a *practically*-tiled (not 1-row) kernel could go past H=256 at all without redesign |
| RSP IMEM (text) | 1,152 / 4,096 B | plenty of room, not the binding constraint |
| Theoretical max H (1-row tiles, pure DMEM math, unverified) | — | ~700, but 1-row tiling would be DMA-transfer-bound, likely far slower than useful — **untested, don't trust this number without actually building it** |
| ROM file size | 704,512 B | EverDrive carts are large; not remotely the binding constraint yet |
| Model blob (`game/rawfs/model.bin`) | 268,621 B | RDRAM is 4MB total, shared with everything else the engine needs — untested how many archetype-instance blobs (M8) this budget actually supports |

**What "pressure test" should mean when you actually run this:** not
just confirming H=256 works (already proven, see `docs/milestones/
m7.md`) — deliberately push past it. Retrain a throwaway H=384 or H=512
blob (no character bible needed, same discipline as the identity spike's
throwaway identities), see whether the *current* 8-row-tile kernel still
fits DMEM at all, and if not, at what H it stops fitting. That number,
not the untested ~700 estimate above, is what M8's capacity planning
should actually be budgeted against.
