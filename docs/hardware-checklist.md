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

Keep one card folder per milestone (`/64gpt/m2/` … `/64gpt/m5/`): the
same demo getting a progressively more real brain — canned line → one
memorized line → 12 prompted lines → fresh sampled lines → fast fresh
sampled lines — is the walking-skeleton story told in cartridge form.
(Current test copies: `m2/m3/m4/m5_64gpt.z64`, built 2026-07-15/16.)
