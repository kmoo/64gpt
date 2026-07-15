# 00 — N64 primer for software engineers

You know how to write software. You've just never shipped any to a 1996
games console. This page is everything about the Nintendo 64 that this
project actually depends on — no nostalgia, no trivia.

## The machine, as a target platform

| thing | value | why it matters here |
|---|---|---|
| CPU | MIPS R4300i @ 93.75 MHz | our chars/sec budget comes from this clock |
| RAM | 4 MB (8 MB with Expansion Pak) | model + text buffers must be tiny; our whole model is ~100 KB |
| Endianness | **big-endian** | your Mac is little-endian — see below |
| FPU | present but slow, and float math ≠ bit-identical across platforms | our inference uses **zero floats** |
| OS | none — bare metal | there is no filesystem, no processes, no stdout; the SDK provides those |

The "GPU" (the RCP: RSP + RDP coprocessors) matters for real games; our
demo only draws text, so for this project you can ignore it. (An
RSP-accelerated matrix-vector multiply is a documented stretch goal.)

## Bare metal, and why that's less scary than it sounds

An N64 program is a single binary containing *everything*: a bootloader,
a minimal kernel, your code, and your assets. The open-source SDK we use,
[libdragon](https://libdragon.dev), provides the kernel plus friendly
APIs (display, controller input, a read-only filesystem, even `printf`
debugging over USB with a flashcart). You write ordinary C/C++, a
MIPS cross-compiler builds it, and the result is a **ROM**: a `.z64`
file that is, byte for byte, what would be on a cartridge chip.

The read-only filesystem (DFS) is a packed directory appended to the ROM.
At runtime `asset_load("rom:/model.bin", &size)` reads a file out of it
into RAM. That's how our neural net's weights get onto the console.

## Endianness: the trap this project is built to dodge

The N64 CPU is big-endian (most significant byte first); modern x86/ARM
machines are little-endian. If you `fread` a struct or cast a byte
buffer to `uint32_t*`, you get **different numbers on each machine** —
the classic porting landmine.

Our answer is twofold:

1. The model blob is defined as a big-endian byte stream, and the parser
   reads it **byte by byte** (`(b[0]<<24)|(b[1]<<16)|...`). That
   expression yields the same value on every CPU ever made.
2. Inference is pure integer math (int8/int16/int32). Integer add,
   multiply, and shift are exactly specified — unlike floats, where
   rounding can differ between FPUs and compilers.

Together these give the property the whole TDD strategy leans on:
**the same input bytes produce the same output bytes on your Mac, in the
emulator, and on the console.** A test that passes on the host is a
proof about the N64, not a hope.

## Emulator vs real hardware

We test on [Ares](https://ares-emu.net) (v147+), a *hardware-accurate*
emulator — it models the console's actual behavior rather than
approximating "well enough to play Mario". Pyrite64's own docs name Ares
and gopher64 as the reference emulators.

For the real console we use an **EverDrive-64**: a flashcart that takes
an SD card full of `.z64` files and presents them to the N64 as if each
were a genuine cartridge. Workflow (see `docs/hardware-checklist.md`,
written at M6): copy ROM to SD card, insert, power on, pick it from the
menu.

Per-milestone gate = Ares. Real-console runs are occasional spot-checks
(recommended at M2, required for v1.0) — that's enough, because of the
bit-exactness property above.

## Why a 100K-parameter model fits

Back-of-envelope, so you can re-derive it:

- 99K int8 weights ≈ **100 KB** — 2.5% of the 4 MB RAM.
- One generated character ≈ one GRU step ≈ **~99K multiply-accumulates**.
- The R4300i does an integer MAC in a few cycles; even assuming a
  pessimistic ~30 cycles/MAC average (cache misses included), that's
  ~3M cycles/char → ~30 chars/sec on a 93.75 MHz core. Readable dialogue
  streams at that speed. M5 measures and tunes the real number.
