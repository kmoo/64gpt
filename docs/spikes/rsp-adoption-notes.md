# RSP adoption notes — lifting the spike into M6.1 (file-level guide)

For the session building `docs/milestones/m6.1.md` (on main). This
branch is the working reference; **copy from it, don't merge it.**

## Take these

- **`game/src/user/rsp_ngpt.S`** — keep `DotRow` and `NgptMatvec`
  verbatim (proven exact). `NgptEcho`/`NgptDot` + the debug dump are
  spike scaffolding: keep temporarily for bring-up, delete before tag.
  For H≥256 later: `DotRow` is written for 128 columns (8 unpack
  chunks, 8+8 MAC quads, `lqv ...,128,t5` / `...,256,t8` offsets) —
  generalize the three loop counts and stream offsets from a DMEM
  header the C side writes, or emit per-H code.
- **`game/Makefile.custom`** — the one-line ELF hook
  (`$(BUILD_DIR)/$(ROM_NAME).elf: $(BUILD_DIR)/src/user/rsp_ngpt.o`).
  n64.mk auto-assembles any `rsp*.S`; the filename prefix matters.
- **From `DialogueDemo.cpp`** — the *patterns*, not the harness:
  `DEFINE_RSP_UCODE` + one-time `rspq_overlay_register`, `rspq_write`
  cmd 2 with `PhysicalAddr`s, syncpoint polling, and the buffer rules
  (aligned(16), 16-multiple sizes, uncached access on the CPU side).

## The C bridge M6.1 needs (new, small)

1. At model load (game-side): copy W_hh out of the blob into a static
   `int8_t wAligned[3*H*H] __attribute__((aligned(16)))` — the blob
   offset is not 8-aligned and DMA shifts unaligned sources (trap #2).
2. Per step, before dispatch: shuffle h into even/odd streams
   (`h[0,2,..], h[1,3,..]` as int16) into an aligned, uncached-written
   buffer (trap #3).
3. Callback registered via the new `ngpt_set_matvec` hook: dispatch
   `NgptMatvec(wAligned_phys, hShuf_phys, out_phys)`, `rspq_wait()`,
   then the engine adds biases. Biases stay CPU-side.
4. Do not let the boot self-test block on the RSP inside `initDelete`
   (trap #1) — move the self-test to the first update() frames or
   register the RSP path after it.

## Numbers to beat / sanity-check against (Ares)

- Spike matvec: RSP 3,022µs vs CPU 8,596µs for 384×128 (2.8×).
- M5 full step (CPU): 9,795µs. Expected M6.1 step: ~4–6ms (matvec
  share replaced; gates/LUT/logits/sampler still CPU).
- If a wrong value appears that no ucode change moves: suspect the
  cache-line trap (#4/report bug 5), not the VU math.

## Verification ladder (same discipline as every milestone)

host suite untouched-green (CPU reference) → clean ROM → Ares SELFTEST
PASS with RSP enabled (the M4 goldens ARE the exactness proof) → boot
CPU-vs-RSP cross-check line → perf overlay number → versions/ +
Desktop → docs → tag `m6.1`.
