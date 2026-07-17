/**
 * 64GPT dialogue demo — Pyrite64 object script.
 *
 * Attach to any (empty) object via a "Code" component in the editor.
 * initDelete: loads the model blob from the ROM filesystem — nothing
 *             else. The self-test must NOT run here: with the RSP
 *             matvec registered it would block on the RSP before the
 *             first frame, which hangs the boot with a black screen
 *             (docs/spikes/rsp-matvec.md, bug #1).
 * update    : first runs the boot sequence over a few frames — CPU vs
 *             RSP cross-check, then the 12-golden self-test through
 *             the RSP path (one golden per frame, progress on screen) —
 *             then streams a few characters per frame. Controls:
 *             D-pad up/down = NPC, left/right = MOOD, C-left/right = EV,
 *             A = regenerate with the current prompt.
 * draw      : prompt line + dialogue box + SELFTEST PASS/FAIL banner,
 *             drawn with the engine's builtin debug font (uppercase only).
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <rsp.h>
#include <rspq.h>
#include "script/userScript.h"
#include "debug/debugDraw.h"
#include "n64gpt/ngpt.h"
#include "selftestGolden.h"
#include "EventBus.h"
#include "WorldState.h"
#include "NPCDatabase.h"
#include "ContextBuilder.h"

// ---- M6.1 RSP MATVEC BACKEND -------------------------------------------
// The engine's ngpt_set_matvec hook, backed by the rsp_ngpt overlay
// (game/src/user/rsp_ngpt.S — the spike-proven kernel). Buffer rules
// from the spike's bug ledger (docs/spikes/rsp-matvec.md): every
// CPU/RSP-shared buffer owns its cache lines — aligned(16), size a
// multiple of 16, and the CPU side accesses it uncached, or dirty-line
// writebacks overwrite the RSP's DMA results with stale bytes.
DEFINE_RSP_UCODE(rsp_ngpt);

namespace
{
  // W_hh copied out of the blob once at init: the blob offset is not
  // 8-aligned and DMA from an unaligned RDRAM source lands shifted in
  // DMEM. Written uncached so RDRAM is coherent before the RSP reads it.
  int8_t rspWhh[3 * 128 * 128] __attribute__((aligned(16)));   // 48 KB
  int16_t rspHShuf[128] __attribute__((aligned(16)));          // 256 B
  int32_t rspMvOut[384] __attribute__((aligned(16)));          // 1536 B
  const uint8_t *rspWhhSrc{};   // the blob W_hh this copy mirrors
  bool rspReady{};
  uint32_t rspOvlId{};

  void rspBackendInit(const ngpt_model *m)
  {
    if(rspReady)return;
    if(m->gru.H != 128)return; // the kernel is written for 128 columns
    volatile int8_t *dst = (volatile int8_t *)UncachedAddr(rspWhh);
    for(uint32_t i = 0; i < 3u * 128 * 128; ++i)dst[i] = (int8_t)m->gru.w_hh[i];
    rspWhhSrc = m->gru.w_hh;
    rspOvlId = rspq_overlay_register(&rsp_ngpt);
    rspReady = true;
  }

  // ngpt_matvec_fn: raw row.h sums for the whole W_hh matvec. Shapes or
  // backings the kernel can't take fall back to the exact CPU loop —
  // that check is what keeps this correct when H grows past 128 before
  // the ucode learns to tile wider rows.
  void rspMatvec(const uint8_t *w, uint32_t rows, uint32_t cols,
                 const int16_t *h, int32_t *out)
  {
    if(!rspReady || w != rspWhhSrc || rows != 384 || cols != 128) {
      for(uint32_t i = 0; i < rows; ++i) {
        int32_t sum = 0;
        const uint8_t *row = w + i * cols;
        for(uint32_t j = 0; j < cols; ++j)sum += (int32_t)(int8_t)row[j] * h[j];
        out[i] = sum;
      }
      return;
    }
    // h shuffled even-indices-first (h[0,2,..126], then h[1,3,..127])
    // to match the ucode's lqv byte-pair unpack; dot products are
    // order-invariant so the sums are unchanged.
    volatile int16_t *hs = (volatile int16_t *)UncachedAddr(rspHShuf);
    for(uint32_t j = 0; j < 128; ++j)hs[(j & 1) ? 64 + j / 2 : j / 2] = h[j];
    volatile int32_t *mo = (volatile int32_t *)UncachedAddr(rspMvOut);
    rspq_write(rspOvlId, 0, 0, PhysicalAddr(rspWhh), PhysicalAddr(rspHShuf),
               PhysicalAddr(rspMvOut));
    rspq_wait(); // never reached from initDelete — see boot sequence below
    for(uint32_t i = 0; i < 384; ++i)out[i] = mo[i];
  }
}
// ---- M6.1 RSP MATVEC BACKEND (END) --------------------------------------

namespace
{
  // M6.1: one step ~4-6ms with the matvec on the RSP (was ~9.9ms all-CPU
  // in M5); one char/frame streams 60 chars/sec with VPS held at 60.
  constexpr int CHARS_PER_FRAME = 1;
  constexpr int WRAP_COLS = 34;      // 7px glyph advance, ~240px text area

  // One demo instance per scene; shared state keeps P64_DATA trivial.
  uint8_t *blobData{};
  ngpt_model model{};
  ngpt_ctx ctx{};
  char text[512]{};
  char prompt[64]{};
  uint32_t textLen{};
  int trustTier{}, moodIdx{}, contextIdx{};
  bool generating{};
  bool loaded{};
  bool selftestPass{};

  // ---- M6.1 boot sequence ----------------------------------------------
  // Runs from update() once the scene is up (the RSP path blocks, so it
  // can never run in initDelete): CPU reference generation -> same
  // generation through the RSP (h-state + bytes must match) -> all 12
  // goldens through the RSP path, one per frame so progress is visible.
  enum BootPhase { BOOT_WAIT, BOOT_XCHK_CPU, BOOT_XCHK_RSP, BOOT_SELFTEST,
                   BOOT_READY };
  BootPhase bootPhase = BOOT_WAIT;
  uint32_t selftestIdx{};
  bool xchkPass{};
  uint8_t xchkCpuText[512];
  int16_t xchkCpuH[NGPT_GRU_MAX_HIDDEN];
  uint32_t xchkCpuLen{};
  uint32_t cpuStepUs{}, rspStepUs{};  // boot-measured, same generation
  // ------------------------------------------------------------------------

  // ---- TEMP ATTRACT MODE (BEGIN) --------------------------------------
  // Headless-verification aid: after IDLE_START secs without input,
  // auto-advance to the next prompt combo every IDLE_STEP secs; any
  // button press takes control back. TO REMOVE: set the define to 0 or
  // delete the three fenced blocks (search: TEMP ATTRACT MODE).
  #define NGPT_ATTRACT_MODE 1
  #if NGPT_ATTRACT_MODE
  constexpr float IDLE_START = 8.0f;
  constexpr float IDLE_STEP = 3.0f;
  float idleTime{};
  float attractTimer{};
  bool attract{};
  #endif
  // ---- TEMP ATTRACT MODE (END) ----------------------------------------

  // M7: the demo's own trust/mood/context cycling drives the Context
  // Builder, which emits the schema string the frozen ngpt_reset(prompt)
  // API primes on (docs/milestones/m7.md "conditioning contract"). The
  // event field stays "none" for interactive cycling — the trained
  // model's per-axis behavior on identity/mood/trust/context is what the
  // trainer's acceptance gates (make_m7_blob.py) actually measure; this
  // scene exists to show the live mechanism working, not to re-run eval.
  void buildPrompt()
  {
    NPCDatabase::selena.trustTier = trustTier;
    NPCDatabase::selena.moodIdx = moodIdx;
    WorldState::setContext(NPCDatabase::CONTEXTS[contextIdx]);
    ContextBuilder::build(prompt, sizeof(prompt), NPCDatabase::selena,
                          WorldState::currentContext(), EventBus::lastTag());
  }

  uint32_t frameCount{}; // demo sampling seed: varies per regenerate

  // Perf instrumentation: EMA of CPU ticks per ngpt_step during live
  // streaming (with the RSP path enabled this IS the RSP number), shown
  // as us/step + raw chars/sec, next to the boot-measured CPU baseline.
  uint64_t stepTicksEma{};
  bool stepMeasured{};

  void restartGeneration()
  {
    textLen = 0;
    generating = loaded;
    buildPrompt();
    if(loaded) {
      ngpt_reset(&ctx, &model, prompt);
      /* M4: sampled generation. The seed comes from the frame counter,
       * so every regenerate (even of the same prompt) speaks a fresh
       * line; the self-test below uses the pinned seed instead. */
      ngpt_set_sampler(&ctx, frameCount, SELFTEST_INV_T_Q8, SELFTEST_TOP_K);
    }
  }

  // One full pinned-seed generation of prompt 0; captures the bytes and
  // the final hidden state, and times it. The same call runs once with
  // the matvec hook off (CPU reference) and once with the RSP path.
  uint32_t runXchkGeneration(uint8_t *outText, uint32_t cap, int16_t *outH,
                             uint32_t *avgStepUs)
  {
    ngpt_reset(&ctx, &model, SELFTEST_PROMPTS[0]);
    ngpt_set_sampler(&ctx, SELFTEST_SAMPLE_SEED, SELFTEST_INV_T_Q8,
                     SELFTEST_TOP_K);
    uint32_t n = 0, steps = 0;
    int c;
    uint64_t t0 = get_ticks();
    while((c = ngpt_step(&ctx)) != NGPT_EOS) {
      if(n < cap)outText[n++] = (uint8_t)c;
      if(++steps > 2000)break; // runaway guard: no debugger on a ROM
    }
    ++steps; // the EOS step does a full h-update too
    *avgStepUs = (uint32_t)(TICKS_TO_US(get_ticks() - t0) / steps);
    memcpy(outH, ctx.h, sizeof(ctx.h));
    return n;
  }

  // Replay golden p (seeded sampled generation vs the committed bytes).
  bool runSelfTestOne(uint32_t p)
  {
    ngpt_reset(&ctx, &model, SELFTEST_PROMPTS[p]);
    ngpt_set_sampler(&ctx, SELFTEST_SAMPLE_SEED, SELFTEST_INV_T_Q8,
                     SELFTEST_TOP_K);
    const char *want = SELFTEST_GOLDEN[p];
    uint32_t i = 0;
    int c;
    while((c = ngpt_step(&ctx)) != NGPT_EOS) {
      if(want[i] == '\0')return false;
      if((uint8_t)c != (uint8_t)want[i])return false;
      ++i;
    }
    return want[i] == '\0';
  }

  // Advances one boot phase per call (== per frame, so every phase's
  // status is drawn before the next blocks the frame for a moment).
  void bootAdvance()
  {
    switch(bootPhase) {
      case BOOT_WAIT:
        if(!loaded) { bootPhase = BOOT_READY; restartGeneration(); return; }
        if(frameCount < 30)return; // let the scene settle first
        bootPhase = BOOT_XCHK_CPU;
        return;

      case BOOT_XCHK_CPU: // reference run, hook off: the M5 CPU path
        ngpt_set_matvec(nullptr);
        xchkCpuLen = runXchkGeneration(xchkCpuText, sizeof(xchkCpuText),
                                       xchkCpuH, &cpuStepUs);
        bootPhase = BOOT_XCHK_RSP;
        return;

      case BOOT_XCHK_RSP: { // same generation through the RSP
        rspBackendInit(&model);
        if(rspReady)ngpt_set_matvec(rspMatvec);
        static uint8_t rspText[512];
        static int16_t rspH[NGPT_GRU_MAX_HIDDEN];
        uint32_t len = runXchkGeneration(rspText, sizeof(rspText), rspH,
                                         &rspStepUs);
        xchkPass = len == xchkCpuLen &&
                   memcmp(rspText, xchkCpuText, len) == 0 &&
                   memcmp(rspH, xchkCpuH, sizeof(rspH)) == 0;
        selftestIdx = 0;
        bootPhase = BOOT_SELFTEST;
        return;
      }

      case BOOT_SELFTEST: // the M4 goldens, replayed through the RSP path
        if(!runSelfTestOne(selftestIdx)) {
          selftestPass = false;
          bootPhase = BOOT_READY;
          restartGeneration();
          return;
        }
        if(++selftestIdx >= SELFTEST_COUNT) {
          selftestPass = true;
          bootPhase = BOOT_READY;
          restartGeneration();
        }
        return;

      case BOOT_READY:
        return;
    }
  }
}

namespace P64::Script::C64D1A106DE00001
{
  P64_DATA();

  /* v0.4.0 lifecycle: one hook for both spawn (isDelete=false) and
   * teardown (isDelete=true) — the scanner only recognizes
   * initDelete/update/draw/onEvent/onCollision (src/build/scriptBuilder.cpp). */
  void initDelete(Object& obj, Data *data, bool isDelete)
  {
    if(isDelete) {
      if(blobData) {
        free(blobData);
        blobData = nullptr;
      }
      ngpt_set_matvec(nullptr);
      loaded = false;
      return;
    }

    int blobSize = 0;
    blobData = (uint8_t*)asset_load("rom:/model.bin", &blobSize);
    loaded = blobData && ngpt_load(&model, blobData, (uint32_t)blobSize) == NGPT_OK;
    // Self-test + RSP registration happen in the first update() frames
    // (bootAdvance) — blocking on the RSP here would hang the boot.
  }

  static int cycle(int idx, int delta, int count)
  {
    idx += delta;
    if(idx < 0)return count - 1;
    if(idx >= count)return 0;
    return idx;
  }

  void update(Object& obj, Data *data, float deltaTime)
  {
    ++frameCount;
    if(bootPhase != BOOT_READY) {
      bootAdvance();
      return;
    }
    auto pressed = joypad_get_buttons_pressed(JOYPAD_PORT_1);
    bool changed = false;
    if(pressed.d_up)   { trustTier  = cycle(trustTier,  +1, 3); changed = true; }
    if(pressed.d_down) { trustTier  = cycle(trustTier,  -1, 3); changed = true; }
    if(pressed.d_right){ moodIdx    = cycle(moodIdx, +1, NPCDatabase::MOOD_COUNT); changed = true; }
    if(pressed.d_left) { moodIdx    = cycle(moodIdx, -1, NPCDatabase::MOOD_COUNT); changed = true; }
    if(pressed.c_right){ contextIdx = cycle(contextIdx, +1, NPCDatabase::CONTEXT_COUNT); changed = true; }
    if(pressed.c_left) { contextIdx = cycle(contextIdx, -1, NPCDatabase::CONTEXT_COUNT); changed = true; }
    if(pressed.a || changed)restartGeneration();

    // ---- TEMP ATTRACT MODE (BEGIN) ------------------------------------
    #if NGPT_ATTRACT_MODE
    bool anyInput = pressed.d_up || pressed.d_down || pressed.d_left ||
                    pressed.d_right || pressed.c_left || pressed.c_right ||
                    pressed.a || pressed.b || pressed.start;
    if(anyInput) {
      idleTime = 0.0f;
      attract = false;
    } else {
      idleTime += deltaTime;
      if(!attract && idleTime >= IDLE_START) {
        attract = true;
        attractTimer = IDLE_STEP; /* advance immediately on entry */
      }
    }
    if(attract) {
      attractTimer += deltaTime;
      if(attractTimer >= IDLE_STEP) {
        attractTimer = 0.0f;
        /* nested odometer: CONTEXT fastest, then MOOD, then TRUST TIER */
        contextIdx = cycle(contextIdx, +1, NPCDatabase::CONTEXT_COUNT);
        if(contextIdx == 0) {
          moodIdx = cycle(moodIdx, +1, NPCDatabase::MOOD_COUNT);
          if(moodIdx == 0)trustTier = cycle(trustTier, +1, 3);
        }
        restartGeneration();
      }
    }
    #endif
    // ---- TEMP ATTRACT MODE (END) --------------------------------------

    for(int i = 0; i < CHARS_PER_FRAME && generating; ++i) {
      uint64_t t0 = get_ticks();
      int c = ngpt_step(&ctx);
      uint64_t dt = get_ticks() - t0;
      stepTicksEma = stepMeasured ? (stepTicksEma * 7 + dt) / 8 : dt;
      stepMeasured = true;
      if(c == NGPT_EOS) {
        generating = false;
        break;
      }
      if(textLen < sizeof(text) - 1)text[textLen++] = (char)c;
    }
  }

  void draw(Object& obj, Data *data, float deltaTime)
  {
    DrawLayer::use2D();
      Debug::printStart();

      char line[64];
      if(bootPhase != BOOT_READY) {
        const char *phase =
          bootPhase == BOOT_WAIT     ? "BOOT" :
          bootPhase == BOOT_XCHK_CPU ? "XCHK CPU REF" :
          bootPhase == BOOT_XCHK_RSP ? "XCHK RSP" : "GOLDENS";
        snprintf(line, sizeof(line), "SELFTEST RUNNING %lu/%lu %s",
                 (unsigned long)selftestIdx, (unsigned long)SELFTEST_COUNT,
                 phase);
        Debug::print(24, 24, line);
      } else {
        Debug::print(24, 24, selftestPass ? "SELFTEST PASS" : "SELFTEST FAIL");
      }
      Debug::print(24, 40, "64GPT V1.2 - SELENA (M7)");
      Debug::print(24, 60, prompt);

      if(stepMeasured) {
        uint32_t us = (uint32_t)TICKS_TO_US(stepTicksEma);
        snprintf(line, sizeof(line), "STEP %lu US  RAW %lu CH/S",
                 (unsigned long)us, (unsigned long)(us ? 1000000u / us : 0));
        Debug::print(24, 80, line);
      }
      if(bootPhase == BOOT_READY && loaded) {
        snprintf(line, sizeof(line), "%s XCHK %s  CPU %lu RSP %lu US",
                 rspReady ? "RSP ON" : "RSP OFF",
                 xchkPass ? "PASS" : "FAIL",
                 (unsigned long)cpuStepUs, (unsigned long)rspStepUs);
        Debug::print(24, 96, line);
      }

      // dialogue box: wrap the streamed text into rows
      char row[WRAP_COLS + 1];
      uint32_t pos = 0;
      int y = 120;
      while(pos < textLen && y < 190) {
        uint32_t n = 0;
        while(n < WRAP_COLS && pos < textLen)row[n++] = text[pos++];
        row[n] = '\0';
        Debug::print(36, y, row);
        y += 10;
      }
      if(generating)Debug::print(36, y, ">"); // cursor while streaming

      // ---- TEMP ATTRACT MODE (BEGIN/END: one line) ----------------------
      #if NGPT_ATTRACT_MODE
      Debug::print(24, 200, attract ? "AUTO CYCLE - PRESS ANY BUTTON"
                                    : "DPAD TRUST/MOOD  C CTX  A REGEN");
      #else
      Debug::print(24, 200, "DPAD TRUST/MOOD  C CTX  A REGEN");
      #endif
    DrawLayer::useDefault();
  }
}
