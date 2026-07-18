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
 *             D-pad up/down = TRUST, left/right = MOOD, C-left/right = EV,
 *             START = switch NPC (Selena, then the 4 guard archetype
 *             instances, M8 task #11), A = regenerate with the current
 *             prompt, B = show more of the current line once it
 *             overflows one page (companion lines routinely run longer
 *             than the 7-row dialogue box).
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
#include "NpcService.h"
#include "SaveData.h"

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
  // H=320 generalization (M9, spike: rsp-matvec-h256 -> this): sizes
  // scale the same way the H=128->256 step did -- W_hh is 3H x H so
  // it's 4x, not 2x, at 320/256 = 1.25x the hidden size (960*320 =
  // 3H*H). rspHShuf/rspMvOut scale linearly with H/3H respectively.
  constexpr int RSP_H = 320;
  int8_t rspWhh[3 * RSP_H * RSP_H] __attribute__((aligned(16)));   // 300 KB
  int16_t rspHShuf[RSP_H] __attribute__((aligned(16)));            // 640 B
  int32_t rspMvOut[3 * RSP_H] __attribute__((aligned(16)));        // 3840 B
  const uint8_t *rspWhhSrc{};   // the blob W_hh this copy mirrors
  bool rspReady{};
  uint32_t rspOvlId{};

  void rspBackendInit(const ngpt_model *m)
  {
    if(rspReady)return;
    if(m->gru.H != RSP_H)return; // the kernel is written for RSP_H columns
    volatile int8_t *dst = (volatile int8_t *)UncachedAddr(rspWhh);
    for(uint32_t i = 0; i < 3u * RSP_H * RSP_H; ++i)dst[i] = (int8_t)m->gru.w_hh[i];
    rspWhhSrc = m->gru.w_hh;
    rspOvlId = rspq_overlay_register(&rsp_ngpt);
    rspReady = true;
  }

  // ngpt_matvec_fn: raw row.h sums for the whole W_hh matvec. Shapes or
  // backings the kernel can't take fall back to the exact CPU loop —
  // that check is what keeps this correct if H ever changes again
  // before the ucode learns to tile arbitrary widths.
  void rspMatvec(const uint8_t *w, uint32_t rows, uint32_t cols,
                 const int16_t *h, int32_t *out)
  {
    if(!rspReady || w != rspWhhSrc || rows != 3u * RSP_H || cols != RSP_H) {
      for(uint32_t i = 0; i < rows; ++i) {
        int32_t sum = 0;
        const uint8_t *row = w + i * cols;
        for(uint32_t j = 0; j < cols; ++j)sum += (int32_t)(int8_t)row[j] * h[j];
        out[i] = sum;
      }
      return;
    }
    // h shuffled even-indices-first (h[0,2,..318], then h[1,3,..319])
    // to match the ucode's lqv byte-pair unpack; dot products are
    // order-invariant so the sums are unchanged. Odd half starts at
    // RSP_H/2 (== new even-half element count), same relationship the
    // H=256 kernel had at 128 -- see rsp_ngpt.S's file header.
    volatile int16_t *hs = (volatile int16_t *)UncachedAddr(rspHShuf);
    for(uint32_t j = 0; j < (uint32_t)RSP_H; ++j)
      hs[(j & 1) ? RSP_H / 2 + j / 2 : j / 2] = h[j];
    volatile int32_t *mo = (volatile int32_t *)UncachedAddr(rspMvOut);
    rspq_write(rspOvlId, 0, 0, PhysicalAddr(rspWhh), PhysicalAddr(rspHShuf),
               PhysicalAddr(rspMvOut));
    rspq_wait(); // never reached from initDelete — see boot sequence below
    for(uint32_t i = 0; i < 3u * RSP_H; ++i)out[i] = mo[i];
  }
}
// ---- M6.1 RSP MATVEC BACKEND (END) --------------------------------------

namespace
{
  // M6.1: one step ~4-6ms with the matvec on the RSP (was ~9.9ms all-CPU
  // in M5); one char/frame streams 60 chars/sec with VPS held at 60.
  constexpr int CHARS_PER_FRAME = 1;
  constexpr int WRAP_COLS = 34;      // 7px glyph advance, ~240px text area
  constexpr int TEXT_ROW_Y0 = 130;  // +10 vs. before: the prompt line now reserves 2 rows
  constexpr int TEXT_ROW_DY = 10;
  constexpr int TEXT_ROWS_PER_PAGE = 6;  // (190-130)/10: fits above the controls row
  // Companion opener+body+closer lines (M7) routinely exceed one page —
  // this used to just stop drawing at the bottom of the box and silently
  // drop the rest of the generated text. text[]/textLen still hold the
  // FULL line either way; pageStart only windows what draw() shows.
  constexpr int TEXT_CHARS_PER_PAGE = WRAP_COLS * TEXT_ROWS_PER_PAGE;

  // One demo instance per scene; shared state keeps P64_DATA trivial.
  uint8_t *blobData{};
  ngpt_model model{};
  ngpt_ctx ctx{};
  char text[512]{};
  // >=96 bytes per NpcService::buildPromptFields()'s contract (the new
  // P:/D:/OCC:/R:/M:/C:/EV: schema runs longer than ContextBuilder's old
  // N:/TR: one -- e.g. Fergus + a long context/event can reach ~105
  // chars); 64 was sized for the old schema only and would have silently
  // truncated the actual conditioning string for the new cast (M9).
  char prompt[128]{};
  uint32_t textLen{};
  uint32_t pageStart{};  // draw() windows text[pageStart..] one page at a time
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
  // Headless-verification aid: after IDLE_START secs without input, auto-
  // cycle to the next prompt combo; any button press takes control back.
  // Paced on actual completion, not a blind timer: waits for the current
  // line to finish streaming, then holds each page (auto-turning through
  // any that overflow one screen, same "MORE" mechanism B uses manually)
  // for ATTRACT_PAGE_HOLD secs before moving on -- so hands-off playback
  // always shows a full line, never a mid-stream or mid-page slice of
  // one, but still always keeps moving (just paced by the mode, not
  // stuck). TO REMOVE: set the define to 0 or delete the three fenced
  // blocks (search: TEMP ATTRACT MODE).
  #define NGPT_ATTRACT_MODE 1
  #if NGPT_ATTRACT_MODE
  constexpr float IDLE_START = 8.0f;
  constexpr float ATTRACT_PAGE_HOLD = 2.0f;
  float idleTime{};
  float attractHoldTimer{};
  bool attract{};
  #endif
  // ---- TEMP ATTRACT MODE (END) ----------------------------------------

  // M9: the curated compositional cast (trainer/ngpt_trainer/cast_corpus.py
  // CHARACTERS, byte-for-byte the same traits/occupation/age/gender that
  // were actually trained) -- conditioned via NpcService::buildPromptFields()
  // (P:/D:/OCC:/R:/M:/C:/EV:), not ContextBuilder's old N:/TR: schema.
  // "BRAM" deliberately reuses one of M8's own GUARD_NAMES (docs/milestones/
  // m9.md section 4): the same character, speaking through both schemas,
  // demonstrating the new one works without inventing a fresh identity.
  constexpr int NEW_CAST_COUNT = 3;
  const char *const NEW_CAST_NAMES[NEW_CAST_COUNT] = { "BRAM", "FERGUS", "KRAGAN" };
  const NpcService::Profile NEW_CAST[NEW_CAST_COUNT] = {
    { "guard",     35, NpcService::Gender::Male, {30, 15, 20, 80, 70} },
    { "innkeeper", 62, NpcService::Gender::Male, {80, 75, 50, 50, 40} },
    { "bandit",    45, NpcService::Gender::Male, {20, 15, 40, 55, 60} },
  };

  // M10: Shadewrath (full-tier villain) + Korrath (mid-tier boss) — old
  // N: scheme like Selena, named individuals rather than archetype
  // instances, so no compositional Profile needed here (see
  // NPCDatabase::shadewrath/korrath).
  constexpr int NAMED_EXTRA_COUNT = 2;

  // M10: one showcase instance per new town archetype, spawnInstance()'s
  // REAL output (not a hardcoded NEW_CAST-style Profile) — proves the
  // archetype -> compositional-scheme bridge (NpcService::profileFor())
  // end to end in the shipped demo, not just in host tests.
  constexpr int NEW_ARCHETYPE_COUNT = 4;
  const char *const NEW_ARCHETYPE_LABELS[NEW_ARCHETYPE_COUNT] = {
    "PUB_PATRON", "BLACKSMITH", "WIZARD", "VILLAGER",
  };
  const NPCDatabase::Archetype *const NEW_ARCHETYPES[NEW_ARCHETYPE_COUNT] = {
    &NPCDatabase::PUB_PATRON_ARCHETYPE, &NPCDatabase::BLACKSMITH_ARCHETYPE,
    &NPCDatabase::WIZARD_ARCHETYPE, &NPCDatabase::VILLAGER_ARCHETYPE,
  };
  constexpr uint32_t NEW_ARCHETYPE_SEEDS[NEW_ARCHETYPE_COUNT] = {
    0x2001, 0x2002, 0x2003, 0x2004,
  };
  NPCDatabase::NPC newArchetypeInstances[NEW_ARCHETYPE_COUNT]{};

  void initNewArchetypeInstances()
  {
    for(int i = 0; i < NEW_ARCHETYPE_COUNT; ++i)
      newArchetypeInstances[i] =
          NPCDatabase::spawnInstance(*NEW_ARCHETYPES[i], NEW_ARCHETYPE_SEEDS[i]);
  }

  // M8 task #11 (extended M10): which NPC the demo is currently talking
  // to. Slot layout: 0 = Selena (full character, M7); 1..GUARD_INSTANCE_
  // COUNT = guardInstances[n-1] (archetype instances, M8); next
  // NEW_CAST_COUNT = the M9 compositional cast; next NAMED_EXTRA_COUNT =
  // Shadewrath, Korrath (M10 named individuals, old N: scheme); final
  // NEW_ARCHETYPE_COUNT = the M10 town-archetype showcase instances
  // (compositional scheme). Cycled with START. Context cycling (below)
  // stays universal across all 8 CONTEXTS regardless of NPC: guard's own
  // corpus only trained 3 of them (guard_corpus.py's GUARD_CONTEXTS), so
  // an untrained combo on a guard is an honest demonstration of the
  // archetype's limits, not a bug to special-case away — this demo exists
  // to show the mechanism working, not to re-run the trainer's eval.
  constexpr int NEW_CAST_START = 1 + NPCDatabase::GUARD_INSTANCE_COUNT;
  constexpr int NAMED_EXTRA_START = NEW_CAST_START + NEW_CAST_COUNT;
  constexpr int NEW_ARCHETYPE_START = NAMED_EXTRA_START + NAMED_EXTRA_COUNT;
  constexpr int NPC_SLOT_COUNT = NEW_ARCHETYPE_START + NEW_ARCHETYPE_COUNT;
  int currentNpc{};

  bool isNewCastSlot() { return currentNpc >= NEW_CAST_START && currentNpc < NAMED_EXTRA_START; }
  bool isNamedExtraSlot() { return currentNpc >= NAMED_EXTRA_START && currentNpc < NEW_ARCHETYPE_START; }
  bool isNewArchetypeSlot() { return currentNpc >= NEW_ARCHETYPE_START; }

  NPCDatabase::NPC &activeNpc()
  {
    if(currentNpc == 0)return NPCDatabase::selena;
    if(isNamedExtraSlot())
      return (currentNpc - NAMED_EXTRA_START) == 0 ? NPCDatabase::shadewrath
                                                    : NPCDatabase::korrath;
    if(isNewArchetypeSlot())
      return newArchetypeInstances[currentNpc - NEW_ARCHETYPE_START];
    return NPCDatabase::guardInstances[currentNpc - 1];
  }

  bool isShadewrathSlot() { return isNamedExtraSlot() && (currentNpc - NAMED_EXTRA_START) == 0; }
  bool isKorrathSlot()    { return isNamedExtraSlot() && (currentNpc - NAMED_EXTRA_START) == 1; }

  // M10: entering Shadewrath's or Korrath's slot restores THEIR OWN
  // persisted trust tier (SaveData) instead of carrying over whatever
  // trustTier was left from the previously-selected NPC -- the actual
  // player-visible proof that "the bad guy remembers" (m10.md section
  // 3), not just the isPersistent() rule existing on paper with nothing
  // behind it.
  void loadPersistedTrustTierIfNamedExtra()
  {
    if(isShadewrathSlot())trustTier = SaveData::current.shadewrathHighestTier;
    else if(isKorrathSlot())trustTier = SaveData::current.korrathHighestTier;
  }

  // trustTier (0/1/2, the demo's existing D-pad control) maps onto 3 of
  // NpcService's 6 relationship tiers -- stranger/neutral/best_friend --
  // reusing the existing control scheme rather than adding a 4th D-pad
  // axis just for the new cast. Uniform axes at the tier midpoint, same
  // convention as trainer/ngpt_trainer/m9_corpus.py's _relationship_state().
  NpcService::RelationshipState relationshipForTrustTier(int tier)
  {
    uint16_t v = tier == 0 ? 100 : tier == 1 ? 500 : 975;
    return { v, v, v, v, 0 };
  }

  // M7: the demo's own trust/mood/context cycling drives the Context
  // Builder, which emits the schema string the frozen ngpt_reset(prompt)
  // API primes on (docs/milestones/m7.md "conditioning contract"). The
  // event field stays "none" for interactive cycling — the trained
  // model's per-axis behavior on identity/mood/trust/context is what the
  // trainer's acceptance gates (make_m7_blob.py) actually measure; this
  // scene exists to show the live mechanism working, not to re-run eval.
  void buildPrompt()
  {
    if(isNewCastSlot()) {
      const NpcService::Profile &profile = NEW_CAST[currentNpc - NEW_CAST_START];
      NpcService::RelationshipState rel = relationshipForTrustTier(trustTier);
      NpcService::buildPromptFields(prompt, sizeof(prompt), profile, rel,
                                    NPCDatabase::MOODS[moodIdx],
                                    NPCDatabase::CONTEXTS[contextIdx],
                                    EventBus::lastTag());
      return;
    }
    if(isNewArchetypeSlot()) {
      // M10: real spawnInstance() output routed through the same
      // compositional bridge (profileFor()) any new archetype uses --
      // no bespoke wiring, exactly the point of generalizing the
      // archetype system onto NpcService.
      NpcService::Profile profile = NpcService::profileFor(activeNpc());
      NpcService::RelationshipState rel = relationshipForTrustTier(trustTier);
      NpcService::buildPromptFields(prompt, sizeof(prompt), profile, rel,
                                    NPCDatabase::MOODS[moodIdx],
                                    NPCDatabase::CONTEXTS[contextIdx],
                                    EventBus::lastTag());
      return;
    }
    NPCDatabase::NPC &npc = activeNpc();
    npc.trustTier = trustTier;
    npc.moodIdx = moodIdx;
    WorldState::setContext(NPCDatabase::CONTEXTS[contextIdx]);
    ContextBuilder::build(prompt, sizeof(prompt), npc,
                          WorldState::currentContext(), EventBus::lastTag());
  }

  uint32_t frameCount{}; // boot-settle gate (BOOT_WAIT) + secondary seed mix-in

  // Perf instrumentation: EMA of CPU ticks per ngpt_step during live
  // streaming (with the RSP path enabled this IS the RSP number), shown
  // as us/step + raw chars/sec, next to the boot-measured CPU baseline.
  uint64_t stepTicksEma{};
  bool stepMeasured{};

  void restartGeneration()
  {
    textLen = 0;
    pageStart = 0;
    generating = loaded;
    buildPrompt();
    if(loaded) {
      ngpt_reset(&ctx, &model, prompt);
      /* M4: sampled generation, one fresh seed per LINE (not per
       * character -- ngpt_step()'s own PRNG advances internally after
       * this). frameCount alone (M4-era choice) is too coarse a seed:
       * it only ticks once per ~16.7ms frame, so two regenerates in the
       * same frame -- or attract mode's fixed-interval auto-cycling --
       * could land on identical or near-identical seeds. get_ticks() is
       * libdragon's free-running hardware counter at ~46.875 MHz (half
       * the CPU clock), so the exact tick when a real human presses A is
       * effectively unpredictable even within one frame; frameCount is
       * XORed in too as a cheap second mix-in, harmless either way. The
       * self-test below uses the pinned seed instead, unaffected. */
      ngpt_set_sampler(&ctx, (uint32_t)get_ticks() ^ frameCount,
                       SELFTEST_INV_T_Q8, SELFTEST_TOP_K);
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

    NPCDatabase::initGuardInstances(); // M8 task #11: fixed set, cheap, no ROM/model dependency
    initNewArchetypeInstances(); // M10: same, cheap, no ROM/model dependency
    SaveData::init(); // M10: EEPROM save (game/Makefile.custom advertises
                       // eeprom4k) -- falls back to defaults if no EEPROM
                       // is present, never blocks boot

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
    // M10: raising trust tier on Shadewrath/Korrath persists a new
    // high-water mark immediately -- this IS the save, not a separate
    // deferred step, so there's no window where progress could be lost.
    if(pressed.d_up || pressed.d_down)
    {
      if(isShadewrathSlot())SaveData::recordShadewrathTier((uint8_t)trustTier);
      else if(isKorrathSlot())SaveData::recordKorrathTier((uint8_t)trustTier);
    }
    if(pressed.d_right){ moodIdx    = cycle(moodIdx, +1, NPCDatabase::MOOD_COUNT); changed = true; }
    if(pressed.d_left) { moodIdx    = cycle(moodIdx, -1, NPCDatabase::MOOD_COUNT); changed = true; }
    if(pressed.c_right){ contextIdx = cycle(contextIdx, +1, NPCDatabase::CONTEXT_COUNT); changed = true; }
    if(pressed.c_left) { contextIdx = cycle(contextIdx, -1, NPCDatabase::CONTEXT_COUNT); changed = true; }
    if(pressed.start) {
      currentNpc = cycle(currentNpc, +1, NPC_SLOT_COUNT);
      loadPersistedTrustTierIfNamedExtra(); // M10: restore Shadewrath's/
                                             // Korrath's own remembered
                                             // progress, don't carry over
                                             // the previous NPC's trustTier
      changed = true;
    }
    if(pressed.a || changed)restartGeneration();
    if(pressed.b) {
      uint32_t next = pageStart + TEXT_CHARS_PER_PAGE;
      if(next < textLen)pageStart = next;
    }

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
        attractHoldTimer = 0.0f;
      }
    }
    if(attract) {
      if(generating) {
        attractHoldTimer = 0.0f;  // still streaming -- never cut a line off
      } else {
        attractHoldTimer += deltaTime;
        if(attractHoldTimer >= ATTRACT_PAGE_HOLD) {
          attractHoldTimer = 0.0f;
          uint32_t next = pageStart + TEXT_CHARS_PER_PAGE;
          if(next < textLen) {
            pageStart = next;  // auto "press B": more of this line to show
          } else {
            /* nested odometer: CONTEXT fastest, then MOOD, then TRUST TIER */
            contextIdx = cycle(contextIdx, +1, NPCDatabase::CONTEXT_COUNT);
            if(contextIdx == 0) {
              moodIdx = cycle(moodIdx, +1, NPCDatabase::MOOD_COUNT);
              if(moodIdx == 0)trustTier = cycle(trustTier, +1, 3);
            }
            restartGeneration();
          }
        }
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
      {
        char npcLine[64]; // M10: grown from 40 -- the new "(M10, MET TR:N)"
                          // suffix pushes the worst case (longest name +
                          // longest label) past the old size
        if(currentNpc == 0)
          snprintf(npcLine, sizeof(npcLine), "64GPT V1.2 - SELENA (M7)");
        else if(isNewCastSlot())
          snprintf(npcLine, sizeof(npcLine), "64GPT V1.2 - %s (M9 CAST)",
                   NEW_CAST_NAMES[currentNpc - NEW_CAST_START]);
        else if(isNamedExtraSlot()) {
          // M10: show the PERSISTED high-water mark, not just the
          // current dial -- the actual player-visible proof the save
          // system works, not just an invisible mechanism.
          uint8_t remembered = isShadewrathSlot()
            ? SaveData::current.shadewrathHighestTier
            : SaveData::current.korrathHighestTier;
          snprintf(npcLine, sizeof(npcLine), "64GPT V1.2 - %s (M10, MET TR:%u)",
                   activeNpc().name, (unsigned)remembered);
        }
        else if(isNewArchetypeSlot())
          snprintf(npcLine, sizeof(npcLine), "64GPT V1.2 - %s (M10 %s)",
                   activeNpc().name,
                   NEW_ARCHETYPE_LABELS[currentNpc - NEW_ARCHETYPE_START]);
        else
          snprintf(npcLine, sizeof(npcLine), "64GPT V1.2 - %s (M8 GUARD)",
                   activeNpc().name);
        Debug::print(24, 40, npcLine);
      }

      // prompt/seed line: wrap across up to 3 rows (102 chars) so a long
      // schema string (e.g. a long context/event name, or M9's longer
      // P:/D:/OCC:/R:/M:/C:/EV: schema vs. the old N:/TR: one) doesn't
      // silently run off the right edge of the screen -- same fix as the
      // dialogue box below. Fixed 3-row reservation even when the prompt
      // is short, so everything below (STEP line at y=90) stays at a
      // constant position.
      {
        size_t promptLen = strlen(prompt);
        char prow[WRAP_COLS + 1];
        size_t ppos = 0;
        int py = 60;
        while(ppos < promptLen && py <= 80) {
          size_t n = 0;
          while(n < WRAP_COLS && ppos < promptLen)prow[n++] = prompt[ppos++];
          prow[n] = '\0';
          Debug::print(24, py, prow);
          py += 10;
        }
      }

      if(stepMeasured) {
        uint32_t us = (uint32_t)TICKS_TO_US(stepTicksEma);
        snprintf(line, sizeof(line), "STEP %lu US  RAW %lu CH/S",
                 (unsigned long)us, (unsigned long)(us ? 1000000u / us : 0));
        Debug::print(24, 90, line);
      }
      if(bootPhase == BOOT_READY && loaded) {
        snprintf(line, sizeof(line), "%s XCHK %s  CPU %lu RSP %lu US",
                 rspReady ? "RSP ON" : "RSP OFF",
                 xchkPass ? "PASS" : "FAIL",
                 (unsigned long)cpuStepUs, (unsigned long)rspStepUs);
        Debug::print(24, 106, line);
      }

      // dialogue box: wrap one page of the streamed text into rows. The
      // full line always lives in text[0..textLen); pageStart only picks
      // where this page starts, so paging never drops any generated text.
      char row[WRAP_COLS + 1];
      uint32_t pos = pageStart;
      int y = TEXT_ROW_Y0;
      int rowsDrawn = 0;
      while(pos < textLen && rowsDrawn < TEXT_ROWS_PER_PAGE) {
        uint32_t n = 0;
        while(n < WRAP_COLS && pos < textLen)row[n++] = text[pos++];
        row[n] = '\0';
        Debug::print(36, y, row);
        y += TEXT_ROW_DY;
        ++rowsDrawn;
      }
      bool moreText = pos < textLen;
      if(moreText)Debug::print(36, y, "MORE - PRESS B");
      else if(generating)Debug::print(36, y, ">"); // cursor while streaming

      // ---- TEMP ATTRACT MODE (BEGIN/END: one line) ----------------------
      #if NGPT_ATTRACT_MODE
      Debug::print(24, 200, attract ? "AUTO CYCLE - PRESS ANY BUTTON"
                                    : "DPAD TR/MOOD C CTX START:NPC A:REGEN B:MORE");
      #else
      Debug::print(24, 200, "DPAD TR/MOOD C CTX START:NPC A:REGEN B:MORE");
      #endif
    DrawLayer::useDefault();
  }
}
