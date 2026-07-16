/**
 * 64GPT dialogue demo — Pyrite64 object script.
 *
 * Attach to any (empty) object via a "Code" component in the editor.
 * initDelete: loads the model blob from the ROM filesystem and runs the
 *             boot self-test (all 12 prompted generations vs goldens).
 * update    : streams a few characters per frame. Controls:
 *             D-pad up/down = NPC, left/right = MOOD, C-left/right = EV,
 *             A = regenerate with the current prompt.
 * draw      : prompt line + dialogue box + SELFTEST PASS/FAIL banner,
 *             drawn with the engine's builtin debug font (uppercase only).
 */
#include <stdlib.h>
#include <stdio.h>
#include "script/userScript.h"
#include "debug/debugDraw.h"
#include "n64gpt/ngpt.h"
#include "selftestGolden.h"

namespace
{
  constexpr int CHARS_PER_FRAME = 2; // slow enough to see the streaming
  constexpr int WRAP_COLS = 34;      // 7px glyph advance, ~240px text area

  // One demo instance per scene; shared state keeps P64_DATA trivial.
  uint8_t *blobData{};
  ngpt_model model{};
  ngpt_ctx ctx{};
  char text[512]{};
  char prompt[64]{};
  uint32_t textLen{};
  int npcIdx{}, moodIdx{}, evIdx{};
  bool generating{};
  bool loaded{};
  bool selftestPass{};

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

  void buildPrompt()
  {
    snprintf(prompt, sizeof(prompt), "NPC=%s MOOD=%s EV=%s|",
             SELFTEST_NPCS[npcIdx], SELFTEST_MOODS[moodIdx], SELFTEST_EVENTS[evIdx]);
  }

  void restartGeneration()
  {
    textLen = 0;
    generating = loaded;
    buildPrompt();
    if(loaded)ngpt_reset(&ctx, &model, prompt);
  }

  bool runSelfTest()
  {
    // Replay all 12 prompted generations against the committed goldens.
    for(uint32_t p = 0; p < SELFTEST_COUNT; ++p) {
      ngpt_reset(&ctx, &model, SELFTEST_PROMPTS[p]);
      const char *want = SELFTEST_GOLDEN[p];
      uint32_t i = 0;
      int c;
      while((c = ngpt_step(&ctx)) != NGPT_EOS) {
        if(want[i] == '\0')return false;
        if((uint8_t)c != (uint8_t)want[i])return false;
        ++i;
      }
      if(want[i] != '\0')return false;
    }
    return true;
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
      loaded = false;
      return;
    }

    int blobSize = 0;
    blobData = (uint8_t*)asset_load("rom:/model.bin", &blobSize);
    loaded = blobData && ngpt_load(&model, blobData, (uint32_t)blobSize) == NGPT_OK;
    selftestPass = loaded && runSelfTest();
    restartGeneration();
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
    auto pressed = joypad_get_buttons_pressed(JOYPAD_PORT_1);
    bool changed = false;
    if(pressed.d_up)   { npcIdx  = cycle(npcIdx,  +1, SELFTEST_NPC_COUNT);   changed = true; }
    if(pressed.d_down) { npcIdx  = cycle(npcIdx,  -1, SELFTEST_NPC_COUNT);   changed = true; }
    if(pressed.d_right){ moodIdx = cycle(moodIdx, +1, SELFTEST_MOOD_COUNT);  changed = true; }
    if(pressed.d_left) { moodIdx = cycle(moodIdx, -1, SELFTEST_MOOD_COUNT);  changed = true; }
    if(pressed.c_right){ evIdx   = cycle(evIdx,   +1, SELFTEST_EVENT_COUNT); changed = true; }
    if(pressed.c_left) { evIdx   = cycle(evIdx,   -1, SELFTEST_EVENT_COUNT); changed = true; }
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
        /* nested odometer: EV fastest, then MOOD, then NPC */
        evIdx = cycle(evIdx, +1, SELFTEST_EVENT_COUNT);
        if(evIdx == 0) {
          moodIdx = cycle(moodIdx, +1, SELFTEST_MOOD_COUNT);
          if(moodIdx == 0)npcIdx = cycle(npcIdx, +1, SELFTEST_NPC_COUNT);
        }
        restartGeneration();
      }
    }
    #endif
    // ---- TEMP ATTRACT MODE (END) --------------------------------------

    for(int i = 0; i < CHARS_PER_FRAME && generating; ++i) {
      int c = ngpt_step(&ctx);
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

      Debug::print(24, 24, selftestPass ? "SELFTEST PASS" : "SELFTEST FAIL");
      Debug::print(24, 40, "64GPT V0.3 - PROMPTED GRU");
      Debug::print(24, 60, prompt);

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
                                    : "DPAD NPC/MOOD  C EV  A REGEN");
      #else
      Debug::print(24, 200, "DPAD NPC/MOOD  C EV  A REGEN");
      #endif
    DrawLayer::useDefault();
  }
}
