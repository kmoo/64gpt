/**
 * 64GPT dialogue demo — Pyrite64 object script.
 *
 * Attach to any (empty) object via a "Code" component in the editor.
 * Init  : loads the model blob from the ROM filesystem and runs the boot
 *         self-test (full generation vs the committed golden bytes).
 * Update: streams a few characters per frame; A regenerates.
 * Draw  : dialogue box + SELFTEST PASS/FAIL banner, drawn with the
 *         engine's builtin debug font (uppercase only — fine for now).
 */
#include <stdlib.h>
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
  uint32_t textLen{};
  bool generating{};
  bool loaded{};
  bool selftestPass{};

  void restartGeneration()
  {
    textLen = 0;
    generating = loaded;
    if(loaded)ngpt_reset(&ctx, &model, "");
  }

  bool runSelfTest()
  {
    ngpt_reset(&ctx, &model, "");
    uint32_t i = 0;
    int c;
    while((c = ngpt_step(&ctx)) != NGPT_EOS) {
      if(i >= SELFTEST_GOLDEN_LEN)return false;
      if((uint8_t)c != (uint8_t)SELFTEST_GOLDEN[i])return false;
      ++i;
    }
    return i == SELFTEST_GOLDEN_LEN;
  }
}

namespace P64::Script::C64D1A106DE00001
{
  P64_DATA();

  void init(Object& obj, Data *data)
  {
    int blobSize = 0;
    blobData = (uint8_t*)asset_load("rom:/model.bin", &blobSize);
    loaded = blobData && ngpt_load(&model, blobData, (uint32_t)blobSize) == NGPT_OK;
    selftestPass = loaded && runSelfTest();
    restartGeneration();
  }

  void destroy(Object& obj, Data *data)
  {
    if(blobData) {
      free(blobData);
      blobData = nullptr;
    }
    loaded = false;
  }

  void update(Object& obj, Data *data, float deltaTime)
  {
    auto pressed = joypad_get_buttons_pressed(JOYPAD_PORT_1);
    if(pressed.a)restartGeneration();

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
      Debug::print(24, 40, "64GPT V0.1 - CANNED MODEL");

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

      Debug::print(24, 208, "PRESS A TO REGENERATE");
    DrawLayer::useDefault();
  }
}
