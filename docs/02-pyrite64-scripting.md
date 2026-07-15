# 02 — Pyrite64 scripting: `DialogueDemo.cpp` explained

If you've never used a game engine: an engine runs a **loop**, not your
`main()`. Every frame (~60×/sec) it updates all objects in the current
scene, then draws them. Your code is a set of callbacks on an object.
This page walks through our one script.

## The object script model

A scene contains objects; objects have components; one component type is
"Code", which binds a script from `src/user/` to the object. A script is
a namespace with a magic name, a data block, and some or all of a fixed
set of functions the engine calls:

```cpp
namespace P64::Script::C64D1A106DE00001   // 16-hex-digit UUID, the script's identity
{
  P64_DATA();                              // per-instance data (ours is empty)

  void init(Object& obj, Data *data);                      // on spawn
  void destroy(Object& obj, Data *data);                   // on removal
  void update(Object& obj, Data *data, float deltaTime);   // once per frame, logic
  void fixedUpdate(Object& obj, Data *data, float dt);     // physics steps (unused here)
  void draw(Object& obj, Data *data, float deltaTime);     // once per camera pass, rendering
}
```

The UUID in the namespace is how the editor identifies the script — you
can rename the file freely, but never change the UUID. `P64_DATA(...)`
declares per-instance state the engine allocates for you; since our demo
is a singleton we keep shared state in an anonymous namespace instead
(a pattern the official examples also use) and leave `P64_DATA` empty.

## `init` — load the model, prove it works

```cpp
int blobSize = 0;
blobData = (uint8_t*)asset_load("rom:/model.bin", &blobSize);   // DFS → RAM
loaded = blobData && ngpt_load(&model, blobData, blobSize) == NGPT_OK;
selftestPass = loaded && runSelfTest();
restartGeneration();
```

`runSelfTest()` runs one complete generation and compares every byte
against `selftestGolden.h` — the same golden the host tests use
(both are emitted by `trainer/make_canned_blob.py`). If host tests are
green and this shows FAIL on the console, something platform-specific
broke (endianness, alignment, memory) — which is precisely what the
self-test exists to catch.

## `update` — stream a few characters per frame

```cpp
auto pressed = joypad_get_buttons_pressed(JOYPAD_PORT_1);
if(pressed.a) restartGeneration();

for(int i = 0; i < CHARS_PER_FRAME && generating; ++i) {
  int c = ngpt_step(&ctx);
  if(c == NGPT_EOS) { generating = false; break; }
  text[textLen++] = (char)c;
}
```

This is the *streaming API contract* in action: `ngpt_step` does a small
bounded amount of work, so calling it twice per frame costs a fixed
slice of the 16.6 ms frame budget. When the real GRU lands (M2) each
step gets slower but the game-side code doesn't change at all.

`joypad_get_buttons_pressed` is libdragon: edge-triggered button state
(pressed this frame), vs `..._held` (currently down).

## `draw` — text via the engine's debug font

Object `draw` runs during 3D rendering, so we explicitly switch to the
2D layer (same pattern as the official jam25 example's HUD):

```cpp
DrawLayer::use2D();
  Debug::printStart();            // sets up render modes + font texture once
  Debug::print(24, 24, selftestPass ? "SELFTEST PASS" : "SELFTEST FAIL");
  // ... wrapped dialogue rows, "PRESS A TO REGENERATE" footer
DrawLayer::useDefault();
```

`Debug::print` (engine builtin, `debug/debugDraw.h`) draws with the
always-shipped `p64/font.ia4.sprite`: 8×8 glyphs, 7 px advance,
**uppercase only** (lowercase input is folded to uppercase glyphs).
Zero assets to manage — ideal for a walking skeleton. A proper
variable-width font via `rdpq_text_printf` + a font64 asset (see the
jam25 example's `systems/dialog.cpp`) is the planned upgrade once the
demo grows beyond the skeleton.

Wrapping is manual: `Debug::print` has no layout engine, so the script
slices the streamed buffer into 34-character rows.

## Wiring the script to a scene (one-time, in the editor)

`game/README.md` has the exact steps: create the project into `game/`,
add an empty object, add a Code component, pick `DialogueDemo`. The
editor stores that binding in `data/` (editor-owned JSON); the build
then generates the glue in `src/p64/` that routes engine callbacks to
our functions.
