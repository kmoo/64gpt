# ADR 0001 — Isolate decision logic into portable C++, host-test it; verify the hardware-bound residual on real hardware

**Status:** Accepted. Followed as an unwritten convention since M7
(`docs/milestones/m7.md`'s Event Bus / Shared World State / NPC Database
/ Context Builder split); this ADR formalizes a decision that was already
being made consistently, rather than proposing a new one.

## Context

The N64 has no host-runnable environment for its own code: there is no
way to compile ROM-side logic for the build machine and run it directly.
The two verification paths available are:

1. **Ares**, the hardware-accurate emulator — but it's GUI-only, with no
   headless or text-output mode (`CLAUDE.md`'s own build-command block
   notes this explicitly). Checking a boot result means launching the
   app, waiting out a self-test that scales with golden-vector count
   (~7s/golden on the RSP path — a 27-golden self-test takes 90-120+
   seconds, more under system load), taking a screenshot, and reading it.
   No assertions, no exit code — a human (or an agent reading an image)
   has to look.
2. **Real hardware** — slower to iterate on than Ares, reserved for
   milestone-level spot checks, not everyday development.

Meanwhile this project's actual defect history is dominated by ordinary
C++ logic bugs — off-by-one trust-tier math, a schema string built with
the wrong field order, a seed that doesn't remap correctly at its fixed
point, a stale `.o` reading a struct field at the wrong offset after an
ABI-breaking header change (M8's `ContextBuilder.o` incident, see
`CLAUDE.md`'s toolchain notes) — the same class of bug any C++ project
hits, with nothing N64-specific about the bug itself, only about how
expensive it is to *discover* on this platform if the only feedback loop
is "rebuild the ROM, boot Ares, wait, screenshot, read."

## Decision

Every module of game-side logic that does **not** inherently depend on
libdragon or real N64 hardware is written as standalone, portable C++
(no libdragon `#include`, no hardware peripheral access) and gets its
own host test executable, wired into `tests/CMakeLists.txt` the same way
`core/`'s inference engine always has been. Concretely, as of M11:

| Module | Host-tested? | Why |
|---|---|---|
| `core/ngpt*.cpp` (the inference engine) | Yes, always has been | Zero libdragon deps by hard constraint (`CLAUDE.md`) |
| `EventBus.cpp` | Yes | Pure ring buffer, no hardware |
| `WorldState.cpp` | Yes | Pure global-fact store, no hardware |
| `NPCDatabase.cpp` | Yes | Pure data + deterministic RNG jitter |
| `ContextBuilder.cpp` | Yes | Pure string formatting |
| `NpcService.cpp` | Yes | Pure compositional-schema mapping |
| `DungeonGenerator.cpp` | Yes | Pure deterministic seed derivation |
| `SaveData.cpp` | **No** | Real EEPROM peripheral (libdragon's `eepfs`) — cannot be simulated on host |
| `DialogueDemo.cpp` | **No** | Pyrite64 script glue: joypad reads, `Debug::print`, RSP DMA staging — inherently hardware-bound |

Modules in the second group are verified the slower way instead: a real
ROM build, a real Ares (or hardware) boot, `SELFTEST PASS`/`XCHK PASS`
on screen, and — where the feature is interactive rather than golden-
vector-provable (e.g. M10's dungeon generation, M11's gossip) — a
specific, exact, screenshot-checkable on-screen flag (`MET TR:N`,
`NEWS`) standing in for an assertion a human or agent can actually read
from an image, rather than asking anyone to judge generated dialogue
text by eye.

**The practical rule this produces**: when a new feature needs real
decision logic (a new predicate, a new derivation, a new routing rule),
that logic gets written and tested in a portable module *first*, and
`DialogueDemo.cpp` is only ever the thin call site that wires it into
input handling and the screen. M10's `DungeonGenerator::npcsForLevel()`
and M11's `NpcService::isGossipHub()`/`eventFor()` both followed this
shape: the decision ("which archetype/seed does this level get", "does
this occupation react to gossip") lives in a tested module;
`DialogueDemo.cpp` only calls it and displays the result.

## The actual numbers (measured 2026-07-18, not assumed)

By raw line count across `core/` + `game/src/user/`'s `.cpp` files:

- Host-tested: `core/` (367 lines) + the six portable `game/src/user/`
  modules (500 lines) = **867 lines**
- Not host-tested (real hardware/libdragon-bound by nature):
  `SaveData.cpp` (64) + `DialogueDemo.cpp` (826) = **890 lines**
- Raw split: **~49% of total `.cpp` lines are host-tested.**

That's a real number, and it's a long way from "90%" — worth stating
plainly rather than rounding up. The reason it isn't misleading: nearly
all of the untested half is exactly one file (`DialogueDemo.cpp`, 826 of
the 890 untested lines), and that file is overwhelmingly input-reading
and screen-drawing glue, not decision logic. Reframed as "of the code
that has no *unavoidable* hardware dependency, what fraction is actually
tested" — the honest question this ADR is really answering — the answer
is closer to **100%**: every module in the table above that *could* be
made portable, *has* been, with no exceptions carried forward
untested. `SaveData.cpp` and `DialogueDemo.cpp` aren't gaps in this
discipline; they're its edge, drawn at the actual hardware boundary.

## Consequences

**Positive**: the fast host suite (9 executables, ~2 seconds total, run
with ASan/UBSan) catches the large majority of real logic bugs — schema
formatting, tier math, seed determinism, occupation-gating rules — in
seconds, at the moment they're introduced, rather than 90-120+ seconds
and a manual screenshot later, or worse, on real hardware. Every
milestone from M7 onward has shipped red→green host tests for its new
logic before touching Ares at all, which is what makes the "ROM boots,
SELFTEST PASS" step a confirmation rather than a debugging session.

**Negative / accepted residual**: `DialogueDemo.cpp` will likely remain
the largest untested file in the codebase, structurally, for the life of
the project — it cannot be fully extracted away without also extracting
the libdragon calls that make it a working Pyrite64 script. Bugs in *how
already-tested modules get wired together and displayed* (a wrong call
site, a mis-ordered branch, a truncated `snprintf` buffer) are caught
only by the slow path. This has actually happened: the M10 dungeon-mode
NPC-cycling verification needed multiple false starts because Ares'
input mapping had to be manually rediscovered through its real Settings
dialog rather than guessed from raw scancodes (`CLAUDE.md`'s Ares
key-mapping note) — a class of friction host tests structurally cannot
prevent, because the thing being verified (a human pressing a button on
an emulated N64 controller) doesn't exist on the host at all.

**Mitigation in practice, not just in principle**: keep pushing new
decision logic into portable, tested modules before it ever touches
`DialogueDemo.cpp`, so the file's own line count grows mostly through
call sites and display formatting, not through branching logic that
could have lived somewhere testable. This is a discipline to keep
re-applying at each future milestone (M11's town cast, dungeon loop,
manifest-update skill, etc.), not a one-time refactor — add to
`docs/plan.md`'s Known follow-ups if a future milestone knowingly lets
real logic leak into the untestable file under time pressure.
