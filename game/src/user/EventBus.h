#pragma once
#include <stdint.h>

// Event Bus: the ONLY contract other subsystems (Dungeon System, Music
// System — reserved bus slots, not built yet) must honor to react to
// game state, per docs/milestones/m7.md's architecture diagram. Any
// subsystem publishes; any subsystem reads. Game-side only — needs
// bookkeeping the no-heap/no-float engine rule doesn't apply to;
// core/'s frozen streaming API is untouched by any of this.
//
// M7 scope: publish + "what's the latest event" is all Selena's Context
// Builder needs. No subscriber-callback registry yet — there is exactly
// one reader today, and a callback list with one subscriber is the
// premature abstraction CLAUDE.md warns against. Add it when a second
// real subscriber (Dungeon/Music, M8+) actually needs to be pushed to
// instead of pulling lastTag().
//
// Portable C++, no libdragon includes — this file also builds in the
// host test suite (tests/test_context_builder.cpp).
namespace EventBus
{
  constexpr uint32_t MAX_TAG_LEN = 24;
  constexpr uint32_t HISTORY_DEPTH = 4;

  // Publishes an event; tag is copied and truncated to MAX_TAG_LEN-1.
  // Overwrites the oldest history slot once HISTORY_DEPTH is exceeded —
  // fixed memory, no heap, same discipline as the engine's buffers.
  void publish(const char *tag);

  // Most recently published tag, or "" if publish() was never called
  // (or reset() ran since).
  const char *lastTag();

  // Total events published since the last reset() (saturates, doesn't
  // wrap — only used for test assertions and debug display).
  uint32_t eventCount();

  void reset();
}
