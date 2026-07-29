#pragma once
#include <stdint.h>

// Time-of-day state (docs/ideas-m7-living-npcs.md Part 3's "World
// Context" bullet listed time as already built; Part 4 corrects that --
// no T: field exists, nothing computes or feeds time-of-day into
// ContextBuilder today). This closes the gap on the GAME-STATE half
// only: a tick counter -> discrete time-of-day bucket. Deriving a
// T:-style conditioning token from this and regenerating the corpus is
// separate game-content work, explicitly out of scope here, same
// "build the pure state layer first" discipline as every other
// NPCState-adjacent header tonight.
namespace TimeOfDay
{
  enum Period { DAWN, DAY, DUSK, NIGHT };

  // Arbitrary placeholder cadence -- the real engine integration decides
  // ticksPerDay for its own frame-rate/pacing. Every function below
  // takes it as an explicit parameter rather than hardcoding this
  // default, so callers aren't locked into it.
  constexpr uint32_t DEFAULT_TICKS_PER_DAY = 24000;

  // Splits one day into four equal quarters -- DAWN, DAY, DUSK, NIGHT,
  // in that cyclic order starting at tick 0 -- and wraps automatically
  // past one day (tick is not required to reset). ticksPerDay <= 0, or
  // too small to divide into meaningful quarters, always returns DAY --
  // the safest silent default rather than dividing by zero.
  inline Period periodForTick(uint32_t tick, uint32_t ticksPerDay)
  {
    if (ticksPerDay == 0) return DAY;
    uint32_t quarter = ticksPerDay / 4;
    if (quarter == 0) return DAY;
    uint32_t phase = (tick % ticksPerDay) / quarter;
    switch (phase) {
      case 0: return DAWN;
      case 1: return DAY;
      case 2: return DUSK;
      default: return NIGHT; // covers phase 3 and any rounding remainder
    }
  }

  inline bool isNight(Period period) { return period == NIGHT; }
}
