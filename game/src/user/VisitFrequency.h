#pragma once
#include <stdint.h>

// Visit cadence (docs/ideas-m7-living-npcs.md Part 4: "a regular daily
// visitor vs someone who shows up once a season" -- explicitly distinct
// from NPCState::Relationship's familiarity, which tracks depth/
// duration, not cadence; two players could reach the same familiarity
// via very different visit patterns). GAME-STATE half only, same
// discipline as every other NPCState-adjacent header tonight.
//
// Simplification, stated plainly rather than overclaimed: this is a
// RECENCY proxy (gap since the last visit), not a true rolling-average
// regularity measure -- a player who visited exactly once, yesterday,
// currently reads identically to an actual daily regular. A real
// historical-regularity metric would need to track multiple past gaps,
// out of scope for this pure data-layer slice.
namespace VisitFrequency
{
  struct VisitLog
  {
    uint32_t lastVisitDay;
    uint32_t visitCount;
  };

  // Idempotent within the same day -- a second interaction on the same
  // day must not double-count (that's Relationship familiarity's job,
  // not cadence).
  inline void recordVisit(VisitLog &log, uint32_t day)
  {
    if (log.visitCount > 0 && log.lastVisitDay == day) return;
    log.lastVisitDay = day;
    log.visitCount++;
  }

  enum Cadence { NEVER_VISITED, DAILY, WEEKLY, OCCASIONAL };

  constexpr uint32_t DAILY_MAX_GAP = 1;
  constexpr uint32_t WEEKLY_MAX_GAP = 7;

  // currentDay before lastVisitDay is caller error (day counters never
  // go backwards elsewhere in this codebase) -- treated as gap 0 rather
  // than wrapping/going negative on an unsigned subtraction.
  inline Cadence classifyCadence(const VisitLog &log, uint32_t currentDay)
  {
    if (log.visitCount == 0) return NEVER_VISITED;
    uint32_t gap = (currentDay > log.lastVisitDay) ? (currentDay - log.lastVisitDay) : 0;
    if (gap <= DAILY_MAX_GAP) return DAILY;
    if (gap <= WEEKLY_MAX_GAP) return WEEKLY;
    return OCCASIONAL;
  }
}
