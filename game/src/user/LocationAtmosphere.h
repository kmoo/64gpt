#pragma once
#include <stdint.h>

// Location atmosphere (docs/ideas-m7-living-npcs.md Part 4: festive/
// tense/abandoned as a property of the PLACE, not the NPC -- "the same
// NPC could read differently depending on where the conversation
// happens," distinct from the NPC's own NpcCondition.h state). GAME-
// STATE half only, same discipline as every other NPCState-adjacent
// header tonight.
namespace LocationAtmosphere
{
  enum Mood { NEUTRAL, FESTIVE, TENSE, ABANDONED };

  struct AtmosphereState
  {
    Mood mood;
    uint32_t setDay; // day the mood was last set, for decay
  };

  inline void setMood(AtmosphereState &state, Mood mood, uint32_t day)
  {
    state.mood = mood;
    state.setDay = day;
  }

  // Moods decay back to NEUTRAL after decayDays -- a festival's glow
  // fades, crime-scene tension eases, matching atmosphere as a
  // transient property rather than a permanent label. NEUTRAL never
  // "decays" further (nothing left to decay to). currentDay before
  // setDay (caller error, day counters never go backwards elsewhere in
  // this codebase) is treated as elapsed 0, not wrapped.
  inline Mood currentMood(const AtmosphereState &state, uint32_t currentDay, uint32_t decayDays)
  {
    if (state.mood == NEUTRAL) return NEUTRAL;
    uint32_t elapsed = (currentDay > state.setDay) ? (currentDay - state.setDay) : 0;
    if (elapsed >= decayDays) return NEUTRAL;
    return state.mood;
  }
}
