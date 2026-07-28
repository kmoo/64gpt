#pragma once

// NPC-to-NPC disposition scoring (docs/ideas.md #4 "NPC-to-NPC argument
// mode": "guard and merchant alternate turns... emergent-feeling
// banter"). This is the DATA-LAYER slice only: a baseline compatibility
// score between two NPCs' personality traits, useful for deciding
// (later, at the dialogue-generation layer this header doesn't touch)
// whether a scripted NPC-to-NPC scene should read as friendly banter or
// a hostile argument. No dependency on NPCDatabase.h -- traits/count
// are passed in by the caller (e.g. NPCDatabase::NPC.personality,
// NPCDatabase::TRAIT_COUNT), same loose-coupling pattern as
// SpawnSeedSource.h.
namespace NpcDisposition
{
  // Manhattan distance across all trait axes (integer, no floats),
  // converted to a 0-100 COMPATIBILITY score: 100 means identical
  // traits (maximally compatible baseline), 0 means maximally opposed
  // on every axis. maxTraitValue is the traits' own scale ceiling (100
  // for this project's 0-100 personality axes) -- needed to normalize
  // the distance into a fixed 0-100 range regardless of traitCount.
  inline int compatibilityScore(const int *traitsA, const int *traitsB,
                                 int traitCount, int maxTraitValue)
  {
    if (traitCount <= 0 || maxTraitValue <= 0) return 0;
    int totalDistance = 0;
    for (int i = 0; i < traitCount; ++i) {
      int d = traitsA[i] - traitsB[i];
      if (d < 0) d = -d;
      totalDistance += d;
    }
    int maxPossibleDistance = traitCount * maxTraitValue;
    return 100 - (totalDistance * 100) / maxPossibleDistance;
  }

  // Named cutoff so call sites don't repeat a bare number -- friendly
  // banter vs. a hostile argument, per this header's own framing above.
  constexpr int FRIENDLY_THRESHOLD = 50;

  inline bool isFriendlyPairing(const int *traitsA, const int *traitsB,
                                 int traitCount, int maxTraitValue)
  {
    return compatibilityScore(traitsA, traitsB, traitCount, maxTraitValue)
             >= FRIENDLY_THRESHOLD;
  }
}
