#pragma once

// NPC Database: per-NPC identity + mutable state (trust tier, mood) the
// Context Builder reads to build the M7 conditioning string. M7 scope:
// Selena only — the milestone's one full-tier character — plus the
// declared mood/context vocabularies her schema uses (see
// docs/milestones/m7.md, "corpus diversity" combo grid: Mood x Trust
// tier x Context). M8 formalizes these lists into a project manifest
// (schema_fields) instead of hardcoded arrays; nothing here needs to
// change for that migration, per m7.md's own forward note.
//
// Portable C++, no libdragon includes — builds in the host test suite.
namespace NPCDatabase
{
  constexpr int MOOD_COUNT = 5;
  extern const char *const MOODS[MOOD_COUNT];

  constexpr int CONTEXT_COUNT = 8;
  extern const char *const CONTEXTS[CONTEXT_COUNT];

  struct NPC
  {
    const char *id;   // trained identity tag, e.g. "selena" — lowercase,
                      // must match the corpus's N: tag exactly
    int trustTier;    // 0..3, per the schema's TR: field
    int moodIdx;      // index into MOODS
  };

  extern NPC selena;
}
