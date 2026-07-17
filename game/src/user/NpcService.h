#pragma once
#include <stdint.h>
#include "NPCDatabase.h"

// NpcService: M9's compositional conditioning mapping -- C++ port of
// trainer/ngpt_trainer/npc_service.py, cross-checked against it byte-for-
// byte (tests/test_npc_service.cpp against real Python output, same
// discipline test_guard_instances.py used checking Python against
// compiled NPCDatabase.cpp ground truth, just the roles reversed since
// here Python is the validated prototype). See docs/milestones/m9.md.
//
// Replaces M7/M8's opaque N:<id> identity tag (and its TR: trust-tier
// dial) with reusable descriptive features that generalize across
// characters sharing traits -- npc.id stays as an internal NPCDatabase
// lookup key only, it no longer needs to be what's fed into the
// conditioning string.
//
// Fixed-point throughout: relationship axes are 0..1000 (Python's
// 0.0..1.0 * 1000) -- this codebase avoids floats even outside core/'s
// hard "no float" constraint, matching the project's N64-performance
// discipline everywhere else (int8/int16 quantization, fixed-point RNG).
//
// Portable C++, no libdragon includes -- builds in the host test suite,
// same discipline as ContextBuilder/NPCDatabase.
namespace NpcService
{
  constexpr int OCCUPATION_COUNT = 12;
  extern const char *const OCCUPATIONS[OCCUPATION_COUNT];

  enum class Gender { Female, Male };

  struct Profile
  {
    const char *occupation;                // one of OCCUPATIONS, lowercase
    int age;                               // years
    Gender gender;
    int traits[NPCDatabase::TRAIT_COUNT];  // 0-100, same axes/order as NPCDatabase::TRAITS
  };

  // 5 axes, 0..1000 fixed-point (Python random_relationship_state()'s
  // 0.0-1.0 range * 1000).
  struct RelationshipState
  {
    uint16_t familiarity, affection, trust, respect, fear;
  };

  // Same RNG discipline as core/ngpt_sample.cpp, NPCDatabase.cpp's
  // spawnInstance, and npc_service.py's xorshift32 -- seed 0 remaps to 1
  // at call sites, same fixed-point rule.
  uint32_t xorshift32(uint32_t x);

  // "girl"/"boy"/"woman"/"man"/"elderly_woman"/"elderly_man" -- spaces
  // pre-replaced with '_' since every prompt_fields() token must be a
  // single space-free unit (matches npc_service.py's prompt_fields(),
  // not its human-readable conditioning_features() form). Writes into
  // out (>= 16 bytes), NUL-terminated.
  void ageGenderToken(int age, Gender gender, char *out, uint32_t outCap);

  // Ordered blend rules over the 5 trait sliders -- mirrors
  // npc_service.py's personality_descriptor() exactly, including the
  // Selena calibration requirement ({90,85,70,55,30} -> "sassy").
  const char *personalityDescriptor(const int traits[NPCDatabase::TRAIT_COUNT]);

  // Mean of familiarity/affection/trust/respect (fear excluded -- a
  // separate modifier, not averaged in: a feared-but-trusted relationship
  // reads differently than a low-trust one and shouldn't cancel out).
  // 0..1000.
  uint16_t closeness(const RelationshipState &state);

  // Buckets closeness (0..1000) into stranger/acquaintance/neutral/
  // friend/close_friend/best_friend -- thresholds 0/200/400/600/800/950,
  // matching npc_service.py's RELATIONSHIP_TIERS scaled by 1000.
  const char *relationshipTier(uint16_t closenessScore);

  // The actual training/inference prompt string: colon-delimited tokens
  // (every space-separated token carries its own colon, matching
  // ContextBuilder's N:/TR:/M:/C:/EV: convention) -- byte-for-byte match
  // to npc_service.py's prompt_fields(). event may be "" or nullptr (an
  // idle NPC with no recent event still needs a valid EV: field).
  // Writes into out (must be >= 96 bytes), NUL-terminates. Returns the
  // written length excluding the NUL.
  //
  // "P:girl D:sassy OCC:villager R:best_friend M:cheerful C:greeting
  //  EV:none|"
  uint32_t buildPromptFields(char *out, uint32_t outCap, const Profile &profile,
                              const RelationshipState &relationship,
                              const char *mood, const char *context,
                              const char *event);
}
