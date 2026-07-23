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
  // M11.1: 12 M9/M10/M11 entries + "villain"/"knight"/"companion" --
  // real manifest-schema decisions made genericizing Shadewrath/Korrath/
  // Selena onto this scheme found no existing entry that fit without
  // merging their voice into an unrelated occupation's bank (docs/
  // milestones/m11.1.md Part 1).
  constexpr int OCCUPATION_COUNT = 15;
  extern const char *const OCCUPATIONS[OCCUPATION_COUNT];

  // M11.1 Part 1/3: SPECIES: and BOND: -- not every value has a trained
  // character yet ("dwarf"/"beast" and "captor"/"family"/"mentor"/
  // "romantic" are declared vocabulary with no corpus this pass, same
  // "declared, not yet exercised" status OCCUPATIONS values commonly
  // start in). Must match trainer/ngpt_trainer/npc_service.py's
  // SPECIES_TYPES/BOND_TYPES exactly, same discipline as OCCUPATIONS.
  constexpr int SPECIES_COUNT = 5;
  extern const char *const SPECIES[SPECIES_COUNT];
  constexpr int BOND_COUNT = 9;
  extern const char *const BOND_TYPES[BOND_COUNT];

  // M11.1 Part 3: AUD: (audience) -- operationalizes the manifest bible's
  // public/private/secret fields (existed since M7 as pure authoring
  // guidance, docs/08-manifest-schema.md) into something the model can
  // act on directly. "trusted" (a closeness threshold unlocking private
  // content even when witnessed) is a recorded v2 candidate, not built
  // this pass.
  constexpr int AUDIENCE_COUNT = 2;
  extern const char *const AUDIENCE_TYPES[AUDIENCE_COUNT];

  enum class Gender { Female, Male };

  struct Profile
  {
    const char *occupation;                // one of OCCUPATIONS, lowercase
    const char *species;                   // one of SPECIES, lowercase
    int age;                               // years
    Gender gender;
    const char *bond;                      // one of BOND_TYPES, lowercase
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
  // audience may be "" or nullptr (defaults to "alone", matching
  // prompt_fields()'s Python default). Writes into out (must be >= 192
  // bytes -- M11.1 grew the M9 96-byte budget by 3 new fields, see
  // docs/milestones/m11.1.md Part 1's buffer-size note), NUL-terminates.
  // Returns the written length excluding the NUL.
  //
  // "P:girl D:sassy OCC:villager SPECIES:human R:best_friend BOND:rival
  //  M:cheerful C:greeting AUD:alone EV:none|"
  uint32_t buildPromptFields(char *out, uint32_t outCap, const Profile &profile,
                              const RelationshipState &relationship,
                              const char *mood, const char *context,
                              const char *audience, const char *event);

  // M10: bridges an NPCDatabase::NPC (spawnInstance()'s output, or a
  // hand-authored named individual like selena/shadewrath) to a Profile,
  // so any occupation+age+species+bond-carrying NPC can feed
  // buildPromptFields() directly. M11.1: every NPC in NPCDatabase now
  // sets these fields (Part 1 "one scheme, not two" retired the old
  // ContextBuilder/N:<id> scheme entirely), so this is the only bridge
  // left, not one of two.
  Profile profileFor(const NPCDatabase::NPC &npc);

  // M11: true for occupations the corpus actually trained to react to
  // town gossip secondhand (trainer/ngpt_trainer/cast_corpus.py's
  // GOSSIP_HUB_OCCUPATIONS: pub_patron, villager -- "EVERY VILLAGE NEEDS
  // A GOOD GOSSIP" is a literal line in villager's own corpus). Routing
  // a gossip tag to any other occupation would feed an out-of-
  // distribution EV: token nothing in its training ever showed it.
  bool isGossipHub(const char *occupation);

  // M11: the EV: value a compositional-scheme NPC's prompt should carry
  // -- gossip if this occupation is a trained hub AND gossip is set,
  // otherwise directEvent (the ordinary per-interaction event, e.g.
  // EventBus::lastTag()). Pure function: the caller supplies both
  // strings, no dependency on WorldState/EventBus here, same portability
  // discipline as the rest of this module. occupation/directEvent/gossip
  // may be "" or nullptr.
  const char *eventFor(const char *occupation, const char *directEvent,
                       const char *gossip);
}
