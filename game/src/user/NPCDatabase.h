#pragma once
#include <stdint.h>

// NPC Database: per-NPC identity + mutable state (trust tier, mood) the
// Context Builder reads to build the M7 conditioning string, plus (M8)
// the archetype/instance model: a fixed personality point (characters[],
// e.g. Selena) vs. a range + deterministic seed jitter (archetypes[],
// e.g. guard) per docs/08-manifest-schema.md and docs/milestones/m8.md
// section 2. The declared mood/context vocabularies match
// manifests/dungeon_crawler.json's schema_fields exactly (see
// docs/milestones/m7.md's "corpus diversity" combo grid: Mood x Trust
// tier x Context) — hardcoded here still, per m7.md's own forward note
// that the manifest formalizes these lists without requiring this file
// to change shape for the migration.
//
// Portable C++, no libdragon includes — builds in the host test suite.
namespace NPCDatabase
{
  constexpr int MOOD_COUNT = 5;
  extern const char *const MOODS[MOOD_COUNT];

  constexpr int CONTEXT_COUNT = 8;
  extern const char *const CONTEXTS[CONTEXT_COUNT];

  constexpr int TRAIT_COUNT = 5;
  extern const char *const TRAITS[TRAIT_COUNT]; // warmth/humor/impulsivity/bravery/focus

  constexpr int MAX_ID_LEN = 20;   // "shopkeeper#ffff" + NUL, room to spare
  constexpr int MAX_NAME_LEN = 16;

  // M10: the cast tier (docs/08-manifest-schema.md, docs/milestones/
  // m10.md section 1) — full (hand-authored individual, e.g. Selena) and
  // mid (a named boss, smaller corpus, persistent within its own arc) are
  // both characters[]-shaped; thin (archetype instances, e.g. guard) is
  // always spawnInstance()'s output, never hand-authored. mid has no
  // concrete instance in code yet (m10.md: "don't build against it
  // expecting specific semantics until M10 writes them") — the enum
  // value exists so isPersistent() can express the real rule now.
  enum class Tier { FULL, MID, THIN };

  // full and mid tier NPCs persist across dungeon-loop iterations (the
  // bad guy remembers you across separate runs); thin-tier archetype
  // instances are ephemeral, no durable memory beyond the current
  // encounter. Explicit decision rule per m10.md section 3, not left
  // implicit in scattered call sites.
  inline bool isPersistent(Tier tier) { return tier != Tier::THIN; }

  struct NPC
  {
    char id[MAX_ID_LEN];         // conditioning N: tag: fixed ("selena")
                                  // or generated ("guard#4f2a"), lowercase
                                  // per schema-value convention
    char name[MAX_NAME_LEN];     // display name; "" for characters that
                                  // don't need one yet (unused until M8 §4)
    int trustTier;               // 0..2, per the schema's TR: field
    int moodIdx;                 // index into MOODS
    int personality[TRAIT_COUNT]; // 0-100 per TRAITS axis; fixed point for
                                   // a character, jittered for an instance
    uint32_t memorySlot;          // opaque per-instance memory handle,
                                   // M9+ scope; 0 = empty/unused
    Tier tier = Tier::THIN;       // default matches spawnInstance()'s only
                                   // valid tier; characters[]-shaped NPCs
                                   // (selena) override explicitly
    const char *occupation = nullptr; // M10: one of NpcService::OCCUPATIONS,
                                   // set by spawnInstance() from the
                                   // archetype; nullptr for old-scheme NPCs
                                   // (selena/guard) that don't feed
                                   // NpcService::buildPromptFields()
    int age = 0;                  // M10: years, jittered from the
                                   // archetype's ageRange; 0 = unset
    bool isFemale = false;        // M10: avoids this header depending on
                                   // NpcService::Gender (NpcService already
                                   // depends on NPCDatabase, not the reverse)
  };

  extern NPC selena;

  // An archetype template (manifest's archetypes[] entry): a
  // personality_ranges box per TRAITS axis, plus a name pool for
  // generated instances. Not itself an NPC — spawnInstance() resolves
  // one into a concrete NPC.
  struct PersonalityRange { int lo, hi; };

  struct Archetype
  {
    const char *idPrefix;                  // e.g. "guard" — the N: tag's
                                            // archetype half, before '#'
    PersonalityRange ranges[TRAIT_COUNT];  // keyed by TRAITS, same order
    const char *const *namePool;
    int namePoolSize;
    const char *occupation;    // M10: one of NpcService::OCCUPATIONS,
                                // lowercase — lets spawnInstance() feed
                                // NpcService::buildPromptFields() directly,
                                // instead of every new archetype needing
                                // its own bespoke ContextBuilder-style wiring
    PersonalityRange ageRange; // M10: inclusive, years — jittered same as
                                // the personality traits
  };

  extern const Archetype GUARD_ARCHETYPE;

  // M10: town archetypes, pulled forward from M11's originally-planned
  // cast (docs/milestones/m11.md) -- same mechanism as GUARD_ARCHETYPE,
  // just a different personality range/name pool/occupation. See
  // NPCDatabase.cpp for the ranges and reasoning.
  extern const Archetype PUB_PATRON_ARCHETYPE;
  extern const Archetype BLACKSMITH_ARCHETYPE;
  extern const Archetype WIZARD_ARCHETYPE;   // town tinker-wizard, not Shadewrath
  extern const Archetype VILLAGER_ARCHETYPE; // generic townsfolk incl. elders

  // M8 task #11: the fixed set of guard instances the demo/game actually
  // ships with (guard_corpus.py's GUARD_IDS — the model was only ever
  // trained on these 4 seeds, per M8's "fixed-set, not runtime-generalizing"
  // design). guardInstances[] starts default-constructed (id[0]=='\0');
  // initGuardInstances() must run once (e.g. from initDelete) before any
  // code reads it.
  constexpr int GUARD_INSTANCE_COUNT = 4;
  constexpr uint32_t GUARD_SEEDS[GUARD_INSTANCE_COUNT] = {
    0x1001, 0x1002, 0x1003, 0x1004,
  };
  extern NPC guardInstances[GUARD_INSTANCE_COUNT];
  void initGuardInstances();

  // Deterministically resolves archetype+seed into a concrete instance:
  // xorshift32 jitter per trait (same RNG discipline as core/'s sampler,
  // see core/ngpt_sample.cpp), a name drawn from the archetype's pool,
  // and an id formatted as "<idPrefix>#<4 lowercase hex digits of seed>".
  // seed 0 remaps to 1 (xorshift32's fixed point, same rule as
  // core/ngpt.cpp's ngpt_reset). Two calls with the same seed are
  // byte-identical; different seeds land at different but reproducible
  // points inside the archetype's ranges.
  NPC spawnInstance(const Archetype &archetype, uint32_t seed);
}
