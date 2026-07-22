#include "NPCDatabase.h"
#include <stdio.h>

namespace NPCDatabase
{
  const char *const MOODS[MOOD_COUNT] = {
    "cheerful", "worried", "sassy", "tender", "embarrassed",
  };

  const char *const CONTEXTS[CONTEXT_COUNT] = {
    "greeting", "combat-banter", "item-found", "damage-taken",
    "quiet-moment", "joke", "encouragement", "farewell",
  };

  const char *const TRAITS[TRAIT_COUNT] = {
    "warmth", "humor", "impulsivity", "bravery", "focus",
  };

  NPC selena{"selena", "", /*trustTier=*/0, /*moodIdx=*/0,
             /*personality=*/{90, 85, 70, 55, 30}, /*memorySlot=*/0,
             /*tier=*/Tier::FULL};

  // M10: traits match manifests/dungeon_crawler.json's characters[]
  // entries exactly (same discipline selena's own values follow).
  NPC shadewrath{"shadewrath", "SHADEWRATH", /*trustTier=*/0, /*moodIdx=*/0,
                  /*personality=*/{8, 20, 12, 88, 95}, /*memorySlot=*/0,
                  /*tier=*/Tier::FULL};
  NPC korrath{"korrath", "KORRATH", /*trustTier=*/0, /*moodIdx=*/0,
              /*personality=*/{38, 10, 10, 75, 80}, /*memorySlot=*/0,
              /*tier=*/Tier::MID};

  // M11: Elowen, the rescued elf princess -- traits match trainer/
  // ngpt_trainer/princess_corpus.py's module docstring and manifests/
  // dungeon_crawler.json's characters[] entry exactly.
  NPC princess{"elowen", "ELOWEN", /*trustTier=*/0, /*moodIdx=*/0,
               /*personality=*/{78, 45, 55, 60, 50}, /*memorySlot=*/0,
               /*tier=*/Tier::MID};

  // Placeholder name pool: manifests/dungeon_crawler.json's guard entry
  // leaves name_gen unset pending real corpus/voice work (M8 task #10);
  // this is the mechanism it will eventually be told to draw from, not a
  // final cast list. Uppercase, matching the debug font's glyph range —
  // same convention as response TEXT (docs/milestones/m7.md).
  static const char *const GUARD_NAMES[] = {
    "BRAM", "CORVIN", "DESMOND", "EDRIC",
    "FENWICK", "GARRICK", "HALVOR", "IVOR",
  };

  // First-pass sketch, not yet trained against — see
  // manifests/dungeon_crawler.json's guard._status.
  const Archetype GUARD_ARCHETYPE{
    "guard",
    {
      {20, 45},  // warmth
      {5, 30},   // humor
      {10, 35},  // impulsivity
      {60, 90},  // bravery
      {55, 85},  // focus
    },
    GUARD_NAMES,
    sizeof(GUARD_NAMES) / sizeof(GUARD_NAMES[0]),
    "guard",     // occupation (M10) -- unused by ContextBuilder's existing
                 // N: scheme, harmless to set; only new archetypes routed
                 // through NpcService actually consume it
    {25, 55},    // ageRange (M10)
  };

  // M10: town archetypes (docs/milestones/m11.md's originally-planned
  // cast, pulled forward alongside Shadewrath since it's cheap content
  // against an already-proven mechanism -- no new engine work per
  // archetype, just a personality range + name pool + occupation, same
  // shape as GUARD_ARCHETYPE). First-pass sketches, not yet trained
  // against, same caveat as guard's own ranges above. Kid-appropriate:
  // ranges are tuned for whimsical/goofy, not actually inebriated or
  // romantic -- pub_patron's high-impulsivity end is silliness, not
  // drunkenness.
  //
  // Name pools are a single shared list per archetype, not gender-split
  // (spawnInstance()'s name draw doesn't currently look at isFemale) --
  // same simplification guard already had before M10 added gender at
  // all; a real fix is a bigger change than one archetype's content
  // warrants right now.

  static const char *const PUB_PATRON_NAMES[] = {
    "WILLA", "TOBIN", "MAISIE", "OSWIN", "PIPPA", "BARNABY", "RUBY", "FINNIAN",
  };
  const Archetype PUB_PATRON_ARCHETYPE{
    "patron",
    {
      {55, 85},  // warmth -- friendly, chatty regulars
      {55, 90},  // humor -- high energy, jokey
      {45, 85},  // impulsivity -- spans measured to giddy/goofy
      {40, 75},  // bravery -- moderate, not fighters
      {20, 55},  // focus -- easily distracted, mid-story tangents
    },
    PUB_PATRON_NAMES,
    sizeof(PUB_PATRON_NAMES) / sizeof(PUB_PATRON_NAMES[0]),
    "pub_patron",
    {19, 70},
  };

  static const char *const BLACKSMITH_NAMES[] = {
    "GRETA", "THORNE", "MABEL", "BRUNO", "HELGA", "ORSON", "AGATHA", "DUNCAN",
  };
  const Archetype BLACKSMITH_ARCHETYPE{
    "smith",
    {
      {20, 45},  // warmth -- gruff but not unkind
      {15, 40},  // humor -- dry, not jokers
      {15, 40},  // impulsivity -- steady, careful with hot metal
      {65, 90},  // bravery -- unflinching, strong
      {65, 90},  // focus -- absorbed in the craft
    },
    BLACKSMITH_NAMES,
    sizeof(BLACKSMITH_NAMES) / sizeof(BLACKSMITH_NAMES[0]),
    "blacksmith",
    {28, 65},
  };

  // Friendly town tinker-wizard -- distinct from Shadewrath (the
  // recurring necromancer villain, tier=full, not an archetype). Keeping
  // "wizard" here per Luke's call: the town archetype stays the
  // approachable one, the villain got renamed instead.
  static const char *const WIZARD_NAMES[] = {
    "MERRIT", "ZEPHYRA", "QUILL", "ORLA", "FIZZLE", "PELLIN", "NIMBLE", "SAGE",
  };
  const Archetype WIZARD_ARCHETYPE{
    "tinker",
    {
      {45, 75},  // warmth -- kindly, approachable
      {40, 70},  // humor -- whimsical
      {40, 75},  // impulsivity -- eccentric, unpredictable ideas
      {30, 60},  // bravery -- cerebral, not a fighter
      {55, 85},  // focus -- absorbed in study/tinkering
    },
    WIZARD_NAMES,
    sizeof(WIZARD_NAMES) / sizeof(WIZARD_NAMES[0]),
    "wizard",
    {24, 80},
  };

  // Generic misc. townsfolk -- wide age range deliberately covers "a few
  // funny old people" (60+ lands the elderly_man/elderly_woman age-gender
  // token, m10.md's requested cast) without a dedicated elder archetype.
  static const char *const VILLAGER_NAMES[] = {
    "TAM", "NELLA", "COLM", "IVY", "PERRIN", "DOT", "WREN", "OTIS",
  };
  const Archetype VILLAGER_ARCHETYPE{
    "villager",
    {
      {40, 80},  // warmth
      {40, 85},  // humor -- skews high, these are the "funny" townsfolk
      {30, 70},  // impulsivity
      {20, 60},  // bravery
      {30, 70},  // focus
    },
    VILLAGER_NAMES,
    sizeof(VILLAGER_NAMES) / sizeof(VILLAGER_NAMES[0]),
    "villager",
    {8, 85},
  };

  // M11: Briar Glen's General Store keeper -- shrewd, practical, good
  // with numbers. Trait box tuned to land unambiguously on
  // personality_descriptor()'s "measured" bucket (impulsivity<35 AND
  // focus>=60) for every sample in range, with warmth kept >=45 so no
  // sample can accidentally trip the earlier "stoic" rule (focus>=70 &&
  // warmth<45) instead -- same calibration discipline
  // trainer/tests/test_cast_corpus.py's _CANON_DESCRIPTOR enforces.
  static const char *const MERCHANT_NAMES[] = {
    "COSMO", "PENNY", "AUGUSTUS", "MARIGOLD", "FLETCHER", "HATTIE",
    "BARTHOLOMEW", "CLOVER",
  };
  const Archetype MERCHANT_ARCHETYPE{
    "merchant",
    {
      {45, 65},  // warmth -- friendly enough for trade, not gushing
      {30, 55},  // humor -- dry wit
      {15, 34},  // impulsivity -- careful, calculating
      {30, 55},  // bravery -- not a fighter
      {65, 90},  // focus -- sharp with the ledger
    },
    MERCHANT_NAMES,
    sizeof(MERCHANT_NAMES) / sizeof(MERCHANT_NAMES[0]),
    "merchant",
    {28, 65},
  };

  // M11: Briar Glen's Herbalist -- gentle, warm, attentive. Trait box
  // tuned to land unambiguously on personality_descriptor()'s "gentle"
  // bucket (warmth>=70 AND bravery<45) for every sample in range.
  static const char *const HEALER_NAMES[] = {
    "WILLOW", "ROSEMARY", "BASIL", "MYRTLE", "FERN", "HAWTHORNE",
    "CLEMENTINE", "PRIMROSE",
  };
  const Archetype HEALER_ARCHETYPE{
    "healer",
    {
      {70, 90},  // warmth -- caring, the point of the role
      {30, 55},  // humor -- gentle, not sassy
      {15, 40},  // impulsivity -- careful with remedies
      {15, 40},  // bravery -- not a fighter
      {60, 85},  // focus -- attentive to whoever's in front of them
    },
    HEALER_NAMES,
    sizeof(HEALER_NAMES) / sizeof(HEALER_NAMES[0]),
    "healer",
    {30, 75},
  };

  static uint32_t xorshift32(uint32_t x)
  {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
  }

  static int jitter(uint32_t &rng, PersonalityRange range)
  {
    rng = xorshift32(rng);
    uint32_t span = (uint32_t)(range.hi - range.lo + 1);
    return range.lo + (int)(rng % span);
  }

  NPC spawnInstance(const Archetype &archetype, uint32_t seed)
  {
    uint32_t rng = seed != 0 ? seed : 1; // xorshift32's fixed point, same
                                          // remap as core/ngpt.cpp's ngpt_reset

    NPC npc{};
    for(int i = 0; i < TRAIT_COUNT; ++i)
      npc.personality[i] = jitter(rng, archetype.ranges[i]);

    rng = xorshift32(rng);
    int nameIdx = archetype.namePoolSize > 0
      ? (int)(rng % (uint32_t)archetype.namePoolSize) : 0;
    snprintf(npc.name, sizeof(npc.name), "%s",
             archetype.namePoolSize > 0 ? archetype.namePool[nameIdx] : "");

    snprintf(npc.id, sizeof(npc.id), "%s#%04x", archetype.idPrefix,
              (unsigned)(seed & 0xFFFFu));

    npc.trustTier = 0;
    npc.moodIdx = 0;
    npc.memorySlot = 0; // empty until M9+ wires actual memory persistence
    npc.tier = Tier::THIN; // every archetype-spawned instance is thin tier
                            // (m10.md: archetypes[] entries are generators,
                            // not hand-authored individuals)

    npc.occupation = archetype.occupation;
    npc.age = jitter(rng, archetype.ageRange);
    rng = xorshift32(rng);
    npc.isFemale = (rng & 1u) != 0; // coin flip off the same RNG stream --
                                     // deterministic given the seed, same
                                     // as every other jittered field here

    return npc;
  }

  NPC guardInstances[GUARD_INSTANCE_COUNT]{};

  void initGuardInstances()
  {
    for(int i = 0; i < GUARD_INSTANCE_COUNT; ++i)
      guardInstances[i] = spawnInstance(GUARD_ARCHETYPE, GUARD_SEEDS[i]);
  }
}
