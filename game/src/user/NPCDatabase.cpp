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

    return npc;
  }

  NPC guardInstances[GUARD_INSTANCE_COUNT]{};

  void initGuardInstances()
  {
    for(int i = 0; i < GUARD_INSTANCE_COUNT; ++i)
      guardInstances[i] = spawnInstance(GUARD_ARCHETYPE, GUARD_SEEDS[i]);
  }
}
