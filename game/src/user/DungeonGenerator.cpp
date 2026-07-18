#include "DungeonGenerator.h"

namespace DungeonGenerator
{
  static uint32_t xorshift32(uint32_t x)
  {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
  }

  // The thin-tier archetype pool a dungeon level can draw from. Guard
  // included alongside the M10 town archetypes -- nothing about being
  // "the town guard" prevents an instance from also appearing in a
  // generated level; the archetype/instance split (M8) was always meant
  // to be reusable across placement contexts, not town-exclusive.
  static const NPCDatabase::Archetype *const ARCHETYPE_POOL[] = {
    &NPCDatabase::GUARD_ARCHETYPE, &NPCDatabase::PUB_PATRON_ARCHETYPE,
    &NPCDatabase::BLACKSMITH_ARCHETYPE, &NPCDatabase::WIZARD_ARCHETYPE,
    &NPCDatabase::VILLAGER_ARCHETYPE,
  };
  constexpr int ARCHETYPE_POOL_SIZE =
      sizeof(ARCHETYPE_POOL) / sizeof(ARCHETYPE_POOL[0]);

  void npcsForLevel(uint32_t levelSeed, NpcPlacement out[NPCS_PER_LEVEL])
  {
    uint32_t rng = levelSeed != 0 ? levelSeed : 1; // xorshift32's fixed
                                                     // point, same remap
                                                     // as spawnInstance()

    for(int i = 0; i < NPCS_PER_LEVEL; ++i)
    {
      rng = xorshift32(rng);
      int archIdx = (int)(rng % (uint32_t)ARCHETYPE_POOL_SIZE);
      out[i].archetype = ARCHETYPE_POOL[archIdx];

      rng = xorshift32(rng);
      out[i].instanceSeed = rng; // spawnInstance() itself remaps 0 -> 1,
                                  // no need to duplicate that rule here
    }
  }
}
