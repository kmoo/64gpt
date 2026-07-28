#pragma once
#include <stdint.h>

// Archetype spawn-seed-source strategy (docs/plan.md Known Follow-ups,
// raised 2026-07-17 during M8): NPCDatabase::spawnInstance(archetype,
// seed) is a pure function (same seed -> same guard forever, so a save
// file only needs to store a seed, not a character record) -- that
// determinism is correct and this header doesn't change it. What was
// missing is the SPAWN-TIME logic that decides WHICH of the fixed
// trained seeds (e.g. NPCDatabase::GUARD_SEEDS) gets used at a given
// dungeon level / spawn slot, so the world doesn't always place the
// same instances in the same order. This only covers CHOOSING AMONG an
// already-trained seed pool -- generalizing to arbitrary UNTRAINED
// seeds (true unlimited procedural variety) is explicitly out of scope,
// separate, harder, unproven work for M10's procedural spawning.
namespace SpawnSeedSource
{
  // Deterministic (levelId, slotIndex) -> index into a fixed-size seed
  // pool. Stateless and replayable: the same (levelId, slotIndex) always
  // resolves to the same index, so nothing about WHICH seed got chosen
  // needs to be saved -- levelId/slotIndex are already implicit in world
  // structure. A simple mixing hash, not real RNG state; doesn't need to
  // be statistically rigorous, only to avoid the degenerate "slot 0 of
  // every level gets seed pool[0]" pattern. Known limitation: this does
  // NOT guarantee no repeats within a single level across its own
  // slots -- two different slotIndex values can hash to the same pool
  // index. Acceptable for the pool sizes this project actually has (4
  // guard seeds); a derangement-style "each of the 4 appears exactly
  // once per level" guarantee would be a real, separate upgrade if ever
  // needed.
  inline uint32_t chooseSeedIndex(uint32_t levelId, uint32_t slotIndex, uint32_t poolSize)
  {
    if (poolSize == 0) return 0;
    uint32_t h = levelId * 2654435761u + slotIndex * 40503u; // Knuth multiplicative hash
    h ^= h >> 15;
    return h % poolSize;
  }

  // Resolves directly to the trained seed VALUE (not just its index) --
  // the form callers actually pass to NPCDatabase::spawnInstance().
  // seedPool/poolSize is caller-owned (e.g. NPCDatabase::GUARD_SEEDS,
  // NPCDatabase::GUARD_INSTANCE_COUNT) -- this header has no dependency
  // on NPCDatabase.h, matching the rest of this project's "extend via a
  // new pure header, don't touch the existing one" discipline.
  inline uint32_t chooseSpawnSeed(const uint32_t *seedPool, uint32_t poolSize,
                                   uint32_t levelId, uint32_t slotIndex)
  {
    if (poolSize == 0) return 0;
    return seedPool[chooseSeedIndex(levelId, slotIndex, poolSize)];
  }
}
