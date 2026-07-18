#pragma once
#include <stdint.h>
#include "NPCDatabase.h"

// M10: seeded thin-tier NPC instantiation for a procedurally-generated
// dungeon level. Scope note, since the name invites a bigger reading
// than this delivers: this derives WHICH thin-tier archetypes populate
// a level and each instance's spawn seed, deterministically from a
// single level seed. It does NOT generate room layout, tiles, or level
// geometry -- no such system exists anywhere in this project yet. This
// is the NPC-placement slice of "dungeon-level generation"
// (docs/milestones/m10.md's DoD), not a full level generator.
//
// Why one deterministic derivation matters (m10.md section 2 + section
// 4): if the level seed is already stored (for level regeneration), the
// same seed must reproduce the same NPC roster AND the same dialogue on
// reload -- free, since spawnInstance() is already a pure function of
// (archetype, seed). This module is the other half: turning ONE level
// seed into the (archetype, seed) pairs spawnInstance() needs, so save/
// load determinism holds without a separate persistence mechanism.
//
// Portable C++, no libdragon includes -- builds in the host test suite.
namespace DungeonGenerator
{
  constexpr int NPCS_PER_LEVEL = 4;

  struct NpcPlacement
  {
    const NPCDatabase::Archetype *archetype;
    uint32_t instanceSeed;
  };

  // Deterministically derives NPCS_PER_LEVEL (archetype, instance seed)
  // pairs from a single level seed. Same levelSeed -> byte-identical
  // placements every time (same xorshift32 discipline as spawnInstance()
  // itself); different seeds land on different but reproducible rosters.
  // seed 0 remaps to 1, same fixed-point rule as spawnInstance() and
  // core/ngpt.cpp's ngpt_reset.
  void npcsForLevel(uint32_t levelSeed, NpcPlacement out[NPCS_PER_LEVEL]);
}
