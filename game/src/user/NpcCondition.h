#pragma once
#include <stdint.h>

// NPC's own current condition (docs/ideas-m7-living-npcs.md Part 4:
// tired/injured/drunk/mid-task -- explicitly distinct from FIXED
// personality (NPCState::Profile) and from Part 1's "emotional
// residue" (reaction to the player specifically): this is the NPC's
// own independent state, e.g. "a normally-cheerful innkeeper reads
// differently after a long night"). GAME-STATE half only, same
// discipline as every other NPCState-adjacent header tonight.
namespace NpcCondition
{
  // Bitmask -- conditions can co-occur (tired AND injured is a
  // sensible NPC state, not a contradiction).
  enum Flag : uint8_t {
    NONE    = 0,
    TIRED   = 1 << 0,
    INJURED = 1 << 1,
    DRUNK   = 1 << 2,
    BUSY    = 1 << 3,
  };

  inline uint8_t applyFlag(uint8_t conditions, Flag flag) { return (uint8_t)(conditions | flag); }
  inline uint8_t clearFlag(uint8_t conditions, Flag flag) { return (uint8_t)(conditions & ~flag); }
  inline bool hasFlag(uint8_t conditions, Flag flag) { return (conditions & flag) != 0; }

  // Priority order for whichever single condition matters most when a
  // future caller needs to pick ONE token (e.g. for a conditioning
  // string) out of a possibly-multi-flag state -- INJURED reads as the
  // most narratively urgent, NONE last. Deriving an actual conditioning
  // token from this pick is corpus/schema work, explicitly out of
  // scope here.
  inline Flag dominantFlag(uint8_t conditions)
  {
    if (hasFlag(conditions, INJURED)) return INJURED;
    if (hasFlag(conditions, DRUNK)) return DRUNK;
    if (hasFlag(conditions, TIRED)) return TIRED;
    if (hasFlag(conditions, BUSY)) return BUSY;
    return NONE;
  }
}
