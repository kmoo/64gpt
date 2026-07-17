#pragma once
#include <stdint.h>
#include "NPCDatabase.h"

// Context Builder: turns NPC Database state + an event tag into the M7
// conditioning string the frozen ngpt_reset(ctx, model, prompt) API
// primes on (core/'s streaming API is untouched — this only changes
// what string the game builds, per docs/milestones/m7.md). Schema fixed
// there: "N:<id> TR:<tier> M:<mood> C:<context> EV:<event>|"
//
// Portable C++, no libdragon includes — builds in the host test suite.
namespace ContextBuilder
{
  // Writes the conditioning string into out (must be >= 64 bytes — the
  // M7 prime-time budget's target ceiling). NUL-terminates. Returns the
  // written length excluding the NUL. event may be "" (an idle NPC with
  // no recent event still needs a valid, parseable EV: field).
  uint32_t build(char *out, uint32_t outCap, const NPCDatabase::NPC &npc,
                 const char *context, const char *event);
}
