#include "ContextBuilder.h"
#include <stdio.h>

namespace ContextBuilder
{
  uint32_t build(char *out, uint32_t outCap, const NPCDatabase::NPC &npc,
                 const char *context, const char *event)
  {
    const char *ev = (event && event[0]) ? event : "none";
    int n = snprintf(out, outCap, "N:%s TR:%d M:%s C:%s EV:%s|",
                     npc.id, npc.trustTier, NPCDatabase::MOODS[npc.moodIdx],
                     context, ev);
    if(n < 0)
    {
      if(outCap)out[0] = '\0';
      return 0;
    }
    uint32_t len = (uint32_t)n;
    return len < outCap ? len : outCap - 1; // snprintf truncated; report the actual written length
  }
}
