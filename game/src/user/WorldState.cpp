#include "WorldState.h"
#include "NPCDatabase.h"

namespace WorldState
{
  namespace
  {
    const char *context = NPCDatabase::CONTEXTS[0];
    const char *gossip = "";
  }

  const char *const GOSSIP_EVENTS[GOSSIP_EVENT_COUNT] = {
    "shadewrath_allied", "korrath_pleaded",
  };

  const char *currentContext()
  {
    return context;
  }

  void setContext(const char *c)
  {
    context = c;
  }

  const char *currentGossip()
  {
    return gossip;
  }

  void setGossip(const char *tag)
  {
    gossip = tag ? tag : "";
  }
}
