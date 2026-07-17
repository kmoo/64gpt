#include "WorldState.h"
#include "NPCDatabase.h"

namespace WorldState
{
  namespace
  {
    const char *context = NPCDatabase::CONTEXTS[0];
  }

  const char *currentContext()
  {
    return context;
  }

  void setContext(const char *c)
  {
    context = c;
  }
}
