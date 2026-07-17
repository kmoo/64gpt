#include "EventBus.h"
#include <string.h>

namespace EventBus
{
  namespace
  {
    char history[HISTORY_DEPTH][MAX_TAG_LEN]{};
    uint32_t writeIdx{};
    uint32_t count{};
  }

  void publish(const char *tag)
  {
    char *slot = history[writeIdx % HISTORY_DEPTH];
    strncpy(slot, tag, MAX_TAG_LEN - 1);
    slot[MAX_TAG_LEN - 1] = '\0';
    ++writeIdx;
    ++count;
  }

  const char *lastTag()
  {
    if(count == 0)return "";
    return history[(writeIdx - 1) % HISTORY_DEPTH];
  }

  uint32_t eventCount()
  {
    return count;
  }

  void reset()
  {
    writeIdx = 0;
    count = 0;
    memset(history, 0, sizeof(history));
  }
}
