#include "NPCDatabase.h"

namespace NPCDatabase
{
  const char *const MOODS[MOOD_COUNT] = {
    "cheerful", "worried", "sassy", "tender", "embarrassed",
  };

  const char *const CONTEXTS[CONTEXT_COUNT] = {
    "greeting", "combat-banter", "item-found", "damage-taken",
    "quiet-moment", "joke", "encouragement", "farewell",
  };

  NPC selena{"selena", /*trustTier=*/0, /*moodIdx=*/0};
}
