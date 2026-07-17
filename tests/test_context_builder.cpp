/* M7 game-side subsystem tests: EventBus, WorldState, NPCDatabase,
 * ContextBuilder. These are pure portable C++ (no libdragon includes),
 * so unlike DialogueDemo.cpp they compile and run in the host suite —
 * same red/green discipline the project uses for core/, extended to the
 * new game-side layer docs/milestones/m7.md adds. */
#include "EventBus.h"
#include "WorldState.h"
#include "NPCDatabase.h"
#include "ContextBuilder.h"
#include "test_util.h"
#include <string.h>

static void test_event_bus()
{
  EventBus::reset();
  CHECK_EQ_INT(EventBus::eventCount(), 0);
  CHECK(strcmp(EventBus::lastTag(), "") == 0);

  EventBus::publish("found_gem");
  CHECK_EQ_INT(EventBus::eventCount(), 1);
  CHECK(strcmp(EventBus::lastTag(), "found_gem") == 0);

  EventBus::publish("player_damaged");
  CHECK_EQ_INT(EventBus::eventCount(), 2);
  CHECK(strcmp(EventBus::lastTag(), "player_damaged") == 0);

  /* history wraps past HISTORY_DEPTH without crashing; lastTag() always
   * reflects the most recent publish regardless of wrap */
  for(int i = 0; i < 10; ++i)EventBus::publish("wrap_test");
  CHECK(strcmp(EventBus::lastTag(), "wrap_test") == 0);
  CHECK_EQ_INT(EventBus::eventCount(), 12);

  /* a tag at exactly MAX_TAG_LEN-1 chars round-trips; longer truncates
   * safely instead of overflowing the fixed buffer */
  EventBus::publish("012345678901234567890123456789"); /* 31 chars, cap is 24 */
  CHECK_EQ_INT(strlen(EventBus::lastTag()), EventBus::MAX_TAG_LEN - 1);

  EventBus::reset();
  CHECK_EQ_INT(EventBus::eventCount(), 0);
  CHECK(strcmp(EventBus::lastTag(), "") == 0);
}

static void test_world_state()
{
  /* defaults to the first declared context category */
  CHECK(strcmp(WorldState::currentContext(), NPCDatabase::CONTEXTS[0]) == 0);

  WorldState::setContext("item-found");
  CHECK(strcmp(WorldState::currentContext(), "item-found") == 0);

  WorldState::setContext(NPCDatabase::CONTEXTS[0]); /* restore for other tests */
}

static void test_npc_database()
{
  CHECK(strcmp(NPCDatabase::selena.id, "selena") == 0);
  CHECK_EQ_INT(NPCDatabase::MOOD_COUNT, 5);
  CHECK_EQ_INT(NPCDatabase::CONTEXT_COUNT, 8);
  CHECK(strcmp(NPCDatabase::MOODS[0], "cheerful") == 0);
  CHECK(strcmp(NPCDatabase::CONTEXTS[0], "greeting") == 0);
}

static void test_context_builder()
{
  NPCDatabase::NPC npc{"selena", 2, 0}; /* trust tier 2, mood[0]=cheerful */
  char out[64];

  uint32_t len = ContextBuilder::build(out, sizeof(out), npc, "item-found", "found_gem");
  CHECK(strcmp(out, "N:selena TR:2 M:cheerful C:item-found EV:found_gem|") == 0);
  CHECK_EQ_INT(len, strlen(out));

  /* schema tag syntax pinned exactly as docs/milestones/m7.md specifies:
   * N:<id> TR:<tier> M:<mood> C:<context> EV:<event>| */
  CHECK(out[len - 1] == '|');

  /* no event yet (idle NPC) still yields a valid, parseable EV: field —
   * the corpus must define a trainable "no event" sentinel, not emit an
   * empty/dangling EV: */
  ContextBuilder::build(out, sizeof(out), npc, "greeting", "");
  CHECK(strcmp(out, "N:selena TR:2 M:cheerful C:greeting EV:none|") == 0);

  /* every produced string stays inside the M7 prime-time budget's
   * target ceiling (64 bytes) for realistic field values */
  CHECK(len < 64);

  /* a too-small buffer truncates safely (snprintf semantics) rather
   * than overflowing — build() must never write past outCap */
  char tiny[8];
  uint32_t tlen = ContextBuilder::build(tiny, sizeof(tiny), npc, "greeting", "x");
  CHECK(tlen < sizeof(tiny));
  CHECK(strlen(tiny) == tlen);
}

int main()
{
  test_event_bus();
  test_world_state();
  test_npc_database();
  test_context_builder();
  return test_summary("test_context_builder");
}
