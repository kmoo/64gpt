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

// M10: the three-tier cast system (full/mid/thin, docs/08-manifest-
// schema.md, docs/milestones/m10.md section 1) needs an actual field in
// the NPC Database, not just a manifest convention nothing reads.
static void test_npc_tier()
{
  /* Selena is a characters[]-shaped, hand-authored individual: full tier. */
  CHECK(NPCDatabase::selena.tier == NPCDatabase::Tier::FULL);

  /* Archetype-spawned instances are always thin tier -- ephemeral, no
   * durable memory beyond the current encounter (m10.md section 3). */
  NPCDatabase::NPC g = NPCDatabase::spawnInstance(NPCDatabase::GUARD_ARCHETYPE, 0x4f2a);
  CHECK(g.tier == NPCDatabase::Tier::THIN);

  /* full and mid tier NPCs persist across dungeon-loop iterations; thin
   * does not -- the explicit decision rule m10.md section 3 calls for,
   * not left implicit. */
  CHECK(NPCDatabase::isPersistent(NPCDatabase::Tier::FULL));
  CHECK(NPCDatabase::isPersistent(NPCDatabase::Tier::MID));
  CHECK(!NPCDatabase::isPersistent(NPCDatabase::Tier::THIN));
}

static void test_context_builder()
{
  NPCDatabase::NPC npc{"selena", "", 2, 0, {90, 85, 70, 55, 30}, 0}; /* trust tier 2, mood[0]=cheerful */
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

static void test_archetype_instance()
{
  /* deterministic: same seed -> byte-identical instance */
  NPCDatabase::NPC a = NPCDatabase::spawnInstance(NPCDatabase::GUARD_ARCHETYPE, 0x4f2a);
  NPCDatabase::NPC b = NPCDatabase::spawnInstance(NPCDatabase::GUARD_ARCHETYPE, 0x4f2a);
  CHECK(strcmp(a.id, b.id) == 0);
  CHECK(strcmp(a.name, b.name) == 0);
  for(int i = 0; i < NPCDatabase::TRAIT_COUNT; ++i)
    CHECK_EQ_INT(a.personality[i], b.personality[i]);

  /* id carries the archetype prefix + 4 lowercase hex digits of the seed */
  CHECK(strcmp(a.id, "guard#4f2a") == 0);

  /* every jittered trait lands inside the archetype's declared range */
  for(int i = 0; i < NPCDatabase::TRAIT_COUNT; ++i)
  {
    NPCDatabase::PersonalityRange r = NPCDatabase::GUARD_ARCHETYPE.ranges[i];
    CHECK(a.personality[i] >= r.lo);
    CHECK(a.personality[i] <= r.hi);
  }

  /* a different seed lands at a different (but still in-range) point --
   * not a hard guarantee for any single trait, but the id and name must
   * differ since they're keyed off the raw seed/derived rng state, not
   * off the personality values */
  NPCDatabase::NPC c = NPCDatabase::spawnInstance(NPCDatabase::GUARD_ARCHETYPE, 0x1b7c);
  CHECK(strcmp(a.id, c.id) != 0);

  /* seed 0 remaps to 1 rather than leaving xorshift32 stuck at its fixed
   * point (0 ^ anything == 0) -- same rule as core/ngpt.cpp's ngpt_reset */
  NPCDatabase::NPC zero = NPCDatabase::spawnInstance(NPCDatabase::GUARD_ARCHETYPE, 0);
  bool anyNonZero = false;
  for(int i = 0; i < NPCDatabase::TRAIT_COUNT; ++i)
    if(zero.personality[i] != 0)anyNonZero = true;
  CHECK(anyNonZero);

  /* fresh memory slot, per M8's "own NPC Database memory slot, initially
   * empty" -- M9+ gives this real meaning, M8 just reserves it */
  CHECK_EQ_INT((int)a.memorySlot, 0);
}

// M8 task #11: the fixed guard instance registry, checked against ground
// truth pulled from the real compiled spawnInstance() (same values
// trainer/tests/test_guard_instances.py cross-checks the Python port
// against — see that file's EXPECTED dict for provenance).
void test_guard_instance_registry()
{
  NPCDatabase::initGuardInstances();
  CHECK_EQ_INT(NPCDatabase::GUARD_INSTANCE_COUNT, 4);

  struct Want { const char *id; const char *name; int p[5]; };
  const Want want[4] = {
    {"guard#1001", "BRAM",   {43, 5, 35, 72, 73}},
    {"guard#1002", "EDRIC",  {42, 24, 16, 60, 80}},
    {"guard#1003", "EDRIC",  {33, 7, 19, 72, 56}},
    {"guard#1004", "IVOR",   {32, 18, 24, 84, 76}},
  };
  for(int i = 0; i < 4; ++i)
  {
    const NPCDatabase::NPC &npc = NPCDatabase::guardInstances[i];
    CHECK(strcmp(npc.id, want[i].id) == 0);
    CHECK(strcmp(npc.name, want[i].name) == 0);
    for(int t = 0; t < NPCDatabase::TRAIT_COUNT; ++t)
      CHECK_EQ_INT(npc.personality[t], want[i].p[t]);
  }
}

int main()
{
  test_event_bus();
  test_world_state();
  test_npc_database();
  test_npc_tier();
  test_context_builder();
  test_archetype_instance();
  test_guard_instance_registry();
  return test_summary("test_context_builder");
}
