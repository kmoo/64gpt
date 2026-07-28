/* Living-NPC-state spike (docs/ideas-living-npc-state.md #6 step 1): host
 * coverage for NPCState.h's pure relationship/memory update logic. Same
 * ADR 0001 treatment as test_save_data.cpp -- header-only, no libdragon
 * in the include chain, no real EEPROM/SD I/O (that wiring is future,
 * hardware-only work per the ideas doc itself). */
#include "NPCState.h"
#include "test_util.h"

using namespace NPCState;

static void test_apply_delta_adds_and_clamps()
{
  Relationship rel{50, 0, 50, 50, 50};
  applyDelta(rel, 10, 10, -10, 5, -5);
  CHECK_EQ_INT(rel.familiarity, 60);
  CHECK_EQ_INT(rel.affection, 10);
  CHECK_EQ_INT(rel.trust, 40);
  CHECK_EQ_INT(rel.respect, 55);
  CHECK_EQ_INT(rel.fear, 45);

  /* upper clamp on every 0..100 axis */
  Relationship high{95, 95, 95, 95, 95};
  applyDelta(high, 50, 50, 50, 50, 50);
  CHECK_EQ_INT(high.familiarity, 100);
  CHECK_EQ_INT(high.affection, 100);
  CHECK_EQ_INT(high.trust, 100);
  CHECK_EQ_INT(high.respect, 100);
  CHECK_EQ_INT(high.fear, 100);

  /* lower clamp: familiarity/trust/respect/fear floor at 0 */
  Relationship low{5, -95, 5, 5, 5};
  applyDelta(low, -50, -50, -50, -50, -50);
  CHECK_EQ_INT(low.familiarity, 0);
  CHECK_EQ_INT(low.trust, 0);
  CHECK_EQ_INT(low.respect, 0);
  CHECK_EQ_INT(low.fear, 0);

  /* affection's floor is -100, not 0 -- the one axis that goes negative */
  CHECK_EQ_INT(low.affection, -100);
}

static void test_record_memory_fills_empty_slots_in_index_order()
{
  MemoryBlock block{};
  recordMemory(block, 101, 50, 80);
  recordMemory(block, 102, 60, 80);

  CHECK_EQ_INT(block.slots[0].eventId, 101);
  CHECK_EQ_INT(block.slots[1].eventId, 102);
  for (int i = 2; i < MEMORY_SLOTS; ++i)
    CHECK_EQ_INT(block.slots[i].eventId, EMPTY_EVENT_ID);
}

static void test_record_memory_evicts_lowest_salience_tie_by_oldest_age()
{
  MemoryBlock block{};
  /* fill all 8 slots; slots 2 and 5 tie for lowest salience (10), but
   * slot 5 is older (higher ageTicks) -- it must be the one evicted */
  for (int i = 0; i < MEMORY_SLOTS; ++i) {
    block.slots[i].eventId = 200 + i;
    block.slots[i].salience = 50;
    block.slots[i].confidence = 50;
    block.slots[i].ageTicks = 100;
  }
  block.slots[2].salience = 10;
  block.slots[2].ageTicks = 50;
  block.slots[5].salience = 10;
  block.slots[5].ageTicks = 999; // oldest of the tied pair

  recordMemory(block, 999, 77, 88);

  CHECK_EQ_INT(block.slots[5].eventId, 999); // evicted
  CHECK_EQ_INT(block.slots[5].salience, 77);
  CHECK_EQ_INT(block.slots[5].ageTicks, 0);
  CHECK_EQ_INT(block.slots[2].eventId, 202); // untouched, not evicted
}

static void test_age_memories_advances_and_decays_with_integer_division()
{
  MemoryBlock block{};
  block.slots[0].eventId = 301;
  block.slots[0].salience = 50;
  block.slots[0].ageTicks = 0;

  ageMemories(block, 25); // 25 / 10 = 2 (integer division, not exact multiple)
  CHECK_EQ_INT(block.slots[0].ageTicks, 25);
  CHECK_EQ_INT(block.slots[0].salience, 48);

  ageMemories(block, 1000); // would go deeply negative without the floor
  CHECK_EQ_INT(block.slots[0].salience, 0);
}

static void test_age_memories_leaves_empty_slots_untouched()
{
  MemoryBlock block{};
  block.slots[3].ageTicks = 42;
  block.slots[3].salience = 7; // slot 3 stays EMPTY_EVENT_ID (0)

  ageMemories(block, 1000);

  CHECK_EQ_INT(block.slots[3].ageTicks, 42);
  CHECK_EQ_INT(block.slots[3].salience, 7);
}

static void test_select_top_memories_sorted_with_salience_tie_by_age()
{
  MemoryBlock block{};
  block.slots[0] = Memory{401, 50, 0, 100}; // tied salience, older
  block.slots[1] = Memory{402, 90, 0, 0};   // highest salience
  block.slots[2] = Memory{403, 50, 0, 10};  // tied salience, newer -> wins tie

  uint32_t out[MEMORY_SLOTS];
  int n = selectTopMemories(block, 3, out);

  CHECK_EQ_INT(n, 3);
  CHECK_EQ_INT(out[0], 402); // highest salience first
  CHECK_EQ_INT(out[1], 403); // tie broken by lower (newer) ageTicks
  CHECK_EQ_INT(out[2], 401);
}

static void test_select_top_memories_returns_fewer_than_n_when_sparse()
{
  MemoryBlock block{};
  block.slots[0] = Memory{501, 10, 0, 0};
  block.slots[4] = Memory{502, 20, 0, 0};
  /* only 2 of 8 slots occupied */

  uint32_t out[MEMORY_SLOTS];
  int n = selectTopMemories(block, 5, out);

  CHECK_EQ_INT(n, 2);
  CHECK_EQ_INT(out[0], 502);
  CHECK_EQ_INT(out[1], 501);
}

static void test_resolve_belief_id_below_and_above_threshold()
{
  Profile profile{100, 200, 999};

  Relationship low{0, 0, PRIVATE_BELIEF_TRUST_THRESHOLD - 1, 0, 0};
  CHECK_EQ_INT(resolveBeliefId(profile, low), 100);

  Relationship atThreshold{0, 0, PRIVATE_BELIEF_TRUST_THRESHOLD, 0, 0};
  CHECK_EQ_INT(resolveBeliefId(profile, atThreshold), 200);

  Relationship high{0, 0, 100, 0, 0};
  CHECK_EQ_INT(resolveBeliefId(profile, high), 200);
}

static void test_resolve_belief_id_reverts_if_trust_drops()
{
  /* No one-way "has revealed" flag -- a betrayal that drops trust back
   * below the threshold must correctly revert to the public belief. */
  Profile profile{10, 20, 0};
  Relationship rel{0, 0, PRIVATE_BELIEF_TRUST_THRESHOLD, 0, 0};
  CHECK_EQ_INT(resolveBeliefId(profile, rel), 20);

  rel.trust = PRIVATE_BELIEF_TRUST_THRESHOLD - 1;
  CHECK_EQ_INT(resolveBeliefId(profile, rel), 10);
}

static void test_propagate_gossip_degrades_confidence_carries_salience()
{
  Memory source{701, 80, 100, 0}; /* eventId, salience, confidence, ageTicks */
  MemoryBlock target{};
  propagateGossip(source, target);

  CHECK_EQ_INT(target.slots[0].eventId, 701);
  CHECK_EQ_INT(target.slots[0].salience, 80);   /* unchanged */
  CHECK_EQ_INT(target.slots[0].confidence, 75); /* 100 - 25% */
  CHECK_EQ_INT(target.slots[0].ageTicks, 0);    /* fresh in the new pool */
}

static void test_propagate_gossip_multi_hop_compounds()
{
  Memory hop1{801, 50, 100, 0};
  MemoryBlock npcB{};
  propagateGossip(hop1, npcB); /* 100 -> 75 */

  Memory hop2 = npcB.slots[0]; /* B re-gossips what it heard */
  MemoryBlock npcC{};
  propagateGossip(hop2, npcC); /* 75 -> 75 - 18 = 57 */

  CHECK_EQ_INT(npcC.slots[0].confidence, 57);
  CHECK_EQ_INT(npcC.slots[0].salience, 50); /* still unchanged after 2 hops */
}

static void test_propagate_gossip_zero_confidence_stays_zero()
{
  Memory source{901, 30, 0, 0};
  MemoryBlock target{};
  propagateGossip(source, target);
  CHECK_EQ_INT(target.slots[0].confidence, 0);
}

static void test_propagate_gossip_uses_normal_eviction_rules()
{
  /* Propagating into a full pool follows recordMemory()'s own eviction
   * rule -- not special-cased for gossip. */
  MemoryBlock target{};
  for (int i = 0; i < MEMORY_SLOTS; ++i) {
    target.slots[i].eventId = 1000 + i;
    target.slots[i].salience = 50;
    target.slots[i].confidence = 50;
    target.slots[i].ageTicks = 0;
  }
  target.slots[3].salience = 5; /* uniquely lowest -- must be the one evicted */

  Memory gossip{999, 60, 80, 0};
  propagateGossip(gossip, target);

  CHECK_EQ_INT(target.slots[3].eventId, 999);
}

static void test_apply_event_reaction_known_event()
{
  Relationship rel{50, 0, 50, 50, 50};
  bool applied = applyEventReaction(rel, "princess_freed");
  CHECK(applied);
  CHECK_EQ_INT(rel.familiarity, 65);
  CHECK_EQ_INT(rel.affection, 25);
  CHECK_EQ_INT(rel.trust, 70);
  CHECK_EQ_INT(rel.respect, 70);
  CHECK_EQ_INT(rel.fear, 35);
}

static void test_apply_event_reaction_unknown_event_is_noop()
{
  Relationship rel{50, 0, 50, 50, 50};
  bool applied = applyEventReaction(rel, "no_such_event");
  CHECK(!applied);
  CHECK_EQ_INT(rel.familiarity, 50);
  CHECK_EQ_INT(rel.affection, 0);
  CHECK_EQ_INT(rel.trust, 50);
  CHECK_EQ_INT(rel.respect, 50);
  CHECK_EQ_INT(rel.fear, 50);
}

static void test_apply_event_reaction_clamps_like_applyDelta()
{
  Relationship rel{95, 90, 90, 95, 5};
  applyEventReaction(rel, "princess_freed"); /* would overshoot 100 on several axes */
  CHECK_EQ_INT(rel.familiarity, 100);
  CHECK_EQ_INT(rel.affection, 100);
  CHECK_EQ_INT(rel.trust, 100);
  CHECK_EQ_INT(rel.respect, 100);
  CHECK_EQ_INT(rel.fear, 0);
}

static void test_memory_retrieval_score_extremes()
{
  Memory m{1, 100, 0, 0}; /* fresh, max salience */
  CHECK_EQ_INT(memoryRetrievalScore(m, 1000, 100), 100); /* 100*100*100/10000 */

  Memory old{1, 100, 0, 1000}; /* at the max-age horizon exactly */
  CHECK_EQ_INT(memoryRetrievalScore(old, 1000, 100), 0);

  Memory irrelevant{1, 100, 0, 0};
  CHECK_EQ_INT(memoryRetrievalScore(irrelevant, 1000, 0), 0);
}

static void test_memory_retrieval_score_clamps_relevance()
{
  Memory m{1, 100, 0, 0};
  CHECK_EQ_INT(memoryRetrievalScore(m, 1000, 999), memoryRetrievalScore(m, 1000, 100));
  CHECK_EQ_INT(memoryRetrievalScore(m, 1000, -5), memoryRetrievalScore(m, 1000, 0));
}

static void test_select_by_relevance_orders_by_combined_score()
{
  MemoryBlock block{};
  block.slots[0] = Memory{401, 50, 0, 0};   /* low salience, but... */
  block.slots[1] = Memory{402, 100, 0, 0};  /* highest salience+recency */
  block.slots[2] = Memory{403, 100, 0, 900}; /* high salience, nearly aged out */

  int relevance[MEMORY_SLOTS] = {100, 100, 100, 0, 0, 0, 0, 0};

  uint32_t out[MEMORY_SLOTS];
  int n = selectByRelevance(block, relevance, 1000, 3, out);

  CHECK_EQ_INT(n, 3);
  CHECK_EQ_INT(out[0], 402); /* highest salience, freshest */
  CHECK_EQ_INT(out[1], 401); /* mid: lower salience but fully fresh beats... */
  CHECK_EQ_INT(out[2], 403); /* ...high salience but nearly fully aged out */
}

static void test_select_by_relevance_zero_relevance_excludes_via_zero_score()
{
  MemoryBlock block{};
  block.slots[0] = Memory{501, 100, 0, 0};
  block.slots[1] = Memory{502, 100, 0, 0};
  int relevance[MEMORY_SLOTS] = {100, 0, 0, 0, 0, 0, 0, 0};

  uint32_t out[MEMORY_SLOTS];
  int n = selectByRelevance(block, relevance, 1000, 2, out);

  CHECK_EQ_INT(n, 2); /* both occupied slots still returned (n=2 requested)... */
  CHECK_EQ_INT(out[0], 501); /* ...but the relevant one is ranked first */
  CHECK_EQ_INT(out[1], 502);
}

int main()
{
  test_apply_delta_adds_and_clamps();
  test_record_memory_fills_empty_slots_in_index_order();
  test_record_memory_evicts_lowest_salience_tie_by_oldest_age();
  test_age_memories_advances_and_decays_with_integer_division();
  test_age_memories_leaves_empty_slots_untouched();
  test_select_top_memories_sorted_with_salience_tie_by_age();
  test_select_top_memories_returns_fewer_than_n_when_sparse();
  test_memory_retrieval_score_extremes();
  test_memory_retrieval_score_clamps_relevance();
  test_select_by_relevance_orders_by_combined_score();
  test_select_by_relevance_zero_relevance_excludes_via_zero_score();
  test_resolve_belief_id_below_and_above_threshold();
  test_resolve_belief_id_reverts_if_trust_drops();
  test_apply_event_reaction_known_event();
  test_apply_event_reaction_unknown_event_is_noop();
  test_apply_event_reaction_clamps_like_applyDelta();
  test_propagate_gossip_degrades_confidence_carries_salience();
  test_propagate_gossip_multi_hop_compounds();
  test_propagate_gossip_zero_confidence_stays_zero();
  test_propagate_gossip_uses_normal_eviction_rules();
  return test_summary("test_npc_state");
}
