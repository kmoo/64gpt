/* M11: host coverage for SaveData.h's pure, header-only half --
 * MAX_TRUST_TIER and isNewHighWaterMark(). SaveFile/init()/recordXTier()
 * stay real-hardware-only (they need libdragon's eepromfs.h), but the
 * "would this be a new high-water mark" rule those functions rely on is
 * ordinary C++ math with no peripheral dependency, so ADR 0001
 * (docs/adr/0001-host-test-portable-cpp-separate-from-libdragon.md)
 * says it belongs in the fast host suite, not left to the slow
 * write-then-relaunch Ares proof alone. This test links against
 * SaveData.h only, not SaveData.cpp -- no eepromfs.h in the include
 * chain at all. */
#include "SaveData.h"
#include "test_util.h"

static void test_max_trust_tier_matches_npc_database_ceiling()
{
  /* NPCDatabase's trust tiers are always 0..2 (TR: field) -- pinned here
   * so a future change to one without the other fails loudly instead of
   * silently drifting apart. */
  CHECK_EQ_INT(SaveData::MAX_TRUST_TIER, 2);
}

static void test_is_new_high_water_mark()
{
  /* strictly higher -> true */
  CHECK(SaveData::isNewHighWaterMark(1, 0));
  CHECK(SaveData::isNewHighWaterMark(2, 1));
  CHECK(SaveData::isNewHighWaterMark(2, 0));

  /* equal -> false: re-reaching an already-recorded tier is not a NEW
   * high-water mark, matching recordShadewrathTier()/recordKorrathTier()'s
   * own "never lowers, and never re-fires on a re-visit" contract */
  CHECK(!SaveData::isNewHighWaterMark(0, 0));
  CHECK(!SaveData::isNewHighWaterMark(1, 1));
  CHECK(!SaveData::isNewHighWaterMark(2, 2));

  /* lower -> false: cycling trust back down (D-pad down) and back up to
   * an already-reached tier must not look like a new high-water mark */
  CHECK(!SaveData::isNewHighWaterMark(0, 1));
  CHECK(!SaveData::isNewHighWaterMark(1, 2));
  CHECK(!SaveData::isNewHighWaterMark(0, 2));
}

static void test_gossip_trigger_fires_exactly_once_across_a_realistic_sequence()
{
  /* Simulates DialogueDemo.cpp's actual D-pad sequence: 0 -> 1 -> 2 (first
   * time reaching max, gossip SHOULD fire) -> 1 -> 2 again (re-visit,
   * gossip must NOT re-fire) -- the exact scenario the "checked BEFORE
   * the save call, against the still-stored high-water mark" comment in
   * DialogueDemo.cpp depends on. */
  uint8_t storedHighest = 0;
  int gossipFires = 0;

  auto pressUp = [&](uint8_t newTier) {
    if(newTier == SaveData::MAX_TRUST_TIER &&
       SaveData::isNewHighWaterMark(newTier, storedHighest))
      ++gossipFires;
    if(SaveData::isNewHighWaterMark(newTier, storedHighest))
      storedHighest = newTier; // mirrors recordXTier()'s own persist step
  };

  pressUp(1);              // TR 0->1: not max yet, no gossip
  CHECK_EQ_INT(gossipFires, 0);
  pressUp(2);              // TR 1->2: first time at max -> gossip fires
  CHECK_EQ_INT(gossipFires, 1);
  CHECK_EQ_INT(storedHighest, 2);

  /* D-pad down then back up to 2 -- a re-visit, must not re-fire */
  storedHighest = 2; // recordXTier() never lowers the stored value even
                     // though the live trustTier dial itself can cycle down
  pressUp(1);
  CHECK_EQ_INT(gossipFires, 1);
  pressUp(2);
  CHECK_EQ_INT(gossipFires, 1);
}

int main()
{
  test_max_trust_tier_matches_npc_database_ceiling();
  test_is_new_high_water_mark();
  test_gossip_trigger_fires_exactly_once_across_a_realistic_sequence();
  return test_summary("test_save_data");
}
