/* NPC's own current condition (docs/ideas-m7-living-npcs.md Part 4),
 * game-state half only -- see NpcCondition.h's own header comment for
 * scope. */
#include "NpcCondition.h"
#include "test_util.h"

using namespace NpcCondition;

static void test_apply_and_has_flag()
{
  uint8_t c = NONE;
  CHECK(!hasFlag(c, TIRED));
  c = applyFlag(c, TIRED);
  CHECK(hasFlag(c, TIRED));
}

static void test_flags_can_co_occur()
{
  uint8_t c = NONE;
  c = applyFlag(c, TIRED);
  c = applyFlag(c, INJURED);
  CHECK(hasFlag(c, TIRED));
  CHECK(hasFlag(c, INJURED));
  CHECK(!hasFlag(c, DRUNK));
}

static void test_apply_same_flag_twice_is_idempotent()
{
  uint8_t c = NONE;
  c = applyFlag(c, TIRED);
  c = applyFlag(c, TIRED);
  CHECK_EQ_INT((int)c, (int)TIRED);
}

static void test_clear_flag_leaves_others_untouched()
{
  uint8_t c = NONE;
  c = applyFlag(c, TIRED);
  c = applyFlag(c, INJURED);
  c = applyFlag(c, BUSY);

  c = clearFlag(c, INJURED);
  CHECK(hasFlag(c, TIRED));
  CHECK(!hasFlag(c, INJURED));
  CHECK(hasFlag(c, BUSY));
}

static void test_clear_flag_not_set_is_noop()
{
  uint8_t c = applyFlag(NONE, TIRED);
  c = clearFlag(c, DRUNK); /* was never set */
  CHECK(hasFlag(c, TIRED));
  CHECK(!hasFlag(c, DRUNK));
}

static void test_dominant_flag_priority_order()
{
  /* INJURED > DRUNK > TIRED > BUSY > NONE */
  uint8_t all = applyFlag(applyFlag(applyFlag(applyFlag(NONE, TIRED), INJURED), DRUNK), BUSY);
  CHECK(dominantFlag(all) == INJURED);

  uint8_t drunkAndTired = applyFlag(applyFlag(NONE, TIRED), DRUNK);
  CHECK(dominantFlag(drunkAndTired) == DRUNK);

  uint8_t tiredAndBusy = applyFlag(applyFlag(NONE, TIRED), BUSY);
  CHECK(dominantFlag(tiredAndBusy) == TIRED);

  CHECK(dominantFlag(applyFlag(NONE, BUSY)) == BUSY);
}

static void test_dominant_flag_none_when_no_conditions()
{
  CHECK(dominantFlag(NONE) == NONE);
}

int main()
{
  test_apply_and_has_flag();
  test_flags_can_co_occur();
  test_apply_same_flag_twice_is_idempotent();
  test_clear_flag_leaves_others_untouched();
  test_clear_flag_not_set_is_noop();
  test_dominant_flag_priority_order();
  test_dominant_flag_none_when_no_conditions();
  return test_summary("test_npc_condition");
}
