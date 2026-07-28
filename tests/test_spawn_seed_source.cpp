/* Archetype spawn-seed-source strategy (docs/plan.md Known Follow-ups,
 * raised 2026-07-17 during M8). Header-only, no NPCDatabase.h dependency
 * -- seedPool/poolSize are passed in by the caller. */
#include "SpawnSeedSource.h"
#include "test_util.h"

using namespace SpawnSeedSource;

static const uint32_t GUARD_SEEDS[4] = {0x1001, 0x1002, 0x1003, 0x1004};

static void test_choose_seed_index_deterministic()
{
  uint32_t a = chooseSeedIndex(3, 1, 4);
  uint32_t b = chooseSeedIndex(3, 1, 4);
  CHECK_EQ_INT(a, b);
}

static void test_choose_seed_index_in_range()
{
  for (uint32_t level = 0; level < 20; ++level) {
    for (uint32_t slot = 0; slot < 8; ++slot) {
      uint32_t idx = chooseSeedIndex(level, slot, 4);
      CHECK(idx < 4);
    }
  }
}

static void test_choose_seed_index_zero_pool_is_zero()
{
  CHECK_EQ_INT(chooseSeedIndex(5, 2, 0), 0);
}

static void test_choose_seed_index_varies_across_levels()
{
  /* Not every level should resolve to the same index for slot 0 --
   * the whole point of this header is avoiding that degenerate case.
   * A handful of levels should show at least 2 distinct indices. */
  uint32_t seen[4] = {0, 0, 0, 0};
  int distinct = 0;
  for (uint32_t level = 0; level < 16; ++level) {
    uint32_t idx = chooseSeedIndex(level, 0, 4);
    if (!seen[idx]) { seen[idx] = 1; ++distinct; }
  }
  CHECK(distinct >= 2);
}

static void test_choose_spawn_seed_resolves_to_pool_value()
{
  uint32_t levelId = 7, slotIndex = 2;
  uint32_t idx = chooseSeedIndex(levelId, slotIndex, 4);
  uint32_t seed = chooseSpawnSeed(GUARD_SEEDS, 4, levelId, slotIndex);
  CHECK_EQ_INT(seed, GUARD_SEEDS[idx]);
}

static void test_choose_spawn_seed_zero_pool_is_zero()
{
  CHECK_EQ_INT(chooseSpawnSeed(GUARD_SEEDS, 0, 5, 2), 0u);
}

static void test_choose_spawn_seed_same_inputs_same_output()
{
  uint32_t a = chooseSpawnSeed(GUARD_SEEDS, 4, 9, 3);
  uint32_t b = chooseSpawnSeed(GUARD_SEEDS, 4, 9, 3);
  CHECK_EQ_INT(a, b);
}

int main()
{
  test_choose_seed_index_deterministic();
  test_choose_seed_index_in_range();
  test_choose_seed_index_zero_pool_is_zero();
  test_choose_seed_index_varies_across_levels();
  test_choose_spawn_seed_resolves_to_pool_value();
  test_choose_spawn_seed_zero_pool_is_zero();
  test_choose_spawn_seed_same_inputs_same_output();
  return test_summary("test_spawn_seed_source");
}
