/* Location atmosphere (docs/ideas-m7-living-npcs.md Part 4), game-state
 * half only -- see LocationAtmosphere.h's own header comment for scope. */
#include "LocationAtmosphere.h"
#include "test_util.h"

using namespace LocationAtmosphere;

static void test_fresh_state_is_neutral()
{
  AtmosphereState state{};
  CHECK(currentMood(state, 0, 10) == NEUTRAL);
}

static void test_set_mood_reads_back_before_decay()
{
  AtmosphereState state{};
  setMood(state, FESTIVE, 5);
  CHECK(currentMood(state, 5, 10) == FESTIVE);  /* same day */
  CHECK(currentMood(state, 14, 10) == FESTIVE); /* elapsed 9, not yet decayed */
}

static void test_mood_decays_to_neutral_at_boundary()
{
  AtmosphereState state{};
  setMood(state, TENSE, 5);
  CHECK(currentMood(state, 14, 10) == TENSE);   /* elapsed 9 */
  CHECK(currentMood(state, 15, 10) == NEUTRAL); /* elapsed 10, boundary decays */
  CHECK(currentMood(state, 100, 10) == NEUTRAL);
}

static void test_neutral_never_decays_further()
{
  AtmosphereState state{};
  setMood(state, NEUTRAL, 5);
  CHECK(currentMood(state, 5, 10) == NEUTRAL);
  CHECK(currentMood(state, 1000, 10) == NEUTRAL);
}

static void test_current_day_before_set_day_treated_as_zero_elapsed()
{
  AtmosphereState state{};
  setMood(state, ABANDONED, 50);
  CHECK(currentMood(state, 10, 10) == ABANDONED); /* elapsed clamped to 0, not wrapped */
}

static void test_setting_a_new_mood_resets_the_decay_clock()
{
  AtmosphereState state{};
  setMood(state, FESTIVE, 5);
  CHECK(currentMood(state, 14, 10) == FESTIVE); /* elapsed 9, about to decay */

  setMood(state, TENSE, 14); /* re-set before decay -- clock restarts */
  CHECK(currentMood(state, 15, 10) == TENSE);   /* only elapsed 1 now */
  CHECK(currentMood(state, 23, 10) == TENSE);   /* elapsed 9 from new setDay */
  CHECK(currentMood(state, 24, 10) == NEUTRAL); /* elapsed 10 from new setDay */
}

int main()
{
  test_fresh_state_is_neutral();
  test_set_mood_reads_back_before_decay();
  test_mood_decays_to_neutral_at_boundary();
  test_neutral_never_decays_further();
  test_current_day_before_set_day_treated_as_zero_elapsed();
  test_setting_a_new_mood_resets_the_decay_clock();
  return test_summary("test_location_atmosphere");
}
