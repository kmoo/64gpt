/* Time-of-day state (docs/ideas-m7-living-npcs.md Part 3/4), game-state
 * half only -- see TimeOfDay.h's own header comment for scope. */
#include "TimeOfDay.h"
#include "test_util.h"

using namespace TimeOfDay;

static void test_period_quarters_of_a_day()
{
  /* ticksPerDay=100 -> quarter=25: [0,25)=DAWN, [25,50)=DAY,
   * [50,75)=DUSK, [75,100)=NIGHT */
  CHECK(periodForTick(0, 100) == DAWN);
  CHECK(periodForTick(24, 100) == DAWN);
  CHECK(periodForTick(25, 100) == DAY);
  CHECK(periodForTick(49, 100) == DAY);
  CHECK(periodForTick(50, 100) == DUSK);
  CHECK(periodForTick(74, 100) == DUSK);
  CHECK(periodForTick(75, 100) == NIGHT);
  CHECK(periodForTick(99, 100) == NIGHT);
}

static void test_wraps_past_one_day()
{
  CHECK(periodForTick(100, 100) == DAWN);   /* wraps to tick 0 of day 2 */
  CHECK(periodForTick(175, 100) == NIGHT);  /* wraps to tick 75 */
  CHECK(periodForTick(350, 100) == DUSK);   /* 3 full days + tick 50 */
}

static void test_zero_ticks_per_day_returns_day_not_crash()
{
  CHECK(periodForTick(0, 0) == DAY);
  CHECK(periodForTick(12345, 0) == DAY); /* no divide-by-zero regardless of tick */
}

static void test_ticks_per_day_too_small_for_quarters_returns_day()
{
  /* ticksPerDay < 4 means quarter truncates to 0 -- same divide-by-zero
   * guard as the zero case, just reached via a different input. */
  CHECK(periodForTick(0, 1) == DAY);
  CHECK(periodForTick(2, 3) == DAY);
}

static void test_default_ticks_per_day_constant_divides_cleanly()
{
  /* DEFAULT_TICKS_PER_DAY is meant as a real usable default, not just a
   * placeholder number -- confirm it actually produces all four periods
   * rather than silently falling into the too-small guard above. */
  CHECK(periodForTick(0, DEFAULT_TICKS_PER_DAY) == DAWN);
  CHECK(periodForTick(DEFAULT_TICKS_PER_DAY / 2, DEFAULT_TICKS_PER_DAY) == DUSK);
}

static void test_is_night_only_true_for_night_period()
{
  CHECK(!isNight(DAWN));
  CHECK(!isNight(DAY));
  CHECK(!isNight(DUSK));
  CHECK(isNight(NIGHT));
}

int main()
{
  test_period_quarters_of_a_day();
  test_wraps_past_one_day();
  test_zero_ticks_per_day_returns_day_not_crash();
  test_ticks_per_day_too_small_for_quarters_returns_day();
  test_default_ticks_per_day_constant_divides_cleanly();
  test_is_night_only_true_for_night_period();
  return test_summary("test_time_of_day");
}
