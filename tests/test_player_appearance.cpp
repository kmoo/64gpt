/* Player visible appearance (docs/ideas-m7-living-npcs.md Part 4),
 * game-state half only -- see PlayerAppearance.h's own header comment
 * for scope. */
#include "PlayerAppearance.h"
#include "test_util.h"

using namespace PlayerAppearance;

static void test_low_quality_is_rags()
{
  CHECK(tierForQuality(0) == RAGS);
  CHECK(tierForQuality(29) == RAGS); /* just below PLAIN boundary */
}

static void test_mid_quality_is_plain()
{
  CHECK(tierForQuality(30) == PLAIN); /* boundary, inclusive */
  CHECK(tierForQuality(69) == PLAIN); /* just below FINE boundary */
}

static void test_high_quality_is_fine()
{
  CHECK(tierForQuality(70) == FINE); /* boundary, inclusive */
  CHECK(tierForQuality(1000) == FINE);
}

static void test_negative_quality_is_rags()
{
  CHECK(tierForQuality(-50) == RAGS);
}

static void test_polite_greeting_only_for_non_rags()
{
  CHECK(!getsPoliteGreeting(RAGS));
  CHECK(getsPoliteGreeting(PLAIN));
  CHECK(getsPoliteGreeting(FINE));
}

int main()
{
  test_low_quality_is_rags();
  test_mid_quality_is_plain();
  test_high_quality_is_fine();
  test_negative_quality_is_rags();
  test_polite_greeting_only_for_non_rags();
  return test_summary("test_player_appearance");
}
