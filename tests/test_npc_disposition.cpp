/* NPC-to-NPC disposition scoring (docs/ideas.md #4), data-layer slice
 * only -- see NpcDisposition.h's own header comment for scope. */
#include "NpcDisposition.h"
#include "test_util.h"

using namespace NpcDisposition;

static void test_identical_traits_are_fully_compatible()
{
  int traits[5] = {50, 50, 50, 50, 50};
  CHECK_EQ_INT(compatibilityScore(traits, traits, 5, 100), 100);
}

static void test_maximally_opposed_traits_are_zero_compatible()
{
  int low[5] = {0, 0, 0, 0, 0};
  int high[5] = {100, 100, 100, 100, 100};
  CHECK_EQ_INT(compatibilityScore(low, high, 5, 100), 0);
}

static void test_partial_difference_scores_between_extremes()
{
  /* single-axis difference: distance 20 out of max possible 500 (5*100)
   * -> 20*100/500 = 4 -> score 96 */
  int a[5] = {50, 50, 50, 50, 50};
  int b[5] = {70, 50, 50, 50, 50};
  CHECK_EQ_INT(compatibilityScore(a, b, 5, 100), 96);
}

static void test_compatibility_is_symmetric()
{
  int a[5] = {10, 90, 30, 70, 50};
  int b[5] = {80, 20, 60, 40, 10};
  CHECK_EQ_INT(compatibilityScore(a, b, 5, 100), compatibilityScore(b, a, 5, 100));
}

static void test_zero_trait_count_is_zero()
{
  int a[1] = {50};
  CHECK_EQ_INT(compatibilityScore(a, a, 0, 100), 0);
}

static void test_is_friendly_pairing_threshold()
{
  int identical[5] = {50, 50, 50, 50, 50};
  CHECK(isFriendlyPairing(identical, identical, 5, 100)); /* score 100 */

  int low[5] = {0, 0, 0, 0, 0};
  int high[5] = {100, 100, 100, 100, 100};
  CHECK(!isFriendlyPairing(low, high, 5, 100)); /* score 0 */
}

int main()
{
  test_identical_traits_are_fully_compatible();
  test_maximally_opposed_traits_are_zero_compatible();
  test_partial_difference_scores_between_extremes();
  test_compatibility_is_symmetric();
  test_zero_trait_count_is_zero();
  test_is_friendly_pairing_threshold();
  return test_summary("test_npc_disposition");
}
