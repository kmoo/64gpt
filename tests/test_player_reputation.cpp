/* Player reputation as global state (docs/ideas-m7-living-npcs.md Part
 * 4), see PlayerReputation.h's own header comment for scope. */
#include "PlayerReputation.h"
#include "test_util.h"

using namespace PlayerReputation;

static void test_apply_true_score_delta_clamps()
{
  Reputation rep{0, 90};
  applyTrueScoreDelta(rep, 30);
  CHECK_EQ_INT(rep.trueScore, 100); /* clamped, not 120 */
  CHECK_EQ_INT(rep.publicScore, 0); /* untouched */

  applyTrueScoreDelta(rep, -250);
  CHECK_EQ_INT(rep.trueScore, -100); /* clamped, not -150 */
}

static void test_true_and_public_scores_diverge()
{
  /* the whole point: secretly stealing (trueScore drops) must NOT move
   * publicScore at all, until a rumor actually spreads. */
  Reputation rep{50, 50};
  applyTrueScoreDelta(rep, -80); /* trueScore -> -30 */
  CHECK_EQ_INT(rep.trueScore, -30);
  CHECK_EQ_INT(rep.publicScore, 50); /* still unaware */
}

static void test_spread_rumor_partial_strength_closes_partial_gap()
{
  Reputation rep{0, 50}; /* gap = 50 */
  spreadReputationRumor(rep, 50); /* closes half the gap */
  CHECK_EQ_INT(rep.publicScore, 25);
  CHECK_EQ_INT(rep.trueScore, 50); /* rumors never move ground truth */
}

static void test_spread_rumor_full_strength_closes_gap_completely()
{
  Reputation rep{-40, 70};
  spreadReputationRumor(rep, 100);
  CHECK_EQ_INT(rep.publicScore, 70);
}

static void test_spread_rumor_zero_strength_is_noop()
{
  Reputation rep{10, 90};
  spreadReputationRumor(rep, 0);
  CHECK_EQ_INT(rep.publicScore, 10);
}

static void test_spread_rumor_clamps_strength_percent()
{
  Reputation over{0, 50};
  spreadReputationRumor(over, 500); /* clamped to 100 */
  CHECK_EQ_INT(over.publicScore, 50);

  Reputation under{0, 50};
  spreadReputationRumor(under, -50); /* clamped to 0 */
  CHECK_EQ_INT(under.publicScore, 0);
}

static void test_spread_rumor_works_when_public_ahead_of_true()
{
  /* gap can be negative too -- e.g. reputation was inflated by an
   * earlier false rumor, then a truthful one corrects it back down. */
  Reputation rep{80, 20}; /* gap = -60 */
  spreadReputationRumor(rep, 50);
  CHECK_EQ_INT(rep.publicScore, 50); /* 80 + (-60*50/100) = 80-30=50 */
}

static void test_repeated_partial_rumors_converge_toward_true_score()
{
  /* each call closes a FRACTION of the *remaining* gap, so repeated
   * partial-strength rumors should approach trueScore monotonically
   * without ever overshooting it. */
  Reputation rep{0, 100};
  spreadReputationRumor(rep, 50);
  CHECK_EQ_INT(rep.publicScore, 50); /* gap 100 -> +50 */
  spreadReputationRumor(rep, 50);
  CHECK_EQ_INT(rep.publicScore, 75); /* gap 50 -> +25 */
  spreadReputationRumor(rep, 50);
  CHECK_EQ_INT(rep.publicScore, 87); /* gap 25 -> +12 (integer truncation) */
  CHECK(rep.publicScore < rep.trueScore); /* still converging, not there yet */
}

static void test_tiny_gap_can_get_permanently_stuck_below_full_strength()
{
  /* integer division truncates: a gap of 1 at any strength < 100 always
   * computes delta = (1 * strength) / 100 = 0, so publicScore NEVER
   * moves no matter how many partial-strength rumors spread -- only a
   * strength of exactly 100 can close a gap this small. Real behavior
   * worth documenting, not obviously a bug given the "fraction of the
   * gap" spec, but a caller relying on eventual convergence at low
   * strength would be surprised. */
  Reputation rep{99, 100};
  spreadReputationRumor(rep, 50);
  CHECK_EQ_INT(rep.publicScore, 99); /* unmoved */
  spreadReputationRumor(rep, 99);
  CHECK_EQ_INT(rep.publicScore, 99); /* still unmoved, even at 99% strength */
  spreadReputationRumor(rep, 100);
  CHECK_EQ_INT(rep.publicScore, 100); /* only full strength closes it */
}

int main()
{
  test_apply_true_score_delta_clamps();
  test_true_and_public_scores_diverge();
  test_spread_rumor_partial_strength_closes_partial_gap();
  test_spread_rumor_full_strength_closes_gap_completely();
  test_spread_rumor_zero_strength_is_noop();
  test_spread_rumor_clamps_strength_percent();
  test_spread_rumor_works_when_public_ahead_of_true();
  test_repeated_partial_rumors_converge_toward_true_score();
  test_tiny_gap_can_get_permanently_stuck_below_full_strength();
  return test_summary("test_player_reputation");
}
