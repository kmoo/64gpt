/* Visit cadence (docs/ideas-m7-living-npcs.md Part 4), game-state half
 * only -- see VisitFrequency.h's own header comment for scope. */
#include "VisitFrequency.h"
#include "test_util.h"

using namespace VisitFrequency;

static void test_never_visited_is_never_visited()
{
  VisitLog log{};
  CHECK(classifyCadence(log, 100) == NEVER_VISITED);
}

static void test_record_visit_increments_count()
{
  VisitLog log{};
  recordVisit(log, 10);
  CHECK_EQ_INT((int)log.visitCount, 1);
  CHECK_EQ_INT((int)log.lastVisitDay, 10);

  recordVisit(log, 11);
  CHECK_EQ_INT((int)log.visitCount, 2);
  CHECK_EQ_INT((int)log.lastVisitDay, 11);
}

static void test_record_visit_same_day_is_idempotent()
{
  VisitLog log{};
  recordVisit(log, 10);
  recordVisit(log, 10);
  recordVisit(log, 10);
  CHECK_EQ_INT((int)log.visitCount, 1); /* not tripled */
  CHECK_EQ_INT((int)log.lastVisitDay, 10);
}

static void test_daily_gap_classified_daily()
{
  VisitLog log{};
  recordVisit(log, 10);
  CHECK(classifyCadence(log, 10) == DAILY); /* gap 0, same day */
  CHECK(classifyCadence(log, 11) == DAILY); /* gap 1 */
}

static void test_weekly_gap_classified_weekly()
{
  VisitLog log{};
  recordVisit(log, 10);
  CHECK(classifyCadence(log, 12) == WEEKLY); /* gap 2 */
  CHECK(classifyCadence(log, 17) == WEEKLY); /* gap 7, boundary */
}

static void test_large_gap_classified_occasional()
{
  VisitLog log{};
  recordVisit(log, 10);
  CHECK(classifyCadence(log, 18) == OCCASIONAL); /* gap 8, just past weekly */
  CHECK(classifyCadence(log, 100) == OCCASIONAL); /* once a season */
}

static void test_current_day_before_last_visit_treated_as_zero_gap()
{
  /* caller error (day counters never go backwards) -- must not wrap on
   * unsigned subtraction and read as an enormous gap. */
  VisitLog log{};
  recordVisit(log, 100);
  CHECK(classifyCadence(log, 50) == DAILY); /* gap clamped to 0, not wrapped */
}

int main()
{
  test_never_visited_is_never_visited();
  test_record_visit_increments_count();
  test_record_visit_same_day_is_idempotent();
  test_daily_gap_classified_daily();
  test_weekly_gap_classified_weekly();
  test_large_gap_classified_occasional();
  test_current_day_before_last_visit_treated_as_zero_gap();
  return test_summary("test_visit_frequency");
}
