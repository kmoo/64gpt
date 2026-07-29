/* Quest-state memory (docs/ideas.md #8), pure game-state half only --
 * see QuestState.h's own header comment for what's deliberately NOT
 * built yet (prompt-token derivation, corpus regeneration). */
#include "QuestState.h"
#include "test_util.h"

using namespace QuestState;

static void test_mark_quest_done_and_query()
{
  QuestFlags flags{};
  CHECK(!isQuestDone(flags, 501));
  markQuestDone(flags, 501);
  CHECK(isQuestDone(flags, 501));
}

static void test_untracked_quest_is_not_done()
{
  QuestFlags flags{};
  markQuestDone(flags, 501);
  CHECK(!isQuestDone(flags, 999)); /* never started != completed */
}

static void test_mark_quest_done_is_idempotent()
{
  QuestFlags flags{};
  markQuestDone(flags, 501);
  markQuestDone(flags, 501);
  markQuestDone(flags, 501);

  int used = 0;
  for (int i = 0; i < MAX_QUESTS; ++i)
    if (flags.questId[i] == 501) ++used;
  CHECK_EQ_INT(used, 1); /* only one slot consumed, not three */
  CHECK(isQuestDone(flags, 501));
}

static void test_multiple_quests_fill_slots_independently()
{
  QuestFlags flags{};
  markQuestDone(flags, 101);
  markQuestDone(flags, 102);
  markQuestDone(flags, 103);

  CHECK(isQuestDone(flags, 101));
  CHECK(isQuestDone(flags, 102));
  CHECK(isQuestDone(flags, 103));
  CHECK(!isQuestDone(flags, 104));
}

static void test_full_table_new_quest_is_silent_noop()
{
  QuestFlags flags{};
  for (int i = 0; i < MAX_QUESTS; ++i)
    markQuestDone(flags, 1000 + (uint32_t)i);

  markQuestDone(flags, 9999); /* table is full -- must not crash or overwrite */

  CHECK(!isQuestDone(flags, 9999));
  for (int i = 0; i < MAX_QUESTS; ++i)
    CHECK(isQuestDone(flags, 1000 + (uint32_t)i)); /* nothing evicted */
}

static void test_marking_tracked_not_done_quest_transitions_to_done()
{
  /* markQuestDone always sets done=true, so the only way to observe a
   * tracked-but-not-done entry is to construct one directly -- confirms
   * the early-return match branch actually flips done, not just
   * re-confirms an already-true value (test_mark_quest_done_is_idempotent
   * only exercises the already-done case). */
  QuestFlags flags{};
  flags.questId[0] = 501;
  flags.done[0] = false;
  CHECK(!isQuestDone(flags, 501));

  markQuestDone(flags, 501);
  CHECK(isQuestDone(flags, 501));
}

static void test_full_table_existing_quest_still_updates()
{
  QuestFlags flags{};
  for (int i = 0; i < MAX_QUESTS; ++i)
    markQuestDone(flags, 1000 + (uint32_t)i);

  /* re-marking a quest already present in a full table must hit the
   * match branch, not the "table full, new quest" no-op branch. */
  markQuestDone(flags, 1005);
  CHECK(isQuestDone(flags, 1005));
}

static void test_zero_quest_id_collides_with_empty_sentinel()
{
  /* QuestState.h's own header comment: "0 is never a valid real quest
   * id". This documents what actually happens if a caller violates that
   * contract, since the struct is zero-initialized and EMPTY_QUEST_ID
   * is 0 -- there's no crash, but slot 0 silently becomes indistinguishable
   * from a real tracked quest 0 on a fresh table. */
  QuestFlags flags{};
  CHECK(!isQuestDone(flags, EMPTY_QUEST_ID)); /* fresh table: done[0] is false */

  markQuestDone(flags, EMPTY_QUEST_ID);
  CHECK(isQuestDone(flags, EMPTY_QUEST_ID)); /* now "done" -- collided with the sentinel slot */
}

int main()
{
  test_mark_quest_done_and_query();
  test_untracked_quest_is_not_done();
  test_mark_quest_done_is_idempotent();
  test_multiple_quests_fill_slots_independently();
  test_full_table_new_quest_is_silent_noop();
  test_marking_tracked_not_done_quest_transitions_to_done();
  test_full_table_existing_quest_still_updates();
  test_zero_quest_id_collides_with_empty_sentinel();
  return test_summary("test_quest_state");
}
