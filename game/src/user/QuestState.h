#pragma once
#include <stdint.h>

// Quest-state memory (docs/ideas.md #8 "Quest-state memory": "extend
// the prompt protocol with game state (EV=QUEST_DONE, MET=TRUE) and
// regenerate the corpus -- NPCs that visibly react to what the player
// did"). This header is the GAME-STATE half only: pure, host-tested,
// no training/corpus/model change. Deriving EV:/MET:-style conditioning
// tokens from this state and retraining on them is real, separate
// corpus-regeneration work the idea doc itself calls "medium effort" --
// explicitly out of scope here, same "build the pure state layer first"
// discipline as every other NPCState-adjacent header tonight. Fixed
// size, no heap, matching this codebase's other state structures.
namespace QuestState
{
  constexpr int MAX_QUESTS = 16;

  // 0 is never a valid real quest id (matches every other "0 = empty"
  // convention in this codebase's state headers, e.g. NPCState::
  // EMPTY_EVENT_ID).
  constexpr uint32_t EMPTY_QUEST_ID = 0;

  struct QuestFlags
  {
    uint32_t questId[MAX_QUESTS]; // EMPTY_QUEST_ID = unused slot
    bool done[MAX_QUESTS];        // parallel array, same index as questId
  };

  // Marks questId as done. Idempotent if already tracked (just sets
  // done=true again). If not yet tracked and a free slot exists, adds
  // it as done. If the table is full and questId isn't already present,
  // this is a silent no-op -- fixed MAX_QUESTS capacity, deliberately
  // no eviction policy (unlike MemoryBlock's episodic memories, quest
  // completion should never need to be forgotten to make room).
  inline void markQuestDone(QuestFlags &flags, uint32_t questId)
  {
    int emptySlot = -1;
    for (int i = 0; i < MAX_QUESTS; ++i) {
      if (flags.questId[i] == questId) {
        flags.done[i] = true;
        return;
      }
      if (emptySlot < 0 && flags.questId[i] == EMPTY_QUEST_ID) emptySlot = i;
    }
    if (emptySlot >= 0) {
      flags.questId[emptySlot] = questId;
      flags.done[emptySlot] = true;
    }
  }

  // True only if questId is tracked AND marked done. An untracked quest
  // is NOT done -- "never started" and "completed" are different
  // things, both represented as needed rather than conflated.
  inline bool isQuestDone(const QuestFlags &flags, uint32_t questId)
  {
    for (int i = 0; i < MAX_QUESTS; ++i) {
      if (flags.questId[i] == questId) return flags.done[i];
    }
    return false;
  }
}
