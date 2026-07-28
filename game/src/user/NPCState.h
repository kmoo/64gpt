#pragma once
#include <stdint.h>
#include <string.h>

// "Living NPC state" step 1 (docs/ideas-living-npc-state.md section 6):
// the pure, host-testable data model for a per-NPC relationship vector
// and an 8-slot episodic memory block, plus the deterministic logic to
// update them. Same split as SaveData.h's pure half (ADR 0001,
// docs/adr/0001-host-test-portable-cpp-separate-from-libdragon.md):
// header-only, no libdragon include, no heap, no floats -- fully
// host-tested in tests/test_npc_state.cpp. Serialization/EEPROM-SD I/O
// and EventBus wiring are separate, later, hardware-only work -- this
// header only defines the shape and the pure update rules.
namespace NPCState
{
  constexpr int MEMORY_SLOTS = 8;

  // 0 is never a valid real event id (matches NPCDatabase's own
  // memorySlot "0 = empty/unused" convention) -- Memory::eventId == 0
  // means the slot is unoccupied.
  constexpr uint32_t EMPTY_EVENT_ID = 0;

  // ticksElapsed / DECAY_TICKS_PER_SALIENCE_POINT (integer division) is
  // how much salience ageMemories() removes -- 10 ticks costs exactly
  // 1 salience point.
  constexpr uint32_t DECAY_TICKS_PER_SALIENCE_POINT = 10;

  struct Relationship
  {
    int familiarity; // [0, 100]
    int affection;   // [-100, 100] -- can go negative, unlike the others
    int trust;       // [0, 100]
    int respect;     // [0, 100]
    int fear;        // [0, 100]
  };

  struct Memory
  {
    uint32_t eventId; // EMPTY_EVENT_ID (0) = unused slot
    int salience;      // [0, 100]
    int confidence;     // [0, 100]
    uint32_t ageTicks;
  };

  struct MemoryBlock
  {
    Memory slots[MEMORY_SLOTS]; // zero-initialized => all slots empty
  };

  // Adds each delta then clamps (add-then-clamp, not clamp-then-add).
  inline void applyDelta(Relationship &rel, int dFamiliarity, int dAffection,
                          int dTrust, int dRespect, int dFear)
  {
    rel.familiarity += dFamiliarity;
    rel.affection += dAffection;
    rel.trust += dTrust;
    rel.respect += dRespect;
    rel.fear += dFear;

    if (rel.familiarity < 0) rel.familiarity = 0;
    if (rel.familiarity > 100) rel.familiarity = 100;
    if (rel.affection < -100) rel.affection = -100;
    if (rel.affection > 100) rel.affection = 100;
    if (rel.trust < 0) rel.trust = 0;
    if (rel.trust > 100) rel.trust = 100;
    if (rel.respect < 0) rel.respect = 0;
    if (rel.respect > 100) rel.respect = 100;
    if (rel.fear < 0) rel.fear = 0;
    if (rel.fear > 100) rel.fear = 100;
  }

  // Writes into the first empty slot (lowest index) if one exists;
  // otherwise evicts the lowest-salience slot, ties broken by highest
  // ageTicks (oldest), remaining ties broken by lowest index.
  inline void recordMemory(MemoryBlock &block, uint32_t eventId, int salience,
                            int confidence)
  {
    if (salience < 0) salience = 0;
    if (salience > 100) salience = 100;
    if (confidence < 0) confidence = 0;
    if (confidence > 100) confidence = 100;

    int target = -1;
    for (int i = 0; i < MEMORY_SLOTS; ++i) {
      if (block.slots[i].eventId == EMPTY_EVENT_ID) {
        target = i;
        break;
      }
    }
    if (target < 0) {
      target = 0;
      for (int i = 1; i < MEMORY_SLOTS; ++i) {
        const Memory &cur = block.slots[target];
        const Memory &cand = block.slots[i];
        bool candLower = cand.salience < cur.salience;
        bool tieOlder = cand.salience == cur.salience &&
                         cand.ageTicks > cur.ageTicks;
        if (candLower || tieOlder) target = i;
      }
    }

    block.slots[target].eventId = eventId;
    block.slots[target].salience = salience;
    block.slots[target].confidence = confidence;
    block.slots[target].ageTicks = 0;
  }

  // Advances ageTicks and decays salience for every occupied slot;
  // empty slots are left completely untouched.
  inline void ageMemories(MemoryBlock &block, uint32_t ticksElapsed)
  {
    for (int i = 0; i < MEMORY_SLOTS; ++i) {
      Memory &m = block.slots[i];
      if (m.eventId == EMPTY_EVENT_ID) continue;
      m.ageTicks += ticksElapsed;
      int decay = (int)(ticksElapsed / DECAY_TICKS_PER_SALIENCE_POINT);
      m.salience -= decay;
      if (m.salience < 0) m.salience = 0;
    }
  }

  // Returns true if slot `a` must sort strictly before slot `b`:
  // descending salience, ties broken by ascending ageTicks. Equal on
  // both -> false for both orderings, so a STABLE sort preserves the
  // remaining tie-break (ascending array index) for free.
  inline bool memoryPrecedes(const Memory &a, const Memory &b)
  {
    if (a.salience != b.salience) return a.salience > b.salience;
    return a.ageTicks < b.ageTicks;
  }

  // Sorts occupied slots best-first (memoryPrecedes order) and writes up
  // to `n` winning eventIds into outEventIds. Does not modify `block`.
  inline int selectTopMemories(const MemoryBlock &block, int n,
                                uint32_t outEventIds[])
  {
    int idx[MEMORY_SLOTS];
    int count = 0;
    for (int i = 0; i < MEMORY_SLOTS; ++i) {
      if (block.slots[i].eventId != EMPTY_EVENT_ID) idx[count++] = i;
    }

    // Stable insertion sort: only shift when the incoming slot strictly
    // precedes the one already placed, so slots tied on every criterion
    // keep their original (ascending index) relative order.
    for (int i = 1; i < count; ++i) {
      int key = idx[i];
      int j = i - 1;
      while (j >= 0 && memoryPrecedes(block.slots[key], block.slots[idx[j]])) {
        idx[j + 1] = idx[j];
        --j;
      }
      idx[j + 1] = key;
    }

    int written = n < count ? n : count;
    if (written < 0) written = 0;
    for (int i = 0; i < written; ++i) outEventIds[i] = block.slots[idx[i]].eventId;
    return written;
  }

  // "Profile" (docs/ideas-living-npc-state.md section 1a): personality
  // axes/occupation/species/bond already live in NPCDatabase::NPC --
  // this only adds what NPCDatabase does NOT have. publicBeliefId/
  // privateBeliefId are deliberately opaque ids, not enums with invented
  // fictional values -- resolving an id to actual belief CONTENT is
  // game-content work for later, not this header's job. defaultGoalId
  // is the same kind of opaque handle for the "default goal/role" the
  // ideas doc lists.
  struct Profile
  {
    int publicBeliefId;
    int privateBeliefId;
    int defaultGoalId;
  };

  // Trust threshold at which an NPC's PRIVATE belief becomes the one
  // Context Builder should surface instead of the public one (section 2:
  // "Read the current Profile + Relationship... pack a compact priming
  // string") -- this is the one piece of real decision logic Profile
  // needs, matching Relationship's own [0,100] trust scale.
  constexpr int PRIVATE_BELIEF_TRUST_THRESHOLD = 60;

  // Returns publicBeliefId below the reveal threshold, privateBeliefId
  // at or above it. Pure, no side effects -- Context Builder calls this
  // fresh every utterance rather than caching a "has revealed" flag,
  // so a relationship that later DROPS back below the threshold (e.g.
  // a betrayal) correctly reverts to the public face without needing
  // separate one-way-reveal bookkeeping.
  inline int resolveBeliefId(const Profile &profile, const Relationship &rel)
  {
    return rel.trust >= PRIVATE_BELIEF_TRUST_THRESHOLD
             ? profile.privateBeliefId
             : profile.publicBeliefId;
  }

  // EventBus -> reaction table -> relationship update (docs/ideas-living-
  // npc-state.md section 6 step 2): "the update rules that turn world
  // events into relationship deltas belong in a small, data-driven
  // reaction table, not inside the GRU." No heap/hash-map available, so
  // this is a small fixed array + linear scan -- fine at N64 scale for a
  // handful of world events, matching the fixed-size discipline every
  // other NPCState structure here already follows.
  struct ReactionRule
  {
    const char *eventTag;
    int dFamiliarity, dAffection, dTrust, dRespect, dFear;
  };

  // Tags match WorldState::GOSSIP_EVENTS exactly (game/src/user/
  // WorldState.cpp) -- the only concrete world events this project
  // publishes today (DialogueDemo.cpp's EventBus::publish() call sites).
  // Delta VALUES are a first tuning pass, not narrative content (same
  // "opaque id, not invented fiction" discipline as Profile above) --
  // expected to be revised as real gameplay balance work happens; the
  // reusable piece is the table-driven MECHANISM, not these numbers.
  constexpr ReactionRule REACTION_TABLE[] = {
    {"shadewrath_allied", 10, 20, 15, 10, -10},
    {"korrath_pleaded",    5, 10,  5, 15,   0},
    {"princess_freed",    15, 25, 20, 20, -15},
  };
  constexpr int REACTION_TABLE_SIZE =
    sizeof(REACTION_TABLE) / sizeof(REACTION_TABLE[0]);

  // Looks up eventTag in REACTION_TABLE and applies its deltas to rel via
  // applyDelta() if found. Returns true if a matching rule existed and
  // was applied; false (a no-op, not an error) if eventTag has no
  // reaction rule -- most published events won't affect every NPC.
  inline bool applyEventReaction(Relationship &rel, const char *eventTag)
  {
    for (int i = 0; i < REACTION_TABLE_SIZE; ++i) {
      if (strcmp(REACTION_TABLE[i].eventTag, eventTag) == 0) {
        const ReactionRule &r = REACTION_TABLE[i];
        applyDelta(rel, r.dFamiliarity, r.dAffection, r.dTrust, r.dRespect, r.dFear);
        return true;
      }
    }
    return false;
  }
}
