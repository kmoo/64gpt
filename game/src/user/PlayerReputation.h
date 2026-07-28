#pragma once

// Player reputation as global state (docs/ideas-m7-living-npcs.md Part
// 4, 2026-07-17 during M9): "a single shared value ('Hero,' 'Thief,'
// 'Stranger')... every NPC's conditioning can read, independent of
// whether that specific NPC has ever met the player." Unlike
// NPCState::Relationship (one vector PER NPC, O(N)), this is a SINGLE
// shared value -- O(1). Also covers that same section's brainstormed
// richer variant: "Reputation vs. actual behavior divergence... tracked
// as two separate values that sometimes contradict" -- secretly
// stealing but never caught should move trueScore without moving
// publicScore at all.
namespace PlayerReputation
{
  // [-100, 100], mirroring NPCState::Relationship::affection's range --
  // positive = heroic, negative = villainous, 0 = unknown/stranger (the
  // default an NPC who's never heard of the player should read as).
  struct Reputation
  {
    int publicScore; // what NPCs have HEARD (gossip/rumor -- can be wrong)
    int trueScore;    // what the player has ACTUALLY done (ground truth)
  };

  constexpr int REPUTATION_MIN = -100;
  constexpr int REPUTATION_MAX = 100;

  inline int clampReputation(int value)
  {
    if (value < REPUTATION_MIN) return REPUTATION_MIN;
    if (value > REPUTATION_MAX) return REPUTATION_MAX;
    return value;
  }

  // Applies a delta to trueScore ONLY. publicScore is a separate,
  // slower-moving value that only shifts via spreadReputationRumor()
  // below (gossip-driven, not automatic on every action) -- this is
  // the mechanism that lets trueScore and publicScore diverge.
  inline void applyTrueScoreDelta(Reputation &rep, int delta)
  {
    rep.trueScore = clampReputation(rep.trueScore + delta);
  }

  // Public reputation drifts toward the true score when a rumor
  // spreads -- moves by a FRACTION of the remaining gap
  // (rumorStrengthPercent, clamped 0-100), not the whole gap in one
  // hop, since one rumor shouldn't make public perception exactly
  // match reality instantly. 100 closes the gap completely in one call.
  inline void spreadReputationRumor(Reputation &rep, int rumorStrengthPercent)
  {
    if (rumorStrengthPercent < 0) rumorStrengthPercent = 0;
    if (rumorStrengthPercent > 100) rumorStrengthPercent = 100;
    int gap = rep.trueScore - rep.publicScore;
    rep.publicScore = clampReputation(rep.publicScore
                                       + (gap * rumorStrengthPercent) / 100);
  }
}
