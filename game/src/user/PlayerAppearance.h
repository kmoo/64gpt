#pragma once

// Player visible appearance (docs/ideas-m7-living-npcs.md Part 4: "fine
// armor vs. rags changes how a guard or merchant opens a conversation,
// independent of actual reputation -- a well-dressed thief still gets
// the polite greeting"). GAME-STATE half only: pure bucketing from a
// caller-supplied equipment-quality score into a visible-gear tier, no
// knowledge of the actual inventory/equipment system. Deliberately
// independent of PlayerReputation.h -- appearance and reputation are
// separate axes that can disagree, same as this header's own framing.
namespace PlayerAppearance
{
  enum Tier { RAGS, PLAIN, FINE };

  constexpr int PLAIN_MIN_QUALITY = 30;
  constexpr int FINE_MIN_QUALITY = 70;

  inline Tier tierForQuality(int equipmentQuality)
  {
    if (equipmentQuality >= FINE_MIN_QUALITY) return FINE;
    if (equipmentQuality >= PLAIN_MIN_QUALITY) return PLAIN;
    return RAGS;
  }

  // Appearance-driven greeting warmth, independent of PlayerReputation
  // -- a well-dressed thief still reads as non-RAGS here regardless of
  // trueScore/publicScore.
  inline bool getsPoliteGreeting(Tier tier) { return tier != RAGS; }
}
