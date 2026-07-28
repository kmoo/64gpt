#pragma once

// NPC secrets (docs/ideas-m7-living-npcs.md "The feature I would add:
// NPC secrets": "PUBLIC BELIEF / PRIVATE BELIEF / SECRET / FEAR /
// DESIRE"). Public/Private are already covered by NPCState::Profile's
// resolveBeliefId(). This header adds the SECRET slot specifically:
// something an NPC actively hides, revealed only once a discovery
// condition is met. secretId/discoveryConditionId are opaque content
// handles, not invented fictional values -- same "resolving an id to
// actual content is game-content work, not this header's job"
// discipline as Profile's belief ids.
namespace NpcSecret
{
  struct Secret
  {
    int secretId;              // opaque content handle
    int discoveryConditionId;  // opaque handle for whatever check gates reveal
    bool revealed;             // sticky -- see revealSecret()
  };

  // Marks the secret revealed. One-way: there is no "un-reveal" path --
  // a secret, once discovered, stays discovered (you can't un-know
  // something), matching real secrets rather than a toggle. Idempotent
  // if already revealed.
  inline void revealSecret(Secret &secret)
  {
    secret.revealed = true;
  }

  // A secret is discoverable if conditionMet is true AND it isn't
  // already revealed -- no point re-triggering a reveal event for
  // something the player already knows. Pure predicate: the CALLER
  // decides what conditionMet means (comparing discoveryConditionId
  // against whatever game-state check applies -- this header has no
  // knowledge of what that check actually is).
  inline bool isDiscoverable(const Secret &secret, bool conditionMet)
  {
    return conditionMet && !secret.revealed;
  }
}
