#include "NpcService.h"
#include <stdio.h>
#include <string.h>

namespace NpcService
{
  // Matches trainer/ngpt_trainer/npc_service.py's OCCUPATIONS tuple
  // exactly, same order.
  const char *const OCCUPATIONS[OCCUPATION_COUNT] = {
    "villager", "guard", "merchant", "wizard", "damsel", "pub_patron",
    "blacksmith", "healer", "noble", "bandit", "farmer", "innkeeper",
  };

  // TRAITS index order, matching NPCDatabase::TRAITS exactly:
  // 0=warmth, 1=humor, 2=impulsivity, 3=bravery, 4=focus.
  enum TraitIdx { WARMTH = 0, HUMOR = 1, IMPULSIVITY = 2, BRAVERY = 3, FOCUS = 4 };

  uint32_t xorshift32(uint32_t x)
  {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
  }

  void ageGenderToken(int age, Gender gender, char *out, uint32_t outCap)
  {
    bool female = gender == Gender::Female;
    const char *tok;
    if(age <= 19)
      tok = female ? "girl" : "boy";
    else if(age <= 59)
      tok = female ? "woman" : "man";
    else
      tok = female ? "elderly_woman" : "elderly_man";
    snprintf(out, outCap, "%s", tok);
  }

  // Ordered blend rules -- first match wins, exact mirror of
  // npc_service.py's _BLENDS. Selena's real personality
  // {90,85,70,55,30} MUST hit rule 1 ("sassy") -- calibration check
  // lives in tests/test_npc_service.cpp.
  const char *personalityDescriptor(const int traits[NPCDatabase::TRAIT_COUNT])
  {
    int w = traits[WARMTH], h = traits[HUMOR], i = traits[IMPULSIVITY],
        b = traits[BRAVERY], f = traits[FOCUS];

    if(h >= 70 && i >= 60) return "sassy";
    if(w >= 70 && b < 45) return "gentle";
    if(b >= 70 && w < 45) return "gruff";
    if(f >= 70 && w < 45) return "stoic";
    if(b < 40 && f < 40) return "anxious";
    if(w >= 70 && h >= 60) return "cheerful";
    if(i < 35 && f >= 60) return "measured";
    if(b >= 75 && i >= 60) return "reckless";
    if(w < 35 && h < 35) return "cold";

    // No blend matched: single most-extreme-from-neutral (50) trait
    // wins, ties broken by TRAITS order (first strictly-greater match
    // kept) -- same as Python's max() over TRAITS in declared order.
    static const char *const highWord[NPCDatabase::TRAIT_COUNT] = {
      "warm", "playful", "impulsive", "bold", "focused",
    };
    static const char *const lowWord[NPCDatabase::TRAIT_COUNT] = {
      "cold", "serious", "careful", "timid", "distracted",
    };
    int dominant = 0;
    int bestDist = -1;
    for(int t = 0; t < NPCDatabase::TRAIT_COUNT; ++t)
    {
      int dist = traits[t] >= 50 ? traits[t] - 50 : 50 - traits[t];
      if(dist > bestDist)
      {
        bestDist = dist;
        dominant = t;
      }
    }
    return traits[dominant] >= 50 ? highWord[dominant] : lowWord[dominant];
  }

  uint16_t closeness(const RelationshipState &state)
  {
    uint32_t sum = (uint32_t)state.familiarity + state.affection
                 + state.trust + state.respect;
    return (uint16_t)(sum / 4);
  }

  const char *relationshipTier(uint16_t closenessScore)
  {
    static const struct { uint16_t threshold; const char *name; } TIERS[] = {
      {0, "stranger"}, {200, "acquaintance"}, {400, "neutral"},
      {600, "friend"}, {800, "close_friend"}, {950, "best_friend"},
    };
    const char *tier = TIERS[0].name;
    for(const auto &t : TIERS)
      if(closenessScore >= t.threshold)
        tier = t.name;
    return tier;
  }

  uint32_t buildPromptFields(char *out, uint32_t outCap, const Profile &profile,
                              const RelationshipState &relationship,
                              const char *mood, const char *context,
                              const char *event)
  {
    char person[16];
    ageGenderToken(profile.age, profile.gender, person, sizeof(person));
    const char *descriptor = personalityDescriptor(profile.traits);
    const char *tier = relationshipTier(closeness(relationship));
    const char *ev = (event && event[0]) ? event : "none";

    int n = snprintf(out, outCap,
                      "P:%s D:%s OCC:%s R:%s M:%s C:%s EV:%s|",
                      person, descriptor, profile.occupation,
                      tier, mood, context, ev);
    if(n < 0)
    {
      if(outCap) out[0] = '\0';
      return 0;
    }
    uint32_t len = (uint32_t)n;
    return len < outCap ? len : outCap - 1; // snprintf truncated; report actual written length
  }

  Profile profileFor(const NPCDatabase::NPC &npc)
  {
    Profile p;
    p.occupation = npc.occupation;
    p.age = npc.age;
    p.gender = npc.isFemale ? Gender::Female : Gender::Male;
    for(int i = 0; i < NPCDatabase::TRAIT_COUNT; ++i)
      p.traits[i] = npc.personality[i];
    return p;
  }

  bool isGossipHub(const char *occupation)
  {
    return occupation && (strcmp(occupation, "pub_patron") == 0 ||
                          strcmp(occupation, "villager") == 0);
  }

  const char *eventFor(const char *occupation, const char *directEvent,
                       const char *gossip)
  {
    if(isGossipHub(occupation) && gossip && gossip[0])
      return gossip;
    return (directEvent && directEvent[0]) ? directEvent : "";
  }
}
