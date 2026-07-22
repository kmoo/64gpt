#pragma once

// Shared World State: the small set of global facts other subsystems
// read or write via the Event Bus contract (docs/milestones/m7.md's
// architecture diagram pairs this with EventBus — "any subsystem
// reads/writes state"). M7 scope: just what Selena's Context Builder
// needs to pick a context category (the schema's C: field) — expand
// only when a second real reader (Dungeon/Music, M8+) needs more.
//
// Portable C++, no libdragon includes — builds in the host test suite.
namespace WorldState
{
  // One of NPCDatabase::CONTEXTS; defaults to the first entry.
  const char *currentContext();
  void setContext(const char *context);

  // M11: gossip -- the world-state half of "event -> world state ->
  // nearby NPCs' conditioning references it" (docs/milestones/m11.md
  // section 2). A player-caused event worth the town hearing about, set
  // once from a real trigger (DialogueDemo.cpp) and read by NpcService::
  // eventFor() for occupations trained to react to it secondhand
  // (NpcService::isGossipHub()). Deliberately just one slot, not a
  // history or a decay timer -- "keep it simple for v1" per the plan;
  // add propagation/decay only when a second real need shows up, same
  // premature-abstraction discipline EventBus's own header note uses.
  constexpr int GOSSIP_EVENT_COUNT = 3;
  extern const char *const GOSSIP_EVENTS[GOSSIP_EVENT_COUNT]; // must match
                                    // trainer/ngpt_trainer/cast_corpus.py's
                                    // GOSSIP_EVENTS exactly -- these are the
                                    // only tags the model was ever trained
                                    // to react to as gossip

  // "" if nothing gossip-worthy has happened yet.
  const char *currentGossip();
  void setGossip(const char *tag);
}
