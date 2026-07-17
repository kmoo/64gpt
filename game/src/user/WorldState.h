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
}
