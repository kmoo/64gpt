# Briar Glen and the Everhollow — the long-term game vision (recorded 2026-07-22)

Not a roadmap for M11 — a record of the actual game this project is
building toward, shared by Luke as a full world map (`docs/img/
briar-glen-map.jpg`). M11 takes one small, real slice consistent with
it (see `docs/milestones/m11.md`); this doc exists so future milestones
(M11.1+) have the whole picture instead of rediscovering it piecemeal.
Same walking-skeleton discipline as everywhere else in this project:
record the vision, ship the smallest real piece, never build ahead of
what's actually tested and booting.

## The pitch, in the design's own words

> A Small Home. A Huge Adventure.

Cozy, family-friendly, explicitly **not** dark-fantasy-toned at the
overworld level ("Safe, welcoming world. Fun for kids and families.
Cozy home outside the village. Explore, befriend, discover. A small
hero in a big beautiful world."). The player is a small hero with a
dad and a dog, living in a small shack outside the village (garden,
workbench, a trophy wall added later).

## The adventure loop

```
1. Prepare in town  →  2. Explore the dungeon
        ↑                        ↓
4. Unlock new areas  ←  3. Return with treasure
   and quests
```

Town preparation → dungeon run → return with treasure → that unlocks
new overworld areas and quests → repeat, with the dungeon "changing
every time" and getting harder ("go deeper, get stronger").

## The town: Briar Glen

A central hub with a fountain, and named shops/services around it:
Pub, General Store, Herbalist, Carpenter, Chapel/Library.

## The dungeon: The Everhollow

Procedurally varies each run, gates world progression ("unlock the
world"). This is the piece the current `dungeon_crawler` manifest
already covers in miniature: Shadewrath (the necromancer captor) and
Korrath (his bound knight guard) are Everhollow boss-tier content.
Per Luke's own framing: "a lot of the town NPCs are outside of the
dungeon... some characters speak in the dungeon (bosses under
Shadewrath) but a lot of monsters won't speak... also a princess +
wizard + etc. (occasional good characters) you find in the dungeon."
So the dungeon isn't ONLY hostile content — it also holds rescuable/
friendly encounters, distinct from the town's own cast.

**Tone note, resolved 2026-07-22**: Shadewrath/Korrath's darker,
tragic material stays as-is under this vision — a real threat inside a
dungeon is normal even under a cozy overworld tone (confirmed by
Luke, not assumed).

## The overworld: 8 regions, boss-gated unlock order

| # | Region | Unlocks after | Flavor |
|---|---|---|---|
| 1 | West Hills | Boss 1 | Cozy hill homes, apple orchards, friendly folk, rare seeds |
| 2 | Greenheart Meadows | Boss 2 | Wildflowers, beekeepers, cooking ingredients, friendly quests |
| 3 | Whispering Woods | Boss 3 | Talking animals, fairies, hidden groves, forest charms |
| 4 | Stonehollow Pass | Boss 4 | (trail clears after Boss 4 — leads toward the player's home / the dungeon) |
| 5 | South Farmlands | Boss 5 | Crops & animals, cooking recipes, harvest festival, helpful merchants |
| 6 | Rivervale | Boss 6 | Elven sanctuary, healing springs, special crafting, beautiful music |
| 7 | Blue Coast | Boss 7 | Fishing village, lighthouse, boats & islands, sea treasures |
| 8 | Ancient Monastery | Boss 8 | Hidden library, lost knowledge, special spells, dungeon clues |
| — | "And beyond..." | — | Frost Peaks, Mystic Vale, Sky Ruins, Island Caves, Final Truth |

Roads/bridges/ferries physically open as each boss falls (e.g. "Road
opens after Boss 1," "Bridge opens after Boss 3," "Ferry runs after
Boss 7") — a real spatial overworld, gated by dungeon progress feeding
back into it. This is the biggest gap versus what exists today: see
"What this is NOT" below.

## How this maps onto what already exists (2026-07-22)

| Map concept | Current project equivalent |
|---|---|
| The Everhollow (dungeon) | `DungeonGenerator::npcsForLevel()` + the `R`/dungeon-mode toggle in `DialogueDemo.cpp` — NPC placement only, no room/tile generation (see "What this is NOT") |
| Briar Glen (town) | The fixed NPC-cycling roster in `DialogueDemo.cpp` (`START` cycles Selena → guards → cast → Shadewrath/Korrath → town archetypes) |
| Town shops (Pub, General Store, Herbalist, etc.) | Partial: Pub ≈ `innkeeper` occupation (Fergus). General Store / Herbalist are the `merchant` / `healer` archetypes M11 adds (see `docs/milestones/m11.md`) |
| Bosses under Shadewrath | `NPCDatabase::shadewrath`/`korrath`, `tier: full`/`mid` in `manifests/dungeon_crawler.json` |
| "Occasional good characters you find in the dungeon" (princess, wizard) | M11 adds one: a rescued elf princess (already referenced in Shadewrath's own bible — "abducted the elf princess of Ravendale") |
| 8-region unlock-gated overworld | **Does not exist.** No spatial world, no boss-gating, no roads/bridges opening |

## What this is NOT (yet) — the real scope gap

The current project (`core/` + `game/src/user/`) is a **dialogue-
inference engine and a flat NPC-cycling demo**, not a game engine.
There is no tile map, no room/level geometry, no player movement, no
collision, no overworld-unlock system anywhere in the codebase — this
was confirmed by direct code inspection during M11 planning (2026-07-
22), not assumed. Building the actual 8-region spatial overworld this
map describes is a different *kind* of project than "a tiny NPC
dialogue brain on N64" — likely the single largest engineering
undertaking this project could take on, bigger than the RSP matvec
work, bigger than the compositional-conditioning rewrite.

This isn't a new idea for this project, either — `docs/ideas.md`'s
very first entry ("A real scene instead of the cube... Pyrite64 is
built for exactly this") already flagged this exact direction as the
highest-leverage post-v1.0 idea. This doc is the concrete version of
that idea, now with an actual world design behind it.

## A rough future decomposition (not committed milestone numbers)

Sketched for planning purposes only — actual milestone numbers and
scope get decided when each one starts, same discipline this project
has followed since M7:

- **M11** (current): town cast content (merchant/healer archetypes),
  a rescued princess in the dungeon, dungeon-loop polish within the
  existing state-machine (no spatial movement), a quality push on
  Shadewrath/Korrath's coherence.
- **M13** (already scoped, renumbered 2026-07-23 from M12 -- M11.1's
  capacity-scaling follow-on claimed the M12 slot instead, see
  `docs/milestones/m12.md`): toolkit portability — meta-schema,
  manifest-update skill, cross-genre portability proof. Independent of
  the Briar Glen content direction.
- **M14+ (sketch only)**: the actual spatial-overworld engine work —
  likely its own multi-milestone arc: (a) a minimal tile/room system
  proven on ONE region first (e.g. West Hills, since it unlocks
  first), (b) player movement + collision, (c) the boss-gate → road/
  bridge/ferry-unlock mechanic, (d) the remaining 7 regions as content
  passes once the mechanism is proven — same "prove the mechanism on
  one slice, then scale it as content" pattern M8 (archetypes) → M10/
  M11 (full cast) already used successfully.
- **Player character (dad + dog + home)**: currently no player-
  character/inventory/home system exists at all either — likely its
  own slice, probably sequenced alongside or just before the overworld
  work above, since "prepare in town" (the adventure loop's step 1)
  depends on it.

None of this is required reading for M11's own DoD — recorded here so
it doesn't have to be rediscovered from a screenshot again.
