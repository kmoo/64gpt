# Ideas — living NPC state (post-M12.1 synthesis, 2026-07-26)

**Not a roadmap.** Speculative architecture note that merges four existing
threads into one coherent picture:

- `docs/ideas-m7-living-npcs.md` (continuity of existence, external
  memory/personality, gossip, Old Man Rowan test, Context Builder as
  director)
- `docs/ideas-m7-living-npcs.md` Part 3 (relationship-state formalization
  during M8)
- `docs/ideas-model-swap-architecture.md` (three-tier weight-file
  swapping, post-M10)
- The persistent-memory + hierarchical-act direction discussed after
  M12.1 closed the coherence gap

Nothing here is scheduled. The single-shared-model + compositional
conditioning approach remains the default; this document only records
how the surrounding *state* systems would have to grow if we ever want
NPCs that feel continuously alive rather than merely fluent.

## Critical constraint: this is a slice of the budget, not the whole game

The N64 target hasn't changed: `docs/ideas-briar-glen-world.md`'s actual
goal is an 8-region spatial overworld, a procedural dungeon, player
movement/collision, a home — an "amazing 3D world and game," not a
dialogue demo with NPCs standing in an empty room. Every layer proposed
below (Profile, Relationship, Memory/gossip, the Context Builder) has to
compete for the same fixed hardware this project has always had to
respect: RDRAM, the RSP's 4 KB DMEM tile, SD/EEPROM I/O bandwidth, and
CPU cycles per frame — all of which the actual game (rendering,
geometry, physics, world simulation) needs the lion's share of once it
exists.

Concretely, that means:

- **RAM budget, not just DMEM.** The model weights already claim a
  measured, fixed RDRAM footprint (`rspWhh` at H=320 is ~300KB; M12
  showed H=1024 costs ~3MB and 9x the inference latency for a
  coherence result that was *worse*, not better — capacity is not free
  and does not obviously buy quality). Relationship vectors, memory
  blocks, and gossip pools for dozens of NPCs are small individually but
  add up across a full town + dungeon cast, and that budget line has to
  be sized against what's left *after* world geometry, textures, and
  collision data claim their share — not sized first and geometry fit
  around the leftovers.
- **Per-frame cost, not just per-conversation cost.** Dialogue
  generation already runs in its own bounded step budget
  (`ngpt_step`/RSP matvec) that the game loop can afford *because it
  only happens during a conversation*, not every frame. Any new
  always-on system this doc proposes (gossip propagation ticking in the
  background, relationship decay, memory salience re-scoring) must stay
  event-driven and off the per-frame hot path, not become a second
  system quietly competing with rendering/physics for CPU time every
  frame the way the neural core deliberately does not.
- **SD/EEPROM I/O is shared, not NPC-state's alone.** Save data, level
  geometry streaming, and (eventually) Tier-2/3 model-file swapping all
  want the same I/O channel this doc's `.mem` blocks would use — sizing
  and access patterns need to be measured against that shared load, not
  assumed to have the channel to themselves.

**The practical implication:** the "suggested first spike" (§6) has to
report its real RAM/DMEM/cycle cost on top of an otherwise-idle engine,
same as every hardware claim elsewhere in this project (`docs/plan.md`'s
own discipline — measured on Ares/EverDrive, not assumed). If the state
layer can't stay a small, bounded slice once real world/rendering work
exists alongside it, that's a finding this doc needs before Tier-2/3
model swapping or richer relationship axes are even considered — an
alive-feeling NPC in a game with no world left to explore isn't the
goal.

## The central claim (unchanged from M7)

> Continuity of existence beats model size.
>
> A ~300–700 k parameter model that remembers, lies, changes, and
> participates in a world will feel more alive than a multi-million
> parameter model that only produces pretty sentences.

M12.1 proved the mouth can be made reliable (zero invented words on the
shipped goldens, 44 ch/s, QAT + min-p + trie). The remaining work is
almost entirely outside the GRU.

## Four layers, one resident model

```
┌─────────────────────────────────────────────────────────────┐
│                     Game / World Systems                    │
│          (EventBus, DungeonGenerator, quest flags, …)       │
└──────────────────────────┬──────────────────────────────────┘
                           │ events
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  NPC State Layer (new)                      │
│  Profile (static) + Relationship (per-player) + Memory      │
│  + Gossip / rumor corruption + Secrets                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ selected slice
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Context Builder (the director)                │
│  Packs the tiny conditioning string the GRU is allowed to   │
│  see *this turn*. Also chooses (or receives) the dialogue   │
│  act from a lightweight planner.                            │
└──────────────────────────┬──────────────────────────────────┘
                           │ text priming string
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          Language Model (M12.1 path, H=320 default)         │
│  Single resident weights (Tier 1). Optional later swap of   │
│  Tier-2/3 weight files from SD when a voice is proven to    │
│  need specialization the shared model cannot carry.         │
│  min-p + trie guard remain on.                              │
└─────────────────────────────────────────────────────────────┘
```

Only the bottom box is neural. Everything above it is ordinary game
state and a careful director.

## 1. NPC State Layer (the new work)

Three sub-pieces, deliberately kept O(N) in the number of important
NPCs (no NPC-to-NPC social graph).

### 1a. Profile (rarely changes)

Already partially exists as `archetype.personality_ranges` + occupation /
species / faction tags. Extend only as needed:

- personality axes (lawful↔chaotic, kind↔cruel, …)
- default goals / role
- public vs private belief seeds
- one or two permanent secrets / fears / desires (the "soul" hooks)

Stored in the manifest / NPCDatabase; almost never written at runtime.

### 1b. Relationship State (per-player, changes over time)

Generalization of the existing `trustTier` (currently one axis, three
buckets). Target shape from the M8 formalization:

| Axis          | Range / buckets          | Notes                          |
|---------------|--------------------------|---------------------------------|
| familiarity   | 0–1 (stranger→lifelong)  | slowly rises with interactions |
| affection     | –1 … +1                  |                                 |
| trust         | 0–1                      | already partially present      |
| respect       | 0–1                      |                                 |
| fear          | 0–1                      |                                 |
| rel-type      | enum (friend/rival/…)    | optional label                  |

**Critical constraint (still true):** conditioning is a text string
through the frozen `ngpt_reset` API. Continuous floats cannot go in
directly. Two practical paths:

- **Bucket aggressively** (3–5 levels per axis) and accept the
  combinatorial cost, or collapse several axes into one derived
  "disposition" token before they reach the schema string.
- **Embedding-table fallback** (the contingency already flagged in
  M7/M8) if the bucketed grid becomes intolerable.

The update rules that turn world events into relationship deltas
(`EVENT_PLAYER_STOLE_ITEM → affection –0.3, trust –0.4`) are the real
design work. They belong in a small, data-driven reaction table, not
inside the GRU.

### 1c. Episodic Memory + Gossip

Not a fact database. Each memory carries:

- event id / short payload
- importance / salience
- confidence
- age / timestamp
- emotional residue (anger, fear, respect deltas)
- optional corruption / rumor variant

At conversation start the Context Builder retrieves the 1–3 most
relevant memories (simple score: salience × recency × relevance to
current goal or player utterance). After the conversation, new
memories may be written, old ones decayed or merged, and a gossip
variant may be seeded into nearby NPCs' memory pools.

Persistence: tiny per-character `.mem` blocks on the SD card (or in the
EEPROM save for the most important NPCs). Only the active working set
lives in RDRAM. This is exactly the "survives save/load" requirement
of the Old Man Rowan test.

## 2. Context Builder as director

Still the most important component. Its job on every utterance:

1. Read the current Profile + Relationship + retrieved Memories.
2. (Optional) run a cheap planner / utility system that chooses a
   **dialogue act** (GreetWarmly, Deflect, RevealSecret, Lie, …).
3. Pack a compact priming string the existing schema can accept.
4. Hand the string to `ngpt_reset` + the M12.1 generation path.

The GRU never has to decide *what* the character wants. It only
realizes the act under the state the director made visible.

This is hierarchical generation without any change to the neural core.

## 3. Model weight files (Tier 1 / 2 / 3)

Unchanged from `ideas-model-swap-architecture.md`:

- **Tier 1** — single shared model (default, zero swap cost). Covers
  unlimited archetype instances via seeded personality jitter.
- **Tier 2** — one model per base archetype, swapped from SD only while
  that type is speaking.
- **Tier 3** — one model per named individual, only for the rare
  characters whose voice cannot be carried by conditioning + memory.

SD capacity is not the constraint; RAM and the 4 KB RSP DMEM tile are.
Swap latency must be measured on EverDrive before Tier 2/3 become real
design. Until the cast is proven to outgrow rich state + conditioning,
Tier 1 remains correct.

## 4. The acceptance test that actually matters

**Old Man Rowan** (from the M7 note), upgraded with the new state:

- One cabin, one NPC.
- Remembers the last ~10 interactions across save/load.
- Personality / relationship visibly shifts.
- Holds at least one secret the player can discover or be lied to about.
- Gossip about the player reaches a second NPC.
- Generates novel but consistent lines under the M12.1 mouth
  (zero invented words, trie on).

If that single-NPC slice feels alive on real hardware, scaling to the
Briar Glen town cast is ordinary engineering. If it does not, more
parameters or more model files will not save it.

## 5. What this deliberately does *not* do

- Does not require a larger GRU or H > 320 as a prerequisite.
- Does not require an NPC-to-NPC social graph.
- Does not put free-form open-vocabulary player text into the prompt
  without the density controls already warned about in `ideas.md` #11
  and the M9 freeform-corpus failure.
- Does not treat model swapping as the primary path to distinct voices.

## 6. Suggested first spike (still not a milestone)

1. Persist a minimal relationship vector + 8-slot memory block per
   important NPC (SD or EEPROM).
2. Wire EventBus → simple reaction table → relationship / memory update.
3. Enrich ContextBuilder to retrieve 1–2 memories and emit one extra
   conditioning token.
4. Run the Old Man Rowan test on Ares + one EverDrive spot-check.
5. Only after that measurement decide whether Tier-2 model files or
   richer axes are justified.

The mouth is already reliable. The work that turns it into living
characters is almost entirely in the state layer and the director that
feeds it.
