# M7 idea — living NPCs: worlds, not conversations (recorded 2026-07-16)

Two companion notes from Luke (recorded nearly verbatim per request;
only the diagrams/prompts are fenced so they render). Future-step
material — goes with `docs/ideas.md`, not scheduled.

---

## Part 1 — continuity of existence beats model size

You are a PHD hardware and software engineer from the 80s and N64
expert. How would you make this even more amazing in terms of power of
the neural network and complexity to wow people? I love where this is
going, but I want to push back again on a few things because the
difference between "cool demo" and "people still talk about this in 20
years" is going to be ruthless prioritization.

The vision is right. Some implementation assumptions still smell like
modern AI thinking.

The biggest insight:

**The thing that will make people say "this NPC is alive" is not
linguistic quality. It is continuity of existence.**

A 500k parameter model that remembers, lies, changes, and participates
in a world will feel more alive than a 5M parameter model that spits
pretty sentences but has no life.

### Pushback #1: Don't make the model bigger. Make the world deeper.

I would actually consider going smaller for V1. Something like:

```
Language Core:        200k-400k params
Personality System:   external
Memory System:        external
World Simulation:     external
```

Why? Because the model does not need to know: "Tell me a fantasy
story." It needs to know: "Given this tiny slice of this world's
reality, say something appropriate." The world does the hard work.

Bad prompt:

```
You are a medieval guard. Talk to player.
```

Model response: "Greetings traveler, welcome to our town." Who cares?

Good prompt:

```
YOU ARE:
Guard Thomas
PERSONALITY:
Honor: 92
Suspicion: 80
Humor: 25
MEMORIES:
- PLAYER entered town during wolf attack
- PLAYER ignored warning yesterday
- PLAYER gave child medicine
CURRENT FEELING:
Conflicted
TOWN EVENTS:
- Three merchants robbed
- Moon festival tonight
PLAYER:
"Can I leave town?"
```

Output: "Leave? Now? After what happened on the eastern road? … I won't
stop you. But if you value your skin, wait until dawn."

That feels alive. The magic is not the generator. **The magic is the
state.**

### Pushback #2: Don't make memories "facts"

Most games make memories databases. Humans don't work that way. Your
NPC memory should have:

Memory strength:

```
Silver cup stolen
Importance: 90
Confidence: 95
Age: 30 days
```

Emotional residue:

```
Anger: +40
Fear: +10
Respect: -20
```

Corruption, after time:

```
Original: "The hero returned the cup."
Rumor:    "The hero stole the cup then pretended to return it."
```

Now you have NPCs that feel human.

### Pushback #3: Gossip should be the killer feature

Imagine the demo: you walk into a village.

NPC 1: "You should speak to the blacksmith. He knows what happened."
Blacksmith: "Who told you that?" Farmer: "Everyone knows." Player:
"Knows what?" Farmer: "That you killed the dragon." Player: "I didn't
kill a dragon." Farmer: "That's not what the children say."

That is a viral moment. And it requires almost no neural horsepower.

### Pushback #4: Personality vectors should not only affect language

Don't use `Personality -> dialogue style`. Use
`Personality -> behavior -> dialogue`.

Two merchants. Merchant A (`greed: 90, risk: 20, honesty: 30`):
overcharges, hides information, avoids danger. Merchant B (`greed: 20,
honesty: 95`). Player asks "What's in the forest?" — A: "Nothing worth
your time." B: "Something is wrong there. I would not send my enemy."

### The feature I would add: NPC secrets

Every important NPC gets: PUBLIC BELIEF / PRIVATE BELIEF / SECRET /
FEAR / DESIRE.

Guard example — Public: "The forest is dangerous." Private: "I lost my
brother there." Secret: "I know the wizard caused it." Fear: "Everyone
will discover I abandoned him." Desire: "I want forgiveness."

Now your NPC is a character.

### The "impossible N64 demo" I would build

Not a town. One cabin. One NPC. **Old Man Rowan.**

First visit: "Strangers rarely come here." Second visit: "You again."
10 hours later: "I wondered if you would return." Player: "Why?"
"Because everyone leaves eventually."

No quest. No reward. Just a relationship. People will remember that.

### Architecture I would aim for

```
                GAME ENGINE
                    |
        +-----------+------------+
        |                        |
   World Simulator          NPC Database
        |                        |
   Events/Rumors          Memories/Soul
        |                        |
        +-----------+------------+
                    |
             Context Builder
                    |
             Tiny GRU Model
                    |
              Final Dialogue
```

The Context Builder is secretly the most important component. It is the
"director."

### About 1.5M parameters

My honest engineering guess:

| Model | Likely outcome |
|---|---|
| 100k | Interesting toy |
| 300k | Good NPC voice |
| 500-800k | Probably magic zone |
| 1.5M | Possible but diminishing returns |
| 5M | Probably fighting hardware |

I would rather have: 700k model + excellent memory + gossip +
simulation, than: 2M model + static NPCs. Every time.

### The thing I would build next

Not the architecture document. I would build the **"Old Man Rowan
test"** — a single NPC benchmark:

- remembers 10 interactions
- personality changes
- has secrets
- reacts differently based on player history
- survives save/load
- generates novel but consistent dialogue

If that works on actual hardware, then scaling to a village is just
engineering. And honestly? 64GPT has a chance to become something much
stranger and more interesting than "AI on old hardware." It could
become a new category: **procedural character AI designed for worlds,
not conversations.** That is a much bigger idea.

---

## Part 2 — Pyrite64 changes the architecture (engine-first)

Ah — that changes the architecture discussion significantly. If you are
building this in Pyrite64, then you are not constrained by writing a
literal N64 cartridge engine from scratch. You have a modern
development environment targeting N64-style hardware/software
constraints. That actually makes the vision more achievable, because
you can architect it like a real game first and treat the neural
network, audio, and world simulation as first-class subsystems.

I would structure it like a late-90s Nintendo internal engine, but with
a modern toolchain:

```
                    PYRITE64 GAME ENGINE
                           |
              WORLD SIMULATION LAYER
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v

   Dungeon System      64GPT Brain       Music System

   Rooms               Neural Model      Dynamic Score
   Enemies             Memories          Motifs
   Items               Personality       Atmosphere
   Combat              Intent            Audio DSP

        +------------------+------------------+

                    Shared World State
```

The key is: **do not let the AI system be bolted on.**

### The "world state bus"

I would make a central event system. Everything emits events:

```
EVENT_PLAYER_ENTERED_DUNGEON
EVENT_GOBLIN_DEFEATED
EVENT_NPC_HELPED
EVENT_SECRET_FOUND
EVENT_PLAYER_DIED
EVENT_ITEM_GIVEN
```

NPC brain listens — "Player saved child." → `Trust +15, Respect +10,
Rumor spread +5`. Music listens — "Ancient tomb discovered." →
`Mystery +30, Danger +20, Ancientness +40` → changes arrangement.
Dungeon listens — "Player has learned fire magic." → maybe new paths
unlocked. Everything stays synchronized.

### For Pyrite64 specifically, I would exploit modern development

I would not train on the N64. Pipeline:

```
PC Training Environment (PyTorch/JAX)
     |
     v
Quantization
     |
     v
N64 Neural Runtime
     |
     v
Pyrite64 Game
```

Train the brain on PC. Deploy the tiny inference model.

### The architecture I would target

```
/engine
    world.c
    dungeon.c
    combat.c
    actors.c

/ai
    neural.c
    memory.c
    personality.c

/audio
    music_state.c
    adaptive_score.c
```

### The neural model: a three-stage brain

1. **Perception** — "What happened?" Input: player attacked monster /
   helped NPC / stole item. Output: important event = yes, emotion =
   fear.
2. **Brain state** — persistent hidden state: trust, fear, curiosity,
   anger, attachment. This is where the "soul" lives.
3. **Expression** — turns internal state into dialogue, animation,
   behavior.

### Music architecture

This could be a huge differentiator. Do not generate MIDI randomly.
Make a composer create a musical grammar. A dungeon theme has stems:
base melody, harmony, percussion, bass, ambient layer, danger layer,
discovery layer, boss layer. The engine crossfades; world state
controls the mix.

Low health + ancient room: `bass ↑, heartbeat percussion ↑, choir ↑,
melody ↓`. Finding treasure: `bells ↑, strings ↑, danger ↓`.

### The "N64 magic trick"

The player should never see "AI generated dialogue." They should feel:
the blacksmith remembers my actions; the dungeon feels different after
I return; the music knows something happened; villagers tell stories
about me. **The neural network is invisible.**

Given Pyrite64, I would actually push the ambition higher than the
original hardware-only plan: small on-device neural network, much
richer world simulation, adaptive soundtrack, a 3D dungeon crawler
polished enough to stand alone.

The thing I would prototype first is not the NN. It would be **a single
dungeon room + one NPC + one adaptive music track + event bus.** If
that slice feels alive, the entire game architecture is proven.

---

## Part 3 — relationship state, formalized (Luke, 2026-07-17, during M8)

A refinement of Part 1's "brain state" (trust, fear, curiosity, anger,
attachment) into three explicit layers, and a scoping clarification that
makes it cheaper than it first sounds: **this is relationship-to-the-
player only, not an NPC-to-NPC social graph.** O(N) state (one vector
per NPC), not O(N²) — no relationship-graph subsystem needed.

1. **NPC Profile** (rarely changes): age, occupation, faction,
   personality (lawful<->chaotic, kind<->cruel), role/status, default
   goals. NPC *types* get sensible defaults (a guard defaults lawful, a
   merchant defaults trade-focused) so authors don't hand-set every
   value. This is M8's `archetype.personality_ranges` — already built,
   just not yet extended with an explicit faction/goals axis.

2. **Relationship State** (per-player, changes over time): familiarity
   (0-1, stranger->lifelong), affection (-1 to 1), trust (0-1), respect
   (0-1), fear (0-1), plus a relationship-type label (friend/family/
   rival/customer/employer). `NPCDatabase::NPC.trustTier` is already
   exactly this pattern at 1 axis, 3 buckets — this generalizes it to 5
   axes.

3. **World Context** (already built): events, items, location, quest
   state, time, weather, mood, nearby NPCs, topic, danger — EventBus /
   WorldState / ContextBuilder, unchanged. Explicitly **not** being
   replaced by this idea, only extended.

**Why it's cheap on the game-engine side, expensive on the model side:**
storing 5 floats per NPC instead of 1 int is free — the real cost is
that `core/`'s conditioning mechanism is a *text string* primed through
the frozen `ngpt_reset` API, not raw floats. Each axis needs bucketing
(same reason `trustTier` is 0/1/2, not continuous) before it can enter
the schema, and 5 bucketed axes multiply the training combo grid
combinatorially (3^5 = 243x at just 3 buckets each) against a model
that M8's Data Science Review already called capacity-constrained. Two
ways through, not yet decided: collapse the 5 axes into one derived
"disposition" bucket before they reach the schema string (cheap, fits
today's mechanism, loses resolution), or treat this as the strongest
case yet for the embedding-table fallback M7/M8 already flagged as a
contingency (scales additively with new axes, not multiplicatively).

**The missing piece is the update/decay system** — something has to
turn `EVENT_PLAYER_STOLE_ITEM` into `affection -0.3, trust -0.4`, the
same event-reaction pattern Part 1 already sketched (`Trust +15, Respect
+10` on `NPC_HELPED`). That's `docs/ideas.md`'s idea #8 (quest-state
memory) generalized from one-off flags to continuous decaying state —
not yet designed, would need its own spike before committing corpus
budget to it, same discipline as every other conditioning-mechanism
change on this project.

### Part 3, refined — compositional conditioning, not opaque identity
tags (Luke, 2026-07-17, later the same session)

Correction to an earlier claim in this same conversation that the
translation layer below "already exists" as `ContextBuilder`. True of
the *mechanism* (compile rich state down into the small discrete
vocabulary the frozen `ngpt_reset` API primes on) — wrong about the
*content*. Today's `N:` field is an **opaque per-character identity
tag** (`N:selena`, `N:guard#1001`): the model has to learn a dedicated,
non-transferable association between each raw tag string and a voice.
That's exactly why M8's density table (`docs/milestones/m8.md`, Data
Science Review) shows guard density and Selena's val loss moving in
*opposite* directions — every new identity tag competes for its own
slice of the same shared capacity, because nothing about the tags
themselves is shared or reusable.

The refined idea: replace the opaque tag with a small set of
**reusable, compositional descriptive features** instead — e.g. `girl,
age:12, sassy, VILLAGER` for the profile layer, `1.0 (best friend)` for
the relationship layer — rather than a raw id. A second sassy
12-year-old girl NPC would then condition on nearly the same tokens
Selena already does, and the model learns feature→voice associations
that *generalize* across characters instead of per-id memorization that
only grows. This is the actual structural fix for the capacity-dilution
problem M8 measured — not just fewer buckets (the mitigation path
originally noted above), but a different kind of encoding where shared
traits share capacity instead of competing for it.

**Occupation belongs in the feature set, and M8 already validated it as
one** (Luke, same discussion): `guard` isn't just an archetype id, it's
occupation acting as a single coarse compositional feature — M8's whole
result (4 guard instances, distinguishable from each other *and* from
Selena, sharing one corpus slice) is evidence that occupation alone
already produces a coherent, reusable voice signal. The refined scheme
just makes that explicit and combines it with the other dimensions
(age/gender/personality/relationship-tier) instead of occupation being
the *only* axis a character gets, the way today's flat archetype-vs-
character split forces it to be.

What this changes structurally:

- The C-side translation service (game code, not `core/`) computes this
  compositional tuple from NPC Profile + Relationship State + World
  Context. The model still only ever sees a short discrete string
  through the frozen `ngpt_reset` API — no `core/` changes, same as
  every mitigation path considered so far.
- `npc.id` (e.g. `"guard#1001"`) stays as an internal game-engine lookup
  key — `NPCDatabase` indexing, save state, `memorySlot` — it does
  **not** need to be what's fed into the conditioning string anymore.
- Corpus authoring shifts from "one voice pool per character id"
  (today's `selena_corpus.py`/`guard_corpus.py` shape) to "voice
  content organized by feature combination" — a sassy-12-year-old-girl
  OPENER pool any matching NPC draws from, authored once, reused by
  every character with those features. That's a real methodology
  change, and it's the same territory M8.1 (LLM-generated corpus vs.
  template grammar, `docs/milestones/m8.1.md`) already flagged as worth
  testing — generating rich per-feature-combination lines at scale is
  arguably a better fit for LLM generation than the current
  per-archetype isolated-dispatch pattern M8 used.

**Open question, not yet answered:** how many feature dimensions, and
how many values each can take, before the combo grid blows up again —
the same combinatorial concern as before, just moved from "5
relationship axes" to "however many profile+relationship feature
dimensions get declared." Worth a small spike (a handful of features,
a handful of values each, measured with the same divergence methodology
M7/M8 already use) before committing real corpus budget to it — same
discipline as every other conditioning-mechanism change on this
project.

## Part 4 — player reputation as global state (2026-07-17, during M9)

A gap check against this doc during M9's corpus-quality debugging: most
of what "an amazing NPC system" needs turned out to already be recorded
here (continuity-of-existence over model size, gossip via EventBus now
scheduled as M11's headline feature, event-specific memory in Part 1's
"Memory strength/Emotional residue/Corruption" sketch). Two things
genuinely aren't recorded anywhere yet:

**Player reputation — global, not per-NPC.** Part 3's Relationship State
is explicitly scoped as relationship-*to-the-player*, one vector per
NPC (O(N), not a social graph). Reputation is a different axis entirely:
a single shared value ("Hero," "Thief," "Stranger") that *every* NPC's
conditioning can read, independent of whether that specific NPC has
ever met the player. A guard who's never spoken to you before could
still greet you differently if you're locally famous — relationship
state alone can't express that (it's undefined for a stranger by
design), reputation can. Cheap to store (one global int, not O(N)) but
raises the same combo-grid concern Part 3 already flagged for
relationship axes: another schema field multiplies the training combo
grid, and needs its own divergence-methodology check before committing
corpus budget, not assumed to be free just because storage is cheap.

**Time-of-day** is actually already *listed* in Part 3's "World Context"
bullet ("events, items, location, quest state, time, weather, mood...")
but that was aspirational, not built — no `T:` field exists in the
current schema, and nothing computes or feeds time-of-day into
`ContextBuilder` today. Flagging that the doc already claimed this as
"already built" when it wasn't concretely wired — worth fixing that
claim's precision next time this doc gets revised, not just adding it
to the pile of future ideas.

Neither is scheduled or scoped into any milestone yet — recorded here
so they don't get lost, same as everything else in this doc.

### More candidate conditioning variables (brainstormed same session)

Checked against the existing list (mood/context/event/relationship/
reputation/time/weather/topic/danger/nearby-NPCs) to avoid duplicates.
None of these are scoped or costed yet — same combo-grid-multiplication
caveat as every axis above applies before committing corpus budget to
any of them:

- **Player's visible gear/appearance** — fine armor vs. rags changes how
  a guard or merchant opens a conversation, independent of actual
  reputation (a well-dressed thief still gets the polite greeting).
- **NPC's own current condition** — tired, injured, drunk, mid-task —
  separate from fixed personality; a normally-cheerful innkeeper reads
  differently after a long night. Distinct from Part 1's "emotional
  residue" (that's reaction to the player specifically; this is the
  NPC's own independent state).
- **Location atmosphere** — festive (festival day), tense (recent
  crime), abandoned — a property of the *place*, not the NPC's personal
  mood; the same NPC could read differently depending on where the
  conversation happens.
- **Visit frequency, not just depth** — a regular daily visitor vs.
  someone who shows up once a season. Distinct from relationship
  "familiarity" (Part 3), which tracks depth/duration, not cadence —
  two players could reach the same familiarity score via very different
  visit patterns and that difference is currently invisible to the NPC.
- **NPC's active want/need** — a merchant who needs a specific item and
  reacts specially if the player happens to be carrying it, tying
  dialogue to real inventory/quest state rather than just vibes.
- **Player's current health/danger in the moment** — distinct from a
  past `EVENT_TOOK_DAMAGE` (already covered): an NPC noticing you're
  visibly near death *right now*, not reacting to something that
  happened earlier.
- **Reputation vs. actual behavior divergence** — Part 4's reputation
  idea and a player's real tracked behavior could be tracked as two
  separate values that sometimes *contradict* (secretly stealing but
  never caught = good reputation, bad behavior) — narratively richer
  than either alone, an NPC who "just has a feeling" about you despite
  your clean public record.

**Explicitly reconsidered and still ruled out**: NPC-to-NPC
relationships (a guard who resents a specific merchant) — tempting, but
Part 3 already scoped this out on cost grounds (O(N²) social graph vs.
O(N) player-relationship state) and nothing about this session changes
that math.

### Player actions as a state-mutation source, not just a generation input

The player should be able to *change* an NPC's mood/relationship state
mid-conversation, not just receive dialogue conditioned on it — saying
something kind to an angry NPC calms them down; giving a gift raises
affection. This is the missing connective tissue between two ideas
already recorded separately:

- `docs/ideas.md` idea #11 (free-text player replies): a `SAY=` slot
  lets the player's typed/chosen words steer what the NPC says *next*,
  grouped by intent (apology, kindness, aggression, ...) — but as
  designed, that's a one-turn effect. Nothing about it persists.
- This doc's Part 3 "missing piece" (the update/decay system): designed
  so far only for *world/quest* events (`EVENT_PLAYER_STOLE_ITEM` →
  `trust -0.4`) — dialogue choices were never wired in as an event
  source, only things the game engine does to the player independent of
  conversation.

The fix is treating a recognized `SAY=` intent as just another event
type the same update/decay system already has to handle — a kind word
is `EVENT_PLAYER_WAS_KIND` the same way stealing is
`EVENT_PLAYER_STOLE_ITEM`, both feeding the same `affection +/-, trust
+/-` mutation pipeline, one from a world action, one from a
conversational choice. No new mechanism needed once the update/decay
system exists — this is a scoping note (dialogue choices are in-scope
as an event source, not a separate system) more than a new build item.
Giving an item likely routes through the *existing* item/inventory
event path rather than needing dialogue-specific plumbing at all.

Still gated on the same "missing piece" both source ideas already
flagged as not yet designed — this doesn't unblock it, it just says the
eventual design needs to cover both event sources from day one instead
of bolting dialogue-driven mutation on as an afterthought later.
