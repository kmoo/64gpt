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
