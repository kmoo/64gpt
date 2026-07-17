# 08 — The archetype/character manifest: mechanism vs. content

M7 proved one thing at real scale: a single shared model can carry more
than one voice if you prime it with the right conditioning string
(`N:selena TR:2 M:cheerful C:item_found EV:found_gem`), and that
identity signal doesn't wash out the way a mood or trust swap might —
identity-swap divergence 0.9405 vs. mood 0.9168 / trust 0.9099 / context
0.9409 (trigram-Jaccard, sampled), decision gate **PASSES**, confirmed
at H=256 (`docs/milestones/m7.md`, Evaluation Protocol). That result is
what unblocks this document: M8 proceeds on the text-priming path as
scoped, not the learned-embedding fallback M7 flagged as a contingency.

But M7's schema baked 64GPT-demo specifics into the field vocabulary —
`M:cheerful`, `C:item_found`, and Selena's five personality trait names
(`Warmth`, `Humor`, `Impulsivity`, `Bravery`, `Focus`) were all written
as if they were part of the mechanism. They aren't. A horror game wants
`menace`/`paranoia`, not `warmth`/`humor`. This doc separates the two
so a second project can supply its own vocabulary without touching
`core/` or the Context Builder.

## The two layers

| layer | what it is | who owns it |
|---|---|---|
| **Meta-schema** | the shape: `characters[]`, `archetypes[]`, `schema_fields{}`, the tier system (full/mid/thin — full spec at M9, thin is what M7 proved) | ships with the toolkit, fixed |
| **Project manifest** | one JSON file per game — the actual mood list, trait names, character entries | authored per-project, e.g. the dungeon crawler's |

The meta-schema is mechanism: it says "you must declare a list of
personality trait names," never what those names are. The project
manifest is content: it says `["warmth", "humor", "impulsivity",
"bravery", "focus"]`. Every consumer of a manifest field — the Context
Builder, the corpus generator, the NPC Database — reads the field name
out of `schema_fields`, never hardcodes it. This is the same discipline
`core/`'s frozen streaming API already applies to the model itself,
extended to the data that primes it.

## Shape

```json
{
  "version": 1,
  "schema_fields": {
    "mood": ["cheerful", "worried", "sassy", "tender", "embarrassed"],
    "context": ["greeting", "combat-banter", "item-found", "quiet-moment"],
    "trust_tiers": [0, 1, 2, 3],
    "personality_traits": ["warmth", "humor", "impulsivity", "bravery", "focus"]
  },
  "characters": [
    {
      "id": "selena",
      "tier": "full",
      "personality": {"warmth": 90, "humor": 85, "impulsivity": 70, "bravery": 55, "focus": 30},
      "bible": {"public": "...", "private": "...", "secret": "...", "fear": "...", "desire": "..."},
      "corpus_ref": "corpus/selena/",
      "memory_persistence": "full"
    }
  ],
  "archetypes": [
    {
      "id": "guard",
      "tier": "thin",
      "personality_ranges": {"warmth": [20, 50], "humor": [10, 40]},
      "corpus_ref": "corpus/archetypes/guard/",
      "name_gen": "fantasy_male_names.txt"
    }
  ]
}
```

### `schema_fields{}`

Declares the field *names* every `characters[]`/`archetypes[]` entry is
allowed to use, plus their legal values (for categorical fields like
`mood`/`context`) or just the axis name (for `personality_traits`,
whose range is per-entry, not global). This is the single place a new
project edits to swap 64GPT-demo vocabulary for its own — nothing
downstream should have a mood or trait name literal in it.

### `characters[]` — one fixed, authored individual

A `characters[]` entry is a named, hand-built character: a fixed point
in personality space (not a range), a full character bible, its own
corpus slice, its own memory persistence. Selena is the only entry that
exists as of M7/M8. This is the `full` tier — everything M7 built.

### `archetypes[]` — one template, many instances

An `archetypes[]` entry is *not* a character — it's a generator:

- `personality_ranges` — a min/max per trait (keyed by
  `schema_fields.personality_traits`, not a fixed point)
- `corpus_ref` — a shared corpus slice all instances of this archetype
  draw voice from
- `name_gen` — a rule for generating instance names

**Instances are not declared in the manifest.** This is the answer to
the "generic type / default / random" question M7's planning first
raised: an instance is `archetype_id + seed`, resolved at runtime (NPC
Database, M8 §2) by deterministic jitter — the seed selects a point
inside the archetype's `personality_ranges` via xorshift32 (same PRNG
already used by the M4 sampler, one RNG discipline project-wide), plus
a generated name and an empty memory slot. The conditioning string's
`N:` field carries this as `archetype:seed`, e.g. `N:guard#4f2a`. The
manifest declares the template once; the game world can spawn as many
jittered instances as it wants without touching the manifest or
retraining — only the archetype's corpus and range need to exist ahead
of time.

Two `guard` instances therefore share a corpus and a personality range
but resolve to different jittered values, so they should read as
individuals, not clones. Whether that actually holds is not assumed —
it's the new **within-archetype divergence metric** (`guard#1` vs.
`guard#2`, same discipline as M7's between-category table), measured
before the archetype system is called proven (M8 DoD).

### Tiers

`full` (Selena, M7) and `thin` (guard archetypes, M8) are in active use.
`mid` is named here because the field exists in the shape, but its
actual definition — what sits between "full character bible" and "range
+ shared corpus" — is M9 scope, not decided yet. Don't build against a
`mid` tier expecting specific semantics until M9 writes them.

## Authoring convention: `_`-prefixed keys

Any key starting with `_` (e.g. `_meta`, `_status`) is an authoring note
for humans reading the manifest — provenance, TODOs, "not yet
validated" flags. Tooling (the validation test above, any future
manifest editor from M10) must ignore these keys entirely; they are
never read as schema content. See `manifests/dungeon_crawler.json`'s
`guard` entry for the pattern: `corpus_ref`/`name_gen` are `null` with
an `_status` note explaining why, rather than a fabricated path to an
asset that doesn't exist yet.

## Validation is a test, not documentation

Once `personality_traits` (and `mood`/`context`/`trust_tiers`) are
declared in `schema_fields` rather than hardcoded, every
`characters[].personality` and `archetypes[].personality_ranges` key is
checkable against that declaration — same discipline as M4's corpus-
generator grammar/charset/length invariants. This test belongs in the
trainer test suite alongside the first manifest that uses it (M8 §7),
not retrofitted after a typo'd trait name silently trains a model that
ignores a slider nobody meant to drop.

## Why this doc exists

This is the concrete artifact M10's porting guide points at. "Supply a
file shaped like `docs/08-manifest-schema.md`" is a testable
instruction a second project's engineer can follow without reading this
project's source; "supply your own character bibles" was not.
