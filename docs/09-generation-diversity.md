# 09 — How many things can this model actually say? (M10)

A question worth a real answer, not a hand-wave: given a template-
grammar corpus and a trained char-level GRU sampling from it, how many
*distinct* lines can the shipped game actually produce? This walks
through the math, computed against the real phrase banks as of M10
(Selena, guard, the compositional cast, Shadewrath).

## The wrong way to estimate it: character-level entropy

The tempting shortcut: the sampler does top-k=5 at every character, over
a ~68-symbol vocabulary, so isn't the ceiling `5^L` for an L-character
response? Technically that's a real structural bound (the sampler
literally cannot pick outside its top-5 candidates at each step), but it
wildly overstates what actually happens. Training loss (`val loss`,
printed by every `make_mN_blob.py`) is cross-entropy in **nats per
character** — M10's converges to roughly 0.10 nats/char, about 0.14
bits/char. That means each character is around 90% predictable once the
model knows what template it's inside. The real "choice" isn't which
letter comes next; it's which *line* got pulled from a bank. Character-
level math answers the wrong question.

## The right way: count the template grammar's own combinatorics

Every corpus module (`selena_corpus.py`, `guard_corpus.py`,
`cast_corpus.py`, `shadewrath_corpus.py`) builds a response the same
way: a fixed sequence of clauses, each either **mandatory** (always
included) or **optional** (a coin-flip in `_response()`, e.g. "60%
chance of an opener"). For one fixed `(trust_tier, mood, context)`
prompt combo:

```
distinct_strings(combo) = Π over each clause of:
    bank_size            if the clause is mandatory
    bank_size + 1         if the clause is optional (+1 = "skipped")
```

Then sum that across every distinct prompt combo the character has
(trust tiers × moods × contexts, sometimes × occupations/descriptors for
the compositional cast). Optional clauses matter more than they look --
each one *multiplies* the space, not adds to it, so a character with 6
independent optional clauses doesn't get roughly double the diversity of
one with 3 clauses, it gets something like the *square* of the order of
magnitude.

## Real numbers, computed from the actual banks (M10)

| character group | distinct strings | optional clauses |
|---|---|---|
| Selena | 1,370,160 | opener, closer (2) |
| Guard (4 instances combined) | 17,280 | opener, closer (2) |
| Compositional cast (7 characters: Bram/Fergus/Kragan + M10's 4 town archetypes) | 5,065,606,080 | opener, descriptor-tic, occupation-flavor, catchphrase, closer (5) |
| Shadewrath | 745,416 | opener, catchphrase, closer (3) |
| **Total across the whole shared model** | **~5,067,738,936** | |

The compositional cast dwarfs everything else not because its phrase
banks are bigger, but because `cast_corpus._response()` has more
independent optional clauses than Selena's or Shadewrath's — this is a
direct, measurable consequence of M9's compositional design (more
composable axes = more clauses = combinatorially more voice variety per
character, the entire point of building it that way).

## The wrinkle this surfaces: the seed space can become the real ceiling

The sampler is deterministic given `(prompt, seed)` — same seed always
reproduces the same output, the same bit-exactness guarantee this whole
project runs on. The seed is a `uint32_t`, so there are at most 2^32 ≈
4.3 billion distinct seeds to ever draw from, for *any* prompt.

For Selena, guard, and Shadewrath, their template spaces (low millions)
sit comfortably under that 4.3B ceiling — the grammar is the real limit
on their diversity, not the seed. **For the compositional cast, the
opposite is true: its ~5.07B-string template space actually exceeds the
32-bit seed space.** Past that point, adding more phrase-bank content
doesn't add reachable diversity — the seed becomes the bottleneck instead
of the grammar. Worth remembering if a future corpus pass tries to make
the cast's banks even richer: past ~4.3B combinatorial strings for one
character group, further growth stops paying off under the current
32-bit seed.

## Caveat

This is the corpus *generator's* structural space — what the template
grammar could produce — not a proof the trained model reproduces every
one of those combinations. But given how low the measured per-character
entropy actually is, the model is tracking this structure closely rather
than freelancing character-by-character, so it's a meaningfully accurate
estimate of real diversity, not just an upper bound nobody approaches.
