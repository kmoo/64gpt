# Ideas — raising the wow factor 10× (post-M5 brainstorm, 2026-07-16)

Not a roadmap — a ranked idea pool for after v1.0. Each entry: what it
is, why it lands, effort/risk. The walking-skeleton rule still applies
to anything adopted from here: it ships inside a booting ROM or not at
all.

## 1. A real scene instead of the cube
A tiny village square, three low-poly NPCs (guard, merchant, wizard) —
walk up, press A, they speak. The moment the AI is *a game feature*
rather than a tech demo, every other idea doubles in value. Pyrite64 is
built for exactly this. **Medium effort, no risk.**

**Update (2026-07-22):** this became a real, detailed game design —
"Briar Glen and the Everhollow," an 8-region spatial overworld. See
`docs/ideas-briar-glen-world.md` for the full vision and how it maps
onto what's actually built so far (still a flat NPC-cycling demo, no
spatial engine yet — this idea's "medium effort, no risk" framing
turned out to undersell it once the actual scope became clear).

## 2. "Watch it think" overlay
Draw the top-5 next-character candidates as live bars beside the
streaming text — the distribution visibly reshaping as each letter
lands. The logits and sampler weights already exist; this is ~50 lines
of draw code. Makes the invisible visible. **Low effort, huge payoff.**

## 3. Lockstep bit-exactness theater (for the talk)
Side by side on stage: a Mac running `ref_impl.py` and the N64 on a CRT,
same seed, generating the identical line character-by-character in
sync. The project's thesis as a 20-second visual. Everything needed
already exists. **Trivial effort.**

## 4. NPC-to-NPC argument mode
Guard and merchant alternate turns about the theft — each turn is just
another `ngpt_reset` with the scene's prompt. Emergent-feeling banter on
1996 silicon. **Low-medium effort.**

## 5. Character voices (audio blips)
Banjo-Kazooie-style mumble: per-NPC pitch/timbre, one blip per streamed
character. Zero ML, enormous charm, makes every video clip shareable.
**Low-medium effort.**

## 6. RSP matvec — the 5–10× headline
Offload the matvec to the RSP (the graphics coprocessor's 8×16-bit
vector unit). Buys H=256+ or instant replies, and "the neural net runs
on the graphics chip" is a talk-title sentence. Already the documented
stretch goal (plan.md). **High effort, high risk, highest ceiling.**

## 7. Player-composed prompts (word picker)
A picker UI over topic/mood/place tokens from the training grammar, so
the audience *asks* instead of watching a cycle. The safe sibling of
idea 11. **Medium effort, low risk.**

## 8. Quest-state memory
Extend the prompt protocol with game state (`EV=QUEST_DONE`,
`MET=TRUE`) and regenerate the corpus — NPCs that visibly react to what
the player did. Conditioning as *living world*, not string selection.
**Medium effort — corpus work; the pipeline exists.**

## 9. "HALT, LUKE!" personalization
N64 name-entry screen feeds a NAME slot the grammar trained over, so
NPCs greet the player by name. Char-level *copying* is hard for a ~68K
model — needs a training experiment before promising it. **Medium
effort, real risk, peak delight if it works.**

## 10. The physical artifact (M6, staged well)
Real console, real CRT, EverDrive in a labeled cart shell, filmed
properly. The prop on the podium is the wow. **Low effort — it's
already the plan; do it theatrically.**

## 11. Free-text player replies (Luke's) — 1–2 typed words → conversation
An on-screen keyboard (name-entry style: A–Z, space, END) lets the
player type one or two words in response to the NPC's last line; the
reply is spliced into the prompt and the NPC answers *that*:

    NPC=GUARD MOOD=ANGRY EV=THEFT SAY=GIVE BACK|

Why this is feasible **with zero engine changes**: the frozen API
already primes on arbitrary strings, and the char-level vocab already
covers everything typeable on the keyboard UI. The work is entirely in
the corpus + demo:

- **Corpus**: add a SAY= slot to the grammar, trained over a pool of
  plausible player utterances grouped by *intent* — apology (SORRY, MY
  BAD), denial (NOT ME, LIAR), question (WHO, WHY, WHAT NOW), bargain
  (HOW MUCH, DEAL), aggression (FIGHT ME, MOVE) — each intent mapped to
  response styles per NPC×MOOD. The model learns to route on salient
  words, not memorize strings.
- **Demo**: keyboard UI + splice the typed text into the prompt before
  `ngpt_reset`.
- **Failure mode is graceful**: a word the model never saw primes
  nothing specific and the NPC answers in generic mood-consistent
  voice — a shrug, not a crash. (Truly unknown *characters* are already
  skipped by the priming rule since M3.)
- **Test story**: intents are enumerable, so goldens pin one seeded
  reply per (intent × NPC × MOOD) — the self-test discipline survives
  free input.

Combined with idea 4 it yields actual back-and-forth *conversation* on
the N64 — likely the single biggest wow-per-effort on this list.
**Medium effort (mostly corpus + retrain), low risk.**

**Caution, added 2026-07-17 (M9 session):** the intent-grouped design
above (a small trained pool of plausible utterances, not raw open text)
turns out to matter more than it looked when this was written — M9's
first compositional-corpus attempt tried something closer to true
open-vocabulary freeform text (130 near-unique LLM-generated personas)
and the trained model produced visibly garbled spelling, traced to too
little repetition per word at this model's tiny size (~394K params).
Build this idea with the SAY= vocabulary as constrained/repeated as the
intent groups above already imply — resist the temptation to widen it
to genuinely free text later without re-checking corpus density first.

---

Recommended first bites: **2 + 3 + 5** (near-immediate, transform the
demo's feel), then **1** as the foundation, then **11** as the
conversation unlock; **6** is the moonshot worth one dedicated day.

---

**See also:** `ideas-m7-living-npcs.md` — the M7 vision (living NPCs:
continuity of existence, external memory/personality/world state,
gossip, Old Man Rowan test, engine-first Pyrite64 architecture).
Supersedes the scale-focused framing of several entries above.

## 12. Seed ensembling as a training-variance mitigation (2026-07-28/29, prompted by M13's noise-floor finding)

M13's own baseline experiment measured something not previously
quantified: two IDENTICAL training runs (same corpus, same
architecture, only the random seed differing) produced a guard+korrath
coherence gap of 0.31 invented-words/line -- bigger than M12.2->M12.4's
own real, confirmed improvement (0.21). The QAT phase specifically is
where the two seeds diverged (float losses nearly matched); the float
phase looks stable. If that pattern holds up under more sampling, an
obvious mitigation is training 2-3 seeds and either (a) picking the one
with best QAT-phase val loss (cheap, no inference-time cost, but throws
away the other runs' work) or (b) averaging weights across seeds
(cheaper still to deploy, unproven whether GRU weight-averaging even
produces a coherent model rather than a Frankenstein of two solutions).
**Not costed or tried.** Worth a small spike measuring whether (a) or
(b) actually shrinks the noise floor before trusting either for a real
milestone decision -- same "measure before committing corpus budget"
discipline as every conditioning-mechanism idea in this doc. If QAT
variance turns out to be fixable directly (idea 13 below) rather than
just something to average away, that's the better fix.

## 13. Diagnose WHY quantization-aware fine-tuning is the noisy phase (2026-07-28/29, companion to idea 12)

A narrower, more targeted version of idea 12: M13's baseline pair
showed the float phase converging to nearly identical val loss across
seeds (0.1341 vs 0.1339) while QAT diverged sharply (0.1413 vs 0.1818)
from the same two starting points. That's a specific, falsifiable
question -- is this inherent to the straight-through-estimator fake-
quant setup at this model size (small weight matrices mean each
int8-grid rounding decision is a bigger relative perturbation), or is
it something more mundane and fixable (QAT's learning rate/patience
tuned for the general case but happening to sit in a locally unstable
regime, or the fake-quant hooks having their own seed-sensitivity
`qat_finetune`'s `torch.manual_seed(seed)` doesn't fully pin down)? A
small controlled sweep (several seeds x a couple of QAT learning rates/
patience values, float-phase-only weights held fixed as the starting
point so only the QAT hyperparameters vary) would tell the difference
directly, and unlike idea 12's ensembling this could actually close the
noise floor rather than just average around it. **Not costed or
tried** -- flagged as the more scientifically satisfying follow-up to
M13's own "worth separately investigating, NOT done tonight" note.

## 14. Conditions stack: one greeting, four independently-varying axes (2026-07-28/29, prompted by tonight's NPC-engine headers)

Tonight's session built five small, independent state-layer headers
(`TimeOfDay.h`, `NpcCondition.h`, `LocationAtmosphere.h`,
`VisitFrequency.h`, `PlayerAppearance.h`) alongside the existing
`PlayerReputation.h`/`NPCState.h::Relationship` -- each one answers a
different "why does this line read differently" question from
`ideas-m7-living-npcs.md` Part 4, but none of them are wired into the
actual conditioning string yet (deliberately -- each header's own
comment says so). The idea, not yet built: a single demo *moment* where
several of these axes are visibly true at once for the same NPC and the
player can see the game state driving it -- e.g. a guard who is
NIGHT + TIRED + at a TENSE location + being visited by a RAGS-appearing
player for the first time in weeks, versus the same guard DAY +
NORMAL + NEUTRAL location + a FINE-appearing regular. This is a demo/
staging idea, not a training idea -- the actual dialogue difference
still requires deriving new conditioning tokens and retraining (real,
separate corpus work, same caution as every other new schema axis in
this doc), but even BEFORE that retrain exists, the game-state layer
alone could drive a visible on-screen debug readout ("GUARD#1002:
NIGHT, TIRED, TENSE, first visit in 34 days") as a talk-demo moment
proving the world-state plumbing is real, ahead of the model actually
reacting to it. **Low effort for the debug-readout version (all the
state headers already exist and are tested), medium-to-high effort for
the version where the model's dialogue actually changes.**

## 15. A "genre starter pack" alongside the porting guide (2026-07-28/29, prompted by the M14 portability proof)

M14's portability proof (`manifests/scifi_freighter.json`,
`trainer/ngpt_trainer/scifi_engineer_corpus.py`) demonstrates the
toolkit ports to a new genre by hand-authoring one archetype's schema
fields and corpus from scratch. If that proof holds up (real coherence,
not just a clean manifest validation), a natural next idea is a small
library of PRE-BUILT starter schema_fields blocks for a few common
genres (fantasy, sci-fi, contemporary/noir) -- mood/trust_tiers/
audience already established as genre-agnostic and reusable as-is;
occupations/context/species per genre would still need real authored
corpus content (this doesn't shortcut that), but a new project could at
least start from "here's a sci-fi schema_fields block with 15 sensible
occupation names" instead of inventing the vocabulary from zero. **Not
costed.** Only worth pursuing if a second real project actually adopts
the toolkit -- speculative infrastructure for a user that doesn't exist
yet is exactly the kind of premature abstraction this project's own
engineering discipline warns against, so this stays an idea, not a
task, until that changes.
