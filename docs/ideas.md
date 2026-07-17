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
