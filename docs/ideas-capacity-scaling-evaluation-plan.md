# Data Science Review: does more model capacity actually fix the coherence problem? (2026-07-22)

**Not a milestone doc yet** — this is the experimental design a future
milestone (working number: M12, pending the M12→M13 portability
renumber Luke's considering) should execute, written before any of that
implementation starts, per Luke's explicit ask: "we need to ask the
data scientist to write the plan for that model so we see real,
tangible quality improvements and not just garbage/garble." Grounded in
the real, hardware-verified findings already on record in
`docs/spikes/rsp-matvec-ktile.md` (the K-chunk tiling spike, branch
`spike/rsp-matvec-ktile`) — not re-derived from scratch, and not
assuming that spike's own biggest flagged gap is already closed.

**Revised 2026-07-22 after a three-reviewer DS pass.** All three
reviews are genuinely good and are incorporated below where they
strengthen the experiment without changing what it actually tests.
Some recommendations are explicitly **scoped down or deferred rather
than adopted wholesale** — noted inline with the reasoning, not
silently dropped — because this project trains on one CPU, on one
laptop, shared with other work, usually with one human evaluator, and
the walking-skeleton discipline this whole project has followed since
M0 argues against quietly ballooning a capacity experiment into a
multi-axis research program before the first result is even in.

## The actual question, stated precisely

Not "can the RSP kernel run at H=768?" — the ktile spike already
answered that, hardware-verified, nine bit-exact XCHK passes. The real
question, quoted directly from that spike's own Handoff section because
it names the gap exactly: **"whether a real model at whatever H gets
chosen, trained on real corpus, produces better dialogue — not just
correct/fast inference — is completely untested. Tonight proved the
kernel, not the model."** Every ktile test used a throwaway
one-sentence gibberish model. Zero evidence exists, in either direction,
that a bigger H actually reduces the invented-word garbling
(`"HOMESTELD"`, `"USPECTID"`, `"ASWACT"`) documented for Shadewrath/
Korrath since M10. This plan exists to produce that evidence honestly,
whichever way it points.

## Why this isn't a safe assumption

This project has been burned by exactly this gap before, twice on
record: M7/M8 both had moments where val loss and other numeric metrics
looked fine while sampled text was visibly garbled — the whole reason
this project's Data Science Review discipline insists on human-eyeball
review of real generated text, not just loss curves. "More capacity"
is a plausible hypothesis (it's the other half of the documented
root-cause guess in `docs/plan.md`'s Known follow-ups: "either share
more structural content across characters... or accept this needs more
model capacity than a corpus change alone can buy"), but plausible
isn't measured. The M11 retrain in progress as this doc is written is
testing the *first* half of that hypothesis (shared Ravendale-lore
content); this plan is for testing the second half, and it needs the
same rigor.

## Real prerequisites before any quality comparison can run

These aren't implementation details to wave past — each is a genuine
open question or a real decision the ktile spike's own handoff
explicitly deferred:

1. **`core/ngpt.h`'s `NGPT_GRU_MAX_HIDDEN` bump needs explicit human
   sign-off.** `core/` is a frozen interface per this project's hard
   constraints (`CLAUDE.md`) — the ktile spike bumped it to 1024
   worktree-locally and was explicit that touching it on `main` is not
   something to do unilaterally. **This is a decision for Luke, not
   something a data-science plan can pre-approve.**
2. **int32 overflow analysis must be re-validated against REAL trained
   bias magnitudes, not the gibberish model.** The spike's own doc:
   H=512 sits at 99.2% of a self-imposed *comfort* margin, H=1024 sits
   at 99.2% of the *hard* ceiling with none. H=768 is unmeasured on
   this specific axis. Four mitigation strategies are recorded and
   unapplied (widen the bias-add to int64 only, saturating add, shift
   Q-format down, mid-reduction rescale) — pick one only if a real
   trained model's actual weights show the problem, not preemptively.
3. **Reconcile with this session's own H=320 work.** The ktile spike's
   handoff item #5 names this directly: coordinate before either side
   merges, so `core/ngpt.h` doesn't get two uncoordinated bumps from
   two different sessions. This plan's execution needs to happen
   *after* M11 ships at H=320 (below), specifically so there's one
   clean baseline to diverge from, not two moving targets.
4. **Game-side wiring is stale relative to current `main`.** The
   spike's buffer sizing/guards in `DialogueDemo.cpp` were last touched
   under an H=256-shaped assumption per the spike doc's own status log
   — current `main` has moved substantially since (H=320, the gossip
   mechanism, SaveData, the princess). Integrating the K-tiled kernel
   means merging into current `main`, not reviving a stale branch
   wholesale.
5. **The comparison corpus must be M11's real shipped mix, not a
   subset.** Whatever M11 ships with (currently retraining: Selena +
   guard + 9-character compositional cast + Shadewrath + Korrath +
   Elowen + the Ravendale-lore bank) is the corpus both H=320 and
   H=768 get trained on. A stripped-down or different corpus for the
   H=768 run would make the comparison meaningless.

## Experiment design

**Controlled variable: H only.** Same corpus (M11's shipped mix, same
seed), same training procedure (`train_corpus_conditioned`, same
`max_epochs`/`patience`), same evaluation battery, same golden prompts
where the schema allows direct comparison. The only thing that changes
between the two runs is `HIDDEN` (320 vs 768) and whatever engine
changes are mechanically required to support it (the K-tiled kernel).

**Primary target H = 768, not 1024.** Already decided by the spike's
own handoff, for reasons this plan inherits rather than re-litigates:
real margin against the ~5 ch/s comfort floor (9.49 vs. 1024's 5.37),
real RDRAM margin (2.29MB vs. 1024's 3.60MB static footprint), and —
per the spike's later Banjo-Kazooie-scope reframing — closer to what a
real game's RDRAM budget could actually afford once it's not the only
thing in the ROM. 1024 is a demo of the kernel's ceiling, not a serious
target. **H=512 added as a second point** (metric 10 below,
DS-II's Pareto-analysis suggestion) — already hardware-characterized
by the spike at a much smaller ch/s cost (20.30 vs 768's 9.49), so
comparing all three points costs one extra training run, not new
engineering, and may reveal H=512 captures most of any real gain at
half the speed penalty.

**Metrics, reported side by side, none sufficient alone:**

1. **Val loss** — necessary, not sufficient (the exact metric that lied
   twice before in this project's own history). Report it, don't stop
   there.
2. **Int8-vs-float top-1 agreement** — must stay ≥0.95 at H=768 too;
   quantization fidelity isn't guaranteed to hold at a different H.
3. **The per-axis conditioning-ablation divergence table** (identity/
   mood/trust/context, `conditioning_divergence_table()`) — run at both
   H, side by side. A bigger model that helps garbling but flattens
   conditioning distinctiveness isn't a clean win. **Also run this
   table once mid-training (~50% through), not just at the end** (DS1)
   — if an axis is already visibly compressing early, that's worth
   knowing before spending the rest of the training budget on a run
   trending toward a flattened-conditioning failure mode.
4. **The generalization check** (held-out occupation/descriptor combos)
   — same methodology, both H, confirms the compositional mechanism
   itself doesn't quietly regress at bigger H.
5. **Per-token perplexity on the golden prompts, bucketed by
   conditioning difficulty** (DS1) — specifically the combos already
   known to be hard (tier-2 escalation, `worried`+`greeting` — the
   documented `"HOMESTELD"` combo). Average loss can improve while the
   hard tail stays bad; this is the cheap way to check whether H=768
   actually helps the specific place garbling happens, not just the
   easy majority of combos dragging the mean down.
6. **A fixed, pre-selected golden set sampled at IDENTICAL seeds on
   both checkpoints** — the actual human-eyeball comparison. Use the
   same Shadewrath/Korrath/Elowen goldens from M11's shipped self-test
   (known, already-documented failure cases like the `TR:1 M:worried
   C:greeting` "HOMESTELD"/"EAVER" line) plus a fresh sample, and
   **literally count invented-non-word occurrences per checkpoint** —
   a concrete, countable proxy for "garbled" instead of a vibe. This is
   the metric that actually answers Luke's ask ("real, tangible quality
   improvements, not just garbage/garble"). Sample this set **two
   ways** (DS1): greedy (temperature 0) and the shipped sampler's real
   temperature — invented words can emerge specifically under sampling
   noise even when greedy top-1 output looks clean, so greedy-only
   would miss exactly the failure mode already documented on hardware.
7. **Cheap corpus-level linguistic stats, both checkpoints** (scoped
   down from DS-II's fuller list): repetition rate, unique-token
   ratio, and average response length on the golden set. All three are
   plain string statistics over output already being generated for
   metric 6 — no new tooling. **Not adopting** DS-II's suggested
   embedding-based semantic similarity to reference responses: this
   project has no embedding-model infrastructure anywhere in its
   stack, and standing one up to score a single experiment is a
   disproportionate build for what it buys here.
8. **Error categorization by failure type** (DS-II): when logging the
   golden-set comparison, tag each garbled/off output as invented
   words, lore hallucination, identity drift, context-axis bleed (the
   already-documented "wrong conditioning axis" pattern from
   `docs/plan.md`), repetition loop, or grammatical break. Cheap
   (it's a label on data metric 6 already collects) and tells us
   *which* failure modes H=768 actually helps, not just whether the
   aggregate count drops.
9. **`eval_shadewrath_long_horizon.py` run against both checkpoints**,
   transcripts placed side by side, **the exact 4-item checklist
   already written into that script's own `print()` output** (tone
   escalation, no cross-tier contradiction, tier-2 gestures at the
   real offer, coherence holds across the session) — cited here
   explicitly so the checklist is understood as already frozen/
   version-controlled (DS3), not something to redefine at review time.
10. **H=512 as a third, cheap comparison point** (DS-II's Pareto-
    analysis suggestion, adopted) — the ktile spike already
    characterized H=512's speed profile on real hardware (2.77x
    speedup, 20.30 ch/s at chunk=256, comfortably inside the ch/s
    comfort floor), so training it on the same M11 corpus costs one
    more training run, not new engineering. Report quality-per-cost
    across all three points (H=320 baseline, H=512, H=768): does
    H=512 capture most of whatever quality gain H=768 shows, at a
    much smaller ch/s cost? If so, that's the actual ship candidate,
    not H=768 by default.

**Pre-registered acceptance bar — decided now, before seeing either
result, so it can't be rationalized after the fact:**

H=768 counts as a genuine win only if **all** of:
- Agreement stays ≥0.95 (no quantization regression). **If it drops
  below 0.95** (DS3 — the plan previously stated the bar but not what
  happens if it's missed): attempt exactly ONE mitigation from the
  ktile spike's own four recorded Q-format strategies, re-measure
  once. If still below 0.95 after that one attempt, that's a clean
  stop and a negative result — not open-ended tuning.
- The divergence table shows every axis (identity/mood/trust/context)
  staying **within 20% of its H=320 value**, not merely "nonzero"
  (DS3 — a slow squeeze toward uniformity that never technically hits
  zero would still be a real conditioning regression the original
  "no collapsing" wording wouldn't have caught).
- The invented-non-word count on the fixed golden set drops
  measurably. **The exact denominator and baseline count get locked in
  as part of shipping M11** (DS3), before this plan's Step 5 begins —
  e.g. if M11's H=320 goldens produce N invented-word occurrences
  across the fixed set, write down that exact number in this doc, and
  the bar becomes "≤ half of N," not a percentage computed after the
  fact from whatever the baseline turns out to be. *(Placeholder until
  M11 ships: [N to be filled in from the shipped M11 golden set].)*
- The long-horizon eval's checklist (the 4 items already frozen in
  `eval_shadewrath_long_horizon.py`, cited above) scores at least as
  well on all 4 items, better on at least 2.

If any of these fail, or the invented-word rate doesn't measurably
improve, **record that as plainly as M9.1's density-structure
experiment or M10's density-fix retrain were recorded** — an honest
negative result, not a reason to quietly drop the comparison from the
writeup. The point of pre-registering the bar is exactly to prevent
"well it's SOME better" from being retroactively called a win.

**Post-experiment decision framework — defined now so a result of any
kind has a clear next step, not a planning vacuum (DS1 + DS3):**

- **Tier 1 — H=768 (or H=512, if it captures most of the gain per the
  Pareto point above) meets the full bar.** Ship it as the new
  default; pursue the ktile spike's own named-but-unclaimed speed
  levers (async CPU/RSP dispatch, double-buffered DMA) as the next
  pass to claw back some of the ch/s cost.
- **Tier 2 — measurable garbling reduction, but short of the full bar**
  (e.g. divergence stays in range and agreement holds, but the
  invented-word drop is real yet under the 50%-of-baseline threshold).
  Don't ship H=768 outright; the more honest move is investing the
  same effort in decoding-side changes at H=320 (top-k/top-p tuning
  against the same invented-word proxy) before paying the ch/s cost
  for a partial win.
- **Tier 3 — no improvement, or a regression on any bar item.** This
  has a specific, named implication given the ~4.6x ch/s cost already
  paid to even find out: it means M11's corpus-structure change (the
  Ravendale-lore bank) was likely the only viable lever available
  without a genuinely different intervention — the next candidates
  become the ktile spike's own async-dispatch/double-buffered-DMA
  levers (attacking speed, not quality, since quality-via-capacity
  would be ruled out) or a deeper look at tokenization/vocabulary size
  and training curriculum, not another capacity bump. Record this
  outcome with the same honesty as M9.1's own negative result, and
  update `docs/plan.md`'s Known follow-ups accordingly rather than
  letting "maybe more capacity would help" quietly persist as an
  untested assumption after it's actually been tested and failed.

## Sequencing

1. **Finish M11 at H=320 first** (in progress as this doc is written)
   — this run's val loss/agreement/divergence numbers become the
   baseline this whole comparison is measured against. Don't start
   this plan's work until that baseline is real, tagged, and shipped.
2. **Get Luke's explicit sign-off** on bumping `core/ngpt.h`'s
   `NGPT_GRU_MAX_HIDDEN` on `main` — a frozen-interface decision, not
   a data-science call.
3. **Integrate the K-tiled kernel design (Design A, per the spike doc)
   into current `main`'s RSP overlay** — real engineering work, not a
   constant swap: current `main` has moved since the spike branched
   (gossip, SaveData, the princess, the whole M10/M11 session this
   doc's own retrain is part of).
4. **Re-validate the int32 overflow analysis against the real trained
   H=768 model's actual bias magnitudes** — not the gibberish model's.
5. **Train H=512 and H=768 on M11's exact shipped corpus**, same seed/
   procedure for both, checking the divergence table at the midpoint
   of each run (metric 3 above) before letting either run to
   completion unattended.
6. **Run the full metric battery above on all three checkpoints**
   (H=320 baseline, H=512, H=768), report against the pre-registered
   bar and the Tier framework, honestly either way.
7. **Hardware-verify** whichever checkpoint(s) clear the bar:
   `SELFTEST PASS`, `RSP ON`, `XCHK PASS`, and the real measured ch/s
   on real hardware (not extrapolated) — same discipline every
   milestone in this project already uses.

## Corpus-structure alternatives, checked against real data before assuming (2026-07-22)

Two more reviewer suggestions (dialogue-situation diversity per
character; auditing per-character turn-length distribution as a
possible driver of garbling) are corpus-quality levers, not capacity
levers — complementary to this plan, not part of the controlled H-only
comparison, so recorded here rather than folded into the experiment
design above. One was checked immediately against the real shipped
corpus rather than left as a guess, since the check was cheap:

**Turn-length audit (real numbers, `trainer/ngpt_trainer/*_corpus.py`,
seed=0):**

| character | n | mean len | stdev |
|---|---|---|---|
| selena | 360 | 86.7 | 34.5 |
| guard (old scheme) | 540 | 63.0 | 22.6 |
| shadewrath | 960 | 95.5 | 42.3 |
| korrath | 480 | 99.9 | 46.2 |
| princess/elowen | 480 | 117.2 | 47.3 |
| cast (9 chars, compositional) | 720 each | 102.8–133.1 | 30.1–49.1 |

**The "shorter turns cause garbling" hypothesis, as literally stated,
isn't supported by this data** — Shadewrath's and Korrath's mean turn
lengths aren't shorter than the compositional cast's, and both are
*longer* than Selena's (the project's most coherent character). What
the numbers do support is the hypothesis already being tested by the
in-progress retrain: Shadewrath/Korrath/Elowen's *effective* training
signal is weaker not because their turns are short, but because —
until this session's Ravendale-lore-bank addition — none of their
content was reinforced by any other character the way the 9
compositional-cast members all share `_OCCUPATION_FLAVOR`/
`_DESCRIPTOR_TICS` banks with each other. Diversity-of-situation
coverage (greetings/conflict/exposition/emotional-shift breadth per
character) is a real, separate, and reasonable lever worth checking
once M11's lore-bank result is in — but the specific turn-length
mechanism proposed didn't survive contact with the actual corpus, and
it's better to say that plainly than let an untested hypothesis linger
just because it sounded plausible.

## What this plan deliberately does not decide

- Whether H=768 ships as the new default if it wins — that's a product
  call (accepting ~4.6x slower text-streaming in exchange for
  measurably less garbling), not something this evaluation plan
  presupposes. The metrics above inform that call; they don't make it.
- The async CPU/RSP dispatch and double-buffered DMA levers the ktile
  spike names as separate, unclaimed speed improvements (its Handoff
  items #6/#7) — orthogonal to the capacity question this plan is
  about, real candidates for a *different* pass at ch/s if H=768 wins
  the quality question but the speed cost turns out to matter more
  than expected.
- Final milestone numbering (M12 vs M14) — outside this plan's scope,
  a docs-organization call Luke is making separately.

## Reviewer suggestions considered and explicitly scoped down (not silently dropped)

Three DS reviews raised several ideas that are genuinely good in an
unconstrained setting but disproportionate to what this experiment can
actually afford. Recording the reasoning rather than quietly ignoring
them, so a future reader doesn't have to wonder whether they were
missed or deliberately deferred:

- **Multiple training seeds (3-5 runs per H) for statistical
  significance.** Real rigor gap, genuinely acknowledged. Not adopted
  as a requirement: this project trains CPU-only on one laptop shared
  with other work, and H=768 alone is already a `3H²`-scaled training
  cost over H=320 — 3-5 seeds at *two or three* H values each would be
  days of sequential compute for a single-person hobby project, not
  the hours this plan's single-seed version costs. **Compromise, if
  time allows once the single-seed result is in**: 2 seeds at
  whichever H value(s) look most promising, not a full grid — enough
  to sanity-check the result isn't pure init-noise, without committing
  to an open-ended compute budget up front. The single-seed limitation
  gets stated plainly in the writeup either way, not hidden.
- **Formal blinded human evaluation with multi-rater inter-rater
  reliability (Cohen's κ).** This project's human-eyeball reviews to
  date have realistically had one evaluator (Luke, sometimes with
  Claude reading alongside). Requiring a rater pool and computing
  agreement statistics assumes infrastructure (multiple independent
  raters) this project doesn't have and standing it up for one
  experiment isn't proportionate. **Adopted in scoped-down form**: if
  a second rater is ever available for this specific comparison, blind
  the model-size identity when presenting transcripts — cheap
  insurance against the reviewer's own expectation biasing the read,
  costs nothing if it's just one rater either way.
- **Embedding-based semantic similarity to reference responses.** Not
  adopted — see the metrics section above; no embedding-model
  infrastructure exists anywhere in this project's stack, and building
  one to score a single experiment is a disproportionate build. The
  cheaper linguistic stats (repetition rate, unique-token ratio,
  length) cover a meaningful chunk of the same signal without new
  tooling.
- **GRU LayerNorm/RMSNorm or embedding-dim increase, A/B'd alongside
  the capacity test.** Both are genuinely interesting alternative
  levers (the DS1 review's framing — normalization sometimes buys more
  coherence per parameter than raw width — is a real, credible
  hypothesis). **Not folded into this experiment**: doing so would
  break the controlled-variable design every reviewer independently
  praised ("controlled variable: H only") by changing the architecture
  and the capacity at the same time, making it impossible to attribute
  a result to either cause cleanly. Recorded here as a real, named
  candidate for a *follow-up* experiment — run only after this one
  reports a result, and only as its own controlled comparison (same
  H=320, LayerNorm on vs. off, nothing else changed), not stacked into
  Step 5 above.
