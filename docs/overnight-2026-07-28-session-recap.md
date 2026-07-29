# Overnight session recap — 2026-07-28/29

Continues directly from `docs/overnight-2026-07-27-session-recap.md`.
Still branch `overnight-2026-07-27`, off `main`. **Nothing is merged or
pushed** — this is a review surface, not a done deal. `git log --oneline
main..overnight-2026-07-27` for the full commit list.

## TL;DR

- **M13 mechanism 4 is fully closed: FAIL.** Both the judge-approved
  (2.4375 inv/line) and judge-rejected (2.75) fragment sets made
  guard+korrath coherence *worse* than an untouched baseline
  (1.50/1.81) — the underlying premise ("judge-approved content
  improves real training outcomes") did not hold up when actually
  tested. A real, unexpected negative result, written up in full in
  `docs/milestones/m13.md` with plausible-cause analysis, not just a
  number.
- **The checkpointing Known Follow-up is now fully closed.** All three
  `train_corpus_conditioned_*`/`qat_finetune_*` pairs (attr, plain,
  film) have `checkpoint_path` — the plain pair (the one
  `make_m12_1_blob.py`'s actual shipped path uses) and the film pair
  (where the original M12.5 near-OOM incident happened) were the two
  gaps left from the prior session, both closed tonight with TDD
  red-then-green.
- **Five new NPC-engine state headers** (`TimeOfDay.h`,
  `VisitFrequency.h`, `NpcCondition.h`, `LocationAtmosphere.h`,
  `PlayerAppearance.h`), each closing a specific gap named in
  `docs/ideas-m7-living-npcs.md` Part 4's "more candidate conditioning
  variables" list. Same "pure game-state layer, corpus/conditioning
  wiring is separate work" discipline as every other NPCState-adjacent
  header. Host test count: 21 → 26 executables, all green.
- **M14 (portability) got real, honest progress — not closed.** The
  meta-schema/porting-guide DoD items are done (an explicit
  supplies-vs-reuses checklist added to the existing
  `docs/08-manifest-schema.md`, plus a real second-project manifest,
  `manifests/scifi_freighter.json`, validated clean through the
  unmodified tooling). The full-retrain-vs-continued-training design
  decision was made and recorded (full retrain, always). The
  portability *proof* ran for real: the mechanism transferred cleanly
  (manifest validates, corpus generator reuses the schema unmodified,
  training ran through the unmodified pipeline) — but the coherence
  numbers (guard 4.375 / engineer 5.125 inv/line, both from a
  deliberately thin, non-production corpus) don't demonstrate a
  *coherent* result, and the doc says so plainly. Manifest-update
  skill, capacity split-trigger, and capacity-monitoring metric are
  still real, unstarted work — **no tag, no README row, M14 is not
  being claimed as done.**
- **`.clang-format` added, but a repo-wide reformat was deliberately
  NOT run.** Dry-run diffs (including against files written earlier
  this same session) showed this codebase's actual style depends on
  hand judgment — when to collapse a short function to one line, hand-
  aligned enum/struct columns, hand-compacted multi-statement lines in
  `core/`'s dense byte-offset parsing code — that no fixed clang-format
  ruleset reproduces without a large, real diff across bit-exactness-
  critical code. Documented as a real finding in the commit, not a
  skipped task.
- **Four new ideas added to `docs/ideas.md`**, each grounded in
  something actually measured or built tonight (seed ensembling and a
  QAT-variance diagnosis from M13's noise floor; a conditions-stack
  demo idea from the new headers; a genre-starter-pack idea from the
  M14 proof) — not speculative brainstorming.
- **ROM regression check done for real** (a prior instance of this
  session marked it complete without actually doing it — caught and
  fixed): clean rebuild of the shipped M12.1 ROM, booted in Ares,
  SELFTEST PASS / RSP ON / XCHK PASS / 44 ch/s, screenshot filed in
  `talk/` with a dated caption. `talk/narrative.md` also got a new,
  plain-language section on M13's noise-floor finding — real research
  material, not just a boot screenshot.
- Two tooling regression checks (`opencoder --selftest`,
  `qwen-worker --selftest`) both fully green — last session's fixes
  are still solid.
- **Local models were essentially not used tonight**, and that's
  correct, not a shortcut: M13's and M14's training runs occupied the
  GPU almost continuously (MPS), and this project's own hard rule is
  never to run `opencoder`/`qwen-worker` (Metal-based local inference)
  alongside MPS training — the exact OOM-crash scenario that rule
  exists to prevent. All work was done directly by Claude during the
  GPU-busy windows (small, high-context C++ headers, docs, config) and
  the `opencoder`/`qwen-worker` selftest checks ran during the one
  brief window the GPU was actually free.

## What shipped, grouped by area

**M13 mechanism 4 — full close-out**
- Approved-arm training run: guard+korrath 2.4375 inv/line (float
  0.1272, QAT 0.1291) — worse than both baselines.
- Rejected-arm training run: guard+korrath 2.75 inv/line (float 0.1290,
  QAT 0.2095) — worse still.
- Pre-registered bar applied exactly as specified: `noise_floor =
  |1.50 - 1.8125| = 0.3125`; condition 1 (`approved <= baseline +
  noise_floor` = 1.8125) fails by 0.625, more than the noise floor
  itself, not a marginal miss; condition 2 (`gap > noise_floor`) lands
  *exactly* on the boundary (0.3125 == 0.3125) but doesn't rescue the
  result since PASS needs both conditions. **Verdict: FAIL**, decided
  cleanly by condition 1.
- Full writeup in `docs/milestones/m13.md`: the exact numbers, the bar
  computation shown explicitly, three plausible (untested) causes for
  the regression, and an explicit note that the judge's *relative*
  ranking direction was correct (approved beat rejected) even though
  the *absolute* premise failed — those are different claims and only
  one of them held up.
- One methodological fact recorded plainly: the protocol's own step 5
  (human spot-check before spending training compute) was explicitly
  waived by Luke this session ("do the freaking trainings too... you
  can do it"), not silently skipped — affects how much to trust the
  approved/rejected split's face validity.
- `docs/plan.md`'s open corpus-quality follow-up (2026-07-23, still not
  started) cross-referenced with this result as a related but non-
  substitutive data point — M13 tested a narrow K=6 fragment injection,
  not the fixed-H controlled corpus-quality comparison that follow-up
  actually calls for.

**Checkpointing sweep — Known Follow-up now fully closed**
- `train_corpus_conditioned`/`qat_finetune` (plain pair,
  `make_m12_1_blob.py`'s real shipped path): `checkpoint_path` added,
  TDD red-then-green (`trainer/tests/test_checkpointing_plain.py`).
- `train_corpus_conditioned_film`/`qat_finetune_film`: same, TDD
  red-then-green (`trainer/tests/test_checkpointing_film.py`) — this is
  the variant where the *original* M12.5 near-OOM incident happened,
  so closing it matters more than the other two.
- `docs/plan.md` moved the item from Open (partial) to Closed.
- Full `trainer/` pytest run: 257 passed, 3 deselected, zero
  regressions from touching `model.py`'s shared training loops.

**Five new NPC-engine headers (`game/src/user/`, all host-tested)**
- `TimeOfDay.h` — closes a documented gap: `docs/ideas-m7-living-npcs.md`
  Part 3 claimed time-of-day was already part of "World Context"; Part
  4 corrected that (no `T:` field exists). Tick-counter → 4-period
  bucketing.
- `VisitFrequency.h` — cadence (gap-since-last-visit), explicitly
  distinct from `Relationship`'s familiarity (depth, not cadence).
  Documents its own recency-proxy limitation plainly rather than
  overclaiming true rolling-regularity tracking.
- `NpcCondition.h` — the NPC's own current state (tired/injured/drunk/
  mid-task), a bitmask so conditions can co-occur, plus a priority-
  ordered `dominantFlag()` for a future single-token conditioning pick.
- `LocationAtmosphere.h` — place-level mood (festive/tense/abandoned),
  distinct from NPC-level state, decays back to neutral over time.
- `PlayerAppearance.h` — visible gear tier, independent of
  `PlayerReputation` (a well-dressed thief still gets the polite
  greeting).
- Plus edge-case hardening on the five headers from the *previous*
  session (`QuestState.h`, `NpcDisposition.h`, `PlayerReputation.h`,
  `NpcSecret.h`, `WordPicker.h`) — each header's own comment had
  flagged specific untested edge cases; all closed. Host test count:
  15 (start of last session) → 21 (end of last session) → 26 (now).

**M14 — real, partial, honestly-scoped progress**
- `docs/08-manifest-schema.md` (the meta-schema spec, already existed
  from M8/M9/M11.1) got an explicit "what a new project supplies vs.
  reuses" porting checklist, plus a fixed stale self-reference left
  over from the milestone's three renumberings.
- `manifests/scifi_freighter.json` — a real second-project manifest
  (sci-fi setting, one `engineer` character, zero shared vocabulary
  with the dungeon crawler's occupations/context), validated clean
  through the *exact same, unmodified* `validate_manifest_cli.py`.
  Pinned as a permanent regression test
  (`test_no_undeclared_or_orphaned_values_in_scifi_freighter_manifest`),
  not a one-time manual check.
- `trainer/ngpt_trainer/scifi_engineer_corpus.py` — the portability-
  proof archetype, mirroring `guard_corpus.py`'s template-grammar shape
  exactly through the unmodified `npc_service.prompt_fields()` schema.
  9 pytest cases.
- The full-retrain-vs-continued-training design decision (blocked in
  the milestone doc pending a lead call) made and recorded: full
  retrain, always, for goldens-never-drift reproducibility — continued
  training deferred as a future optimization pending real cost data.
- `trainer/m14_portability_proof.py` — an isolated training experiment
  (own checkpoint dir, never touching the shipped `model.bin`/manifest)
  combining guard (existing baseline) + the new engineer archetype
  through the plain, non-experimental training pipeline. Smoke-tested
  on CPU before the real run. Result: guard 4.375 / engineer 5.125
  inv/line under deliberately thin conditions — the mechanism-transfer
  claim is demonstrated (real, falsifiable, and true); the coherence
  claim was never in scope for this lightweight version and stays
  explicitly open. Both claims kept separate in the writeup rather than
  conflated into an overclaimed "portability proven" headline.
- DoD checklist updated item-by-item with honest checkmarks — 4 items
  closed, the rest (manifest-update skill's real implementation,
  capacity split-trigger, capacity-monitoring metric) explicitly left
  open. No tag, no README status row.

**Tooling / hygiene**
- `.clang-format` config added (derived from this codebase's actual
  observed conventions, not a stock preset), but a repo-wide reformat
  deliberately not run — see TL;DR above for why. This resolves the
  task that was blocked all of last session on the install go-ahead.
- `opencoder --selftest` (4/4) and `qwen-worker --selftest` (18/18)
  both green — confirms last session's two real tooling fixes are
  still solid.
- Clean rebuild + Ares boot of the shipped M12.1 ROM: SELFTEST PASS,
  `64GPT V1.9 - SELENA`, RSP ON, XCHK PASS, 44 ch/s. Confirms a night
  of new (currently unwired) `game/src/user/` headers and trainer
  changes didn't quietly break the build or the on-device proof.
  Screenshot filed in `talk/` (git-ignored, machine-local) with a
  dated caption.
- `talk/narrative.md` got a new plain-language section on M13's noise-
  floor finding, written for a general audience, not just engineers.
- `docs/ideas.md`: four new ideas, each explicitly grounded in
  something measured or built tonight, each marked "not costed/tried."

## What's still open (don't assume any of this is done)

1. **M14's manifest-update skill** — only the validate step exists and
   was exercised for real; the full regenerate→retrain→verify→
   ship-or-refuse pipeline as an actual callable skill is real,
   unstarted work.
2. **M14's capacity split-trigger and capacity-monitoring metric** —
   both need M11's real full-cast retrain data, neither started.
3. **M13's own "plausible causes, not disentangled tonight" list** —
   whether K=6 is simply too few fragments, whether the direct-
   authorship pool itself (untested against a no-new-content control)
   is the confound, and the QAT-phase instability question first
   flagged in the prior session — all real, distinct, untried follow-
   up experiments, not answered by tonight's FAIL verdict alone.
4. **The M14 portability proof's coherence claim** — genuinely open,
   not just unstarted: would the sci-fi engineer archetype read as
   coherent at real production corpus density? Needs its own real
   corpus investment and retrain, explicitly out of scope tonight.
5. A stray note: a mid-session mistake was caught and corrected —
   task #46 (ROM rebuild + Ares boot) was briefly marked complete by
   confusing it with the unrelated C++ `ctest` host-test run, then
   caught (by the user asking why device tests weren't showing up) and
   actually done for real. Flagging this here in case anything else in
   this recap deserves a second look — the correction was thorough,
   but a self-reported "I caught my own mistake" is exactly the kind
   of claim worth independently spot-checking, not taking on faith.

## Safety notes, since this ran fully unattended

- A memory watchdog (`scratchpad/night-monitor.sh`) ran the whole
  session: 2+ consecutive sub-8%-free readings would auto-kill
  `mlx_lm.server` only — explicit, scoped-down authorization from Luke
  this session, given he was asleep and unavailable for the
  case-by-case "warn and wait" his standing rule normally requires. It
  never touched the training job, and it never fired for real — memory
  stayed healthy all night, confirmed via `night-monitor.log`.
- A recurring cron check-in (~every 10 minutes) kept the session from
  going idle during the long GPU-bound training waits, alongside
  targeted `run_in_background` waits on each specific training
  process's exit.
- MPS training and `opencoder`/`qwen-worker` were never run
  concurrently — verified before every local-model dispatch, and in
  practice `opencoder` barely ran at all tonight since the GPU was
  almost continuously occupied by M13/M14 training.
- Every commit was checked for `/Users/<name>` path or email leaks
  before landing — worth being precise about what actually caught the
  one real leak that came up: this session's own `grep`-then-`&&
  git commit` check pattern doesn't actually gate anything (`echo`
  always exits 0, so the `&&` runs regardless of what `grep` found —
  a real flaw in the pattern used all session, not just this once).
  The repo's own pre-commit hook is what actually blocked the commit
  (a raw training log's "wrote `<path>`" line staged for commit);
  fixed by not committing raw `.log` files at all (only the clean
  `.json` results), and `.gitignore` updated so this doesn't recur.
  Every other commit tonight likely relied on the same hook as the
  real safety net, not the cosmetic grep check — worth Luke knowing
  the check pattern used tonight wasn't as protective as it looked.
- Nothing was pushed. Everything above is staged/committed locally on
  `overnight-2026-07-27`, waiting for review.
