# Overnight session recap — 2026-07-27/28

Everything below happened on branch `overnight-2026-07-27`, off `main`
at `80103f5`. **Nothing is merged or pushed** — this is a review surface,
not a done deal. `git log --oneline main..overnight-2026-07-27` for the
full commit list (18 commits as of writing).

## TL;DR

- 3 earlier tasks (009/010/011, corpus gates / manifest checker /
  living-NPC-state) already landed on `main` before this session started.
- This session: 2 real tooling bugs found and fixed with TDD
  (`~/bin/opencoder`, `~/bin/qwen-worker`) so the delegation pipeline
  actually works; M13 mechanism 4 run through steps 1-5 with two honest
  mid-course corrections; 7 new host-tested C++ headers; 3 Python test/
  tooling additions; 2 real `docs/plan.md` follow-ups partially closed.
- One training run (M13 baseline, seed A) was still in progress when
  this was written — check `trainer/m13_mechanism4_results/` for
  whether it finished. Seed B and the approved/rejected runs were
  **deliberately not started** — see "What's still open" below.
- Everything is real, tested, and reviewed as it landed (not a queue of
  unverified diffs) — but a second pass from you before merging is
  still the right move, same as any other night's work.

## What shipped, grouped by area

**M13 mechanism 4 (`docs/milestones/m13.md` has the full blow-by-blow)**
- Pre-registered protocol locked *before* any result existed (commit
  `b5bacc5`, earlier in the evening).
- Generation attempt 1 (Mistral, blind): 6/24 passed the structural
  gate. Attempt 2 (vocab-constrained): made it WORSE (1/32) — real
  finding, not a fluke.
- Judge collapsed to uniform scores in a full batch AND an isolated
  6-item batch (Mistral). A calibration test proved why: caught an
  obvious joke, missed a wildly wrong-mood line, dropped an item from
  its own output. Switched judge model to
  `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit` — real spread,
  both violations caught, nothing dropped.
- First real split (K=8): approved 4.25 vs rejected 3.00. **Your
  spot-check standin (mine, clearly labeled as not a substitute for
  yours) found this too weak to trust** — several "rejected" items used
  strong character motifs that arguably beat "approved" ones. Didn't
  spend training compute confirming a contrast that looked shaky before
  any training happened.
- Widened the pool. Three more LLM-generation attempts, all failed
  (voice collapse from batching all characters in one dispatch, then
  two genre-breaks into sci-fi/corporate phrasing even in isolated
  dispatches with different prompting). Fell back to direct authorship
  (30 new lines, natural quality variation, not hand-picked good/bad).
- Second split (K=6): approved 5.00 vs rejected 2.33 — a real gap this
  time. **`trainer/m13_final_split.json` is ready for your actual
  spot-check** (30 seconds, per the protocol's own required step 5).
- Built `trainer/m13_mechanism4_validation.py` (real training script,
  not a spike-and-discard one), kicked off the seed-A baseline run.
  **Seed B, and the approved/rejected runs, were not started** — see
  below for why.

**Two real tooling bugs, both fixed with TDD**
- `~/bin/opencoder`: `active_model()` let `QWEN_MODEL` outrank the
  on-disk state file, so a model-switch request compared itself to
  itself and silently no-op'd — the OLD model kept serving. Fixed
  (STATE file is now ground truth); `opencoder selftest` added,
  red/green confirmed.
- `~/bin/qwen-worker`: two bugs — (1) its mini-YAML parser didn't strip
  quotes from `verification:` commands, so every quoted verify command
  got executed as a literal bogus string; (2) `check_output()` required
  an exact `=== END FILE ===` line, but Qwen2.5-Coder sometimes leaks
  its `<|im_end|>` stop token onto that line with no newline, so
  complete correct output got misdiagnosed as truncated. Both fixed,
  both covered by new `--selftest` regression tests, both confirmed
  red-then-green before being trusted.
- A third infra bug found and fixed live: the overnight watchdog kept
  dying every time a model tier switched, because `opencoder`'s own
  `pkill -f mlx_lm.server` does substring matching against full command
  lines — and the watchdog's own background shell script had the
  literal string "mlx_lm.server" embedded in its source, so it was
  killing itself. Fixed by building the target string from concatenated
  parts instead of a literal.

**New C++ headers (all header-only, host-tested, no libdragon,
`game/src/user/`)**
- `NPCState.h` extended: `Profile` struct + `resolveBeliefId()` (public
  vs private belief, trust-gated); `ReactionRule`/`REACTION_TABLE`/
  `applyEventReaction()` (EventBus event tags -> Relationship deltas,
  tags matched to the real `WorldState::GOSSIP_EVENTS`); memory
  retrieval scoring (`memoryRetrievalScore`/`selectByRelevance` — adds
  the RELEVANCE factor `selectTopMemories` didn't have); gossip
  propagation (`propagateGossip`, confidence degrades 25%/hop, salience
  unchanged); plus edge-case tests for the full 8-way eviction tie and
  sequential `applyDelta` compounding that weren't covered before
  tonight.
- `SpawnSeedSource.h` — closes a real `docs/plan.md` follow-up:
  deterministic (levelId, slotIndex) -> which trained guard seed to
  spawn, so the world doesn't always place the same 4 guards in the
  same order.
- `QuestState.h`, `NpcDisposition.h`, `PlayerReputation.h`,
  `NpcSecret.h`, `WordPicker.h` — grounded in `docs/ideas*.md`, each
  scoped to the safe/testable data-layer slice, with the parts that'd
  need real corpus/training work explicitly called out as NOT done in
  each file's own header comment.
- Host test count: 15 -> 21 executables, all green
  (`ctest --test-dir build`).

**Python / trainer**
- `corpus_gates.py`: 7 new edge-case tests, delegated to Qwen2.5-Coder-
  7B via `.ai/tasks/012-corpus-gates-hardening.md`. One real defect
  caught in review (a test that built its own vocab from the fragment
  under test, so it verified nothing) — fixed.
- `manifest_validate.py`: edge-case tests for missing schema keys,
  missing character fields, empty manifests.
- `validate_manifest_cli.py` — new, the CLI surface M14's planned
  manifest-update skill needs.
- `trainer/ngpt_trainer/model.py`: `checkpoint_path` added to
  `train_corpus_conditioned_attr`/`qat_finetune_attr` — writes
  best-so-far state to disk on every improving epoch, not just at the
  final QAT save. Closes (partially — see `docs/plan.md`) the M12.5
  near-miss follow-up. Verified on a fast toy run
  (`trainer/tests/test_checkpointing.py`), not production scale.
- `make_m11_1_blob.py`, `make_m10_blob.py`, `make_m9_blob.py`,
  `make_m8_blob.py`, `make_m7_blob.py`: back-ported the
  `max_len=MAX_GOLDEN_LEN` fix M12.1 found (4-5 call sites each) — a
  latent bug that silently truncates golden generations instead of
  correctly flagging them as degenerate. Verified (not assumed) that
  `make_m4_blob.py`/`make_m9_rsp_spike_blob.py`/`make_m3_blob.py` don't
  need it — full sweep, `docs/plan.md`'s follow-up is fully Closed now.

**Docs**
- `docs/plan.md` Known Follow-ups: closed the spawn-seed-source item,
  added honest partial-progress notes to the other two (not overclaimed
  as fully done).
- `docs/milestones/m13.md`: full mechanism-4 execution record.

## What's still open (don't assume any of this is done)

1. **M13 baseline seed-A training run** — was still running (MPS) when
   this was written. Check `trainer/m13_mechanism4_results/
   baseline_seedA.json` — if it doesn't exist yet, it's still going or
   it crashed; check `/tmp/m13_baseline_seedA.log` and the watchdog log
   (`watchdog_overnight.log` in the scratchpad) for what happened.
2. **Seed-B baseline run** — not started. Needed for the noise-floor
   measurement the pre-registered bar depends on.
3. **The approved/rejected training runs** — not started, on purpose.
   The protocol requires your spot-check of `trainer/m13_final_split.json`
   first. That's steps 6-7 of `docs/milestones/m13.md`'s mechanism-4
   protocol, still fully outstanding.
4. **`clang-format` pass** — blocked on `brew install clang-format` (not
   installed) and your go-ahead to install it. Didn't install a new
   tool unilaterally overnight.
5. ~~Older `make_mN_blob.py` scripts still have the `max_len` bug~~ —
   **UPDATE: finished the sweep later in the session.**
   `make_m10_blob.py`/`m9`/`m8`/`m7` all fixed too; verified (not
   assumed) that `make_m4_blob.py` and `make_m9_rsp_spike_blob.py`
   genuinely don't need it and `make_m3_blob.py` has no relevant calls.
   `docs/plan.md` now shows this item fully Closed.
6. **`train_corpus_conditioned`/`qat_finetune`** (the plain, non-`_attr`
   pair `make_m12_1_blob.py`'s actual SHIPPED path uses) and the
   `_film` variant (where the original M12.5 incident happened) still
   have no mid-run checkpointing — only the `_attr` variant used by
   tonight's own M13 work was covered.

## Safety notes, since this ran fully unattended

- A kill-capable watchdog ran the whole session: sustained (2+
  consecutive checks, not a single blip) free memory below 8% would
  have killed `mlx_lm.server` automatically to free wired GPU memory.
  It never fired for real tonight — memory stayed in the 70-85% free
  range throughout everything after the M13 mechanism-4 investigation
  phase (which had its own earlier, separate near-OOM incidents, already
  covered by the time this branch's work started).
- MPS training and `opencoder` were never run concurrently (the
  project's own contention rule) — `opencoder` was stopped before the
  M13 training script started and was not restarted while it ran.
- Every commit was checked for `/Users/<name>` path leaks before
  landing (this repo is public).
