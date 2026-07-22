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

**Target H = 768, not 1024.** Already decided by the spike's own
handoff, for reasons this plan inherits rather than re-litigates: real
margin against the ~5 ch/s comfort floor (9.49 vs. 1024's 5.37), real
RDRAM margin (2.29MB vs. 1024's 3.60MB static footprint), and — per the
spike's later Banjo-Kazooie-scope reframing — closer to what a real
game's RDRAM budget could actually afford once it's not the only thing
in the ROM. 1024 is a demo of the kernel's ceiling, not a serious
target.

**Metrics, reported side by side, none sufficient alone:**

1. **Val loss** — necessary, not sufficient (the exact metric that lied
   twice before in this project's own history). Report it, don't stop
   there.
2. **Int8-vs-float top-1 agreement** — must stay ≥0.95 at H=768 too;
   quantization fidelity isn't guaranteed to hold at a different H.
3. **The per-axis conditioning-ablation divergence table** (identity/
   mood/trust/context, `conditioning_divergence_table()`) — run at both
   H, side by side. A bigger model that helps garbling but flattens
   conditioning distinctiveness isn't a clean win.
4. **The generalization check** (held-out occupation/descriptor combos)
   — same methodology, both H, confirms the compositional mechanism
   itself doesn't quietly regress at bigger H.
5. **A fixed, pre-selected golden set sampled at IDENTICAL seeds on
   both checkpoints** — the actual human-eyeball comparison. Use the
   same Shadewrath/Korrath/Elowen goldens from M11's shipped self-test
   (known, already-documented failure cases like the `TR:1 M:worried
   C:greeting` "HOMESTELD"/"EAVER" line) plus a fresh sample, and
   **literally count invented-non-word occurrences per checkpoint** —
   a concrete, countable proxy for "garbled" instead of a vibe. This is
   the metric that actually answers Luke's ask ("real, tangible quality
   improvements, not just garbage/garble").
6. **`eval_shadewrath_long_horizon.py` run against both checkpoints**,
   transcripts placed side by side, the same 4-item human checklist
   filled in for each. Does tone escalation, cross-tier bleed, and the
   tier-2 payoff line actually read better at H=768, not just
   differently?

**Pre-registered acceptance bar — decided now, before seeing either
result, so it can't be rationalized after the fact:**

H=768 counts as a genuine win only if **all** of:
- Agreement stays ≥0.95 (no quantization regression)
- The divergence table shows no axis collapsing toward zero relative to
  H=320 (conditioning distinctiveness preserved)
- The invented-non-word count on the fixed golden set drops
  measurably (a specific number, not "seems better") — proposed
  threshold: at least 50% fewer invented-word occurrences across the
  same fixed golden set, since a marginal improvement wouldn't justify
  a ~4.6x ch/s cost on its own
- The long-horizon eval's checklist scores at least as well on all 4
  items, better on at least 2

If any of these fail, or the invented-word rate doesn't measurably
improve, **record that as plainly as M9.1's density-structure
experiment or M10's density-fix retrain were recorded** — an honest
negative result, not a reason to quietly drop the comparison from the
writeup. The point of pre-registering the bar is exactly to prevent
"well it's SOME better" from being retroactively called a win.

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
5. **Train H=768 on M11's exact shipped corpus**, same seed/procedure.
6. **Run the full metric battery above**, report against the
   pre-registered bar, honestly either way.
7. **Hardware-verify**: `SELFTEST PASS`, `RSP ON`, `XCHK PASS`, and the
   real measured ch/s on real hardware (not extrapolated) — same
   discipline every milestone in this project already uses.

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
- Final milestone numbering (M12 vs M13) — outside this plan's scope,
  a docs-organization call Luke is making separately.
