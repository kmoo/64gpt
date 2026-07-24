# The coherence rescue plan — a data-science diagnosis of the garbling problem

**Status: ACCEPTED as M12.1 (2026-07-23), written after M12's honest
negative.** Milestone doc: `docs/milestones/m12.1.md`. M13 = portability
is unaffected.

## Executive summary

The project has now spent three milestones (M10, M11.1, M12) testing
coherence levers — more pairs, shared structural content, 10x more
capacity — and all three came back negative. That is not bad luck. **All
three levers targeted the float model's ability to learn the corpus, and
the float model was never the main problem.** A controlled decoding
experiment on M12's own cached H=1024 model (methodology and full
numbers below) decomposes the garbling into three multiplicative causes,
none of which is capacity:

| cause | invented words (12-prompt probe) | share of the damage |
|---|---|---|
| float model, greedy decode | 5 | baseline residual (data imbalance + long-generation drift) |
| **+ int8 quantization**, greedy | **17** | **~3.4x multiplier — the largest single cause, never previously suspected** |
| + shipped sampler (top-k=5, T=0.67) | 22 | another ~1.3x, plus per-boot variance |

And a fourth, *meta*-cause explains why three milestones of honest work
never caught this: **every selection and acceptance metric in the
pipeline is blind to all three causes.** Val loss is teacher-forced,
prefix-masked, and float — it never sees free-running generation, never
sees the quantized weights, and never sees the sampler. A model can (and
did) hit val loss 0.10 while its shipped int8 sampled output garbles
every third line.

The fix is not a fourth corpus lever or an architecture swap. It is:
train the model you actually ship (quantization-aware), decode with the
model's confidence instead of against it (integer min-p), make invented
words *structurally impossible* (a lexicon-trie decode guard — the
headline idea, zero invented words by construction), rebalance the 75:1
per-character data skew, and select checkpoints on a metric that can
actually see the failure. Every piece respects the frozen streaming API,
the integer-only `core/` rule, and the blob format's model-type seam.
And it means **going back to H=320** — capacity is exonerated, and 44
ch/s beats 5 ch/s.

---

## The experiment that localizes the fault

Everything below is reproducible from M12's own artifacts: the cached
trained model (`trainer/.m12_model.pt`, git-ignored but present on the
training machine), the M11.1-identical corpus generators, and
`ngpt_trainer/ref_impl.py`'s bit-exact integer path. A 12-prompt probe
(4 Selena, 1 guard, 2 Shadewrath, 2 Korrath, 1 Elowen, 2 town cast —
same construction as the curated goldens, seed `0xC0FFEE`,
`max_len=300`) was decoded four ways through the quantized model plus
once through the float model. Invented-word counts use
`make_m12_blob.py`'s own `invented_word_count()` against the full corpus
vocabulary. "Non-argmax draws" counts generation steps where the sampler
picked something other than the top-1 logit.

| config | decode | invented words | non-argmax draws |
|---|---|---|---|
| float model | greedy | **5** | 0 |
| int8 model | greedy | **17** | 0 |
| int8 model (shipped) | top-k=5, T=0.67 | **22** | 144 |
| int8 model | top-k=5, T=0.40 | **8** | 54 |
| int8 model | top-k=5, T=0.67 + integer min-p gate | **13** | 76 |

Per-character invented words under greedy (float / int8), against
training-set size:

| character | training pairs | float greedy | int8 greedy |
|---|---|---|---|
| Selena | 36,000 | 2 | 3 |
| guard archetype | 4,320 | 0 | 0 |
| town cast (9 chars) | 6,480 | 0 | 2 |
| Shadewrath | 2,880 | 0 | 4 |
| Korrath | 1,440 | 2 | 5 |
| Elowen | 480 | 1 | 3 |

Three facts jump out of these tables, and each one is a diagnosis:

**1. Quantization triples the garbling (5 → 17) with zero sampling
involved.** Same weights, same prompts, same greedy argmax rule — the
only difference is int8 rounding. This is the cause nobody had on the
suspect list: `docs/plan.md`'s standing hypothesis mentions
"architecture, training procedure, decoding strategy, or data/task
mismatch," and the M12 postmortem lists the same four. Quantization
appears in the pipeline only as an *acceptance gate* (top-1 agreement ≥
0.95), and that gate is exactly the problem: **0.95–0.97 per-character
agreement is a catastrophic number for an autoregressive character
model, not a passing one.** A 3% chance of flipping the argmax *per
character* means a 100-character line expects ~3 flips; a flip mid-word
coins a non-word ("HES WASTED IT FORST"), and — worse — pushes the
hidden state into territory the model never saw in training, from which
the next errors compound. The gate was calibrated when it was introduced
(M4-era models had tiny corpora the int model reproduced exactly); it
silently stopped meaning "quantization is safe" as the task got harder,
and nothing ever re-examined it. The direct evidence of compounding:
side-by-side greedy transcripts show float and int8 following *the same
sentence* until one character flips, after which the int8 line degrades
progressively while the float line stays clean — e.g. Shadewrath's
float "THE DUNGEON DOESN'T CARE WHO YOU ARE. YOUR BLOODLINE IS SHOWING."
vs int8 "THE DUNGEON DOESN'T CARE WHO YOU ARE. YOU FIGHT LIKE SOMEONE
WHO HES WASTED IT FORST…".

**2. The garbling that remains is rationed by training-pair count, not
capacity.** Under greedy decode the well-fed characters (Selena, guard,
town cast: 4,320–36,000 pairs) are nearly clean; the starved ones
(Korrath 1,440, Elowen 480) garble even in float. The corpus skew is
75:1 Selena:Elowen. This is the same gradient M10 already observed for
Shadewrath ("960 pairs vs Selena's 36,000 — least coherent of any
character in the boot") — the observation was correct, but the remedy
applied (raise pair counts a bit, add shared structural content) never
came close to closing the ratio. Note this is *pairs per character*,
not corpus MB: the shared model's capacity follows the gradient signal,
and 75x more of the gradient is Selena.

**3. The sampler amplifies whatever noise exists, and the amplification
is controllable.** Unconditional top-k=5 at T=0.67 forced 144 off-argmax
choices across 12 lines. Each one is fine at a genuine branch point
(clause boundaries — that's where the corpus's authored variety lives)
and destructive mid-word (that's where invented words are coined). Both
mitigations tested cut invented words roughly in half; the min-p gate
(keep only candidates whose probability is ≥ 25% of the top candidate's)
does it *without lowering temperature*, i.e. without flattening the
character-voice variety the sampler exists to provide. Low temperature
(T=0.40) cut invented words further in this probe but visibly collapses
toward the same few openers per prompt — it trades away exactly the
thing top-k sampling was added for in M4.

One more observation from the transcripts, worth naming because it
compounds with everything above: **when the model does derail, it also
misses EOS** — the derailed Korrath/Elowen lines run to the 300-char cap
in a fragment-splicing loop, while clean lines stop at natural corpus
lengths. Off-manifold states don't just spell badly, they lose the
end-of-line signal too. (This is the same mechanism that surfaced M12's
256-char golden-truncation bug: H=1024 rambled *because* it derailed
more, not because longer output is natural to it.) Fixing causes 1–3
fixes EOS reliability as a side effect; no separate mechanism needed.

### Why val loss stayed blind through all of this

`train_corpus_conditioned` early-stops on teacher-forced, prefix-masked,
*float* val loss, and the acceptance pipeline gates on the same number
plus the 0.95 agreement floor. Teacher forcing means every prediction is
made from a *correct* prefix — the model is never asked to survive its
own mistakes, so the metric literally cannot observe the compounding
failure mode that dominates real generation (the textbook name is
*exposure bias*). Prefix masking means the score covers only response
bodies whose fragments (OPENER/BODY/CLOSER template banks) are shared
across train and val combos — val loss 0.10 nats/char is perplexity
1.11, i.e. near-memorization of fragments the "held-out" combos share
with training. And float means the quantization damage (the 3.4x
multiplier) is invisible by construction. **M12's "worse val loss at
H=1024" is therefore only weak evidence about text quality in either
direction — the metric can't see the thing being judged.** This is also
why per-axis divergence stayed healthy while text garbled: divergence
measures *conditioning*, and conditioning genuinely works; it was never
a coherence metric.

---

## The plan

Five fixes, one per cause, ordered by leverage-per-effort. Each is
independently shippable and independently measurable (walking-skeleton
discipline: every phase ends in a booted ROM with a metric that moved).
All of them together are "the dream model": H=320 speed, zero invented
words guaranteed, authored variety intact, thin characters coherent.

### Fix 0 — Instrument before operating: a coherence gate that can see

*Cause addressed: metric blindness. Effort: small, trainer-only.*

Add to the trainer a **shipped-config coherence probe**: generate N≈50
lines from the *quantized* model with the *shipped* sampler settings
across all characters, and score (a) invented-word rate per character,
(b) EOS-before-cap rate, (c) verbatim-corpus-line rate (as a
memorization vs. recombination indicator). Then:

- **Early-stop and checkpoint-select on `val_loss` + this probe**, not
  val loss alone. (Cheapest version: keep val-loss early stopping but
  evaluate the probe on the best-3 checkpoints and ship the probe
  winner.)
- **Raise the agreement gate** from 0.95 to ≥ 0.995 on completion-only
  top-1 agreement — the level where per-line flip expectation drops
  below ~0.5. This gate will *fail* against today's quantizer; that is
  the point. Fix 1 is what makes it passable.
- Report per-character, not corpus-wide averages — the per-character
  table above is the disaggregation that finally made the data-imbalance
  signal legible.

This fix has no ROM impact at all, and it converts every later fix from
"vibes" to a number. It is deliberately first: had it existed, M10–M12
would have cost one milestone, not three.

### Fix 1 — Quantization-aware training (the 3.4x lever)

*Cause addressed: int8 rounding flips. Effort: moderate, trainer-only.
No blob-format, core/, or ROM changes whatsoever.*

Today's quantizer is one-shot post-training rounding onto a per-tensor
power-of-2-scale int8 grid — the crudest scheme that exists, applied to
a model that was never told the grid exists. The fix is standard
**quantization-aware fine-tuning (QAT)**: after float convergence, run a
short fine-tune phase in which the forward pass *fake-quantizes* the
weights (round each tensor to its exact `quantize()` grid — same
`pow2_shift` k, same int8 clamp — then de-quantize back to float) while
the backward pass updates the underlying float weights via the
straight-through estimator (`w_q = w + (fake_quant(w) - w).detach()` —
~5 lines of PyTorch). The optimizer then converges to a weight
configuration whose *rounded* version behaves well, instead of hoping
the rounded version of a float optimum behaves. This is precisely
"train the model you ship."

- Scope note: weight rounding is the dominant error term; the int16 Q14
  activation grid and LUT nonlinearities are much finer and can stay
  unsimulated in training unless the probe says otherwise. Measure
  first, extend only if needed. *Literature check (2026-07-23): the RNN
  QAT literature (4-bit LSTM ASR, arXiv 2108.12074; QS4D state-space
  QAT, arXiv 2507.06079) confirms recurrent weights benefit most from
  QAT over post-training quantization — but those setups also simulate
  per-timestep activation quantization, because hidden-state error
  compounds across steps. Our activations are int16 Q14, ~64x finer
  than their int8, which is the defense for skipping it — the named
  escalation path if the probe still shows an int8-vs-float gap after
  weight-only QAT is to add fake-quant of h (int16 Q14 grid) and the
  LUT nonlinearities (256-entry table lookup applied in-graph) to the
  QAT forward, per-timestep, exactly as that literature does.*
- Fallback if QAT fine-tuning proves fiddly: **weight-noise training**
  (add uniform ±half-LSB noise to weights each forward pass) gets a
  large fraction of the robustness for trivial code.
- Acceptance: the Fix-0 agreement gate at ≥ 0.995, and int8-greedy
  invented words landing at (not 3.4x above) float-greedy's count on the
  probe.
- **This is also the fix that must land before any future capacity or
  corpus experiment is interpretable** — M10/M11.1/M12 all measured
  float-side levers through an int8 bottleneck that was drowning them.

### Fix 2 — Rebalance the cast (the float-side residual)

*Cause addressed: 75:1 data skew. Effort: small, corpus-config-only.*

Bring every character within ~4:1 of the best-fed character in
*effective pairs per character* — by raising `PER_COMBO` for the thin
characters (Elowen 4 → ≥48, Korrath 12 → ≥48, Shadewrath 24 → ≥48), by
duplicating thin-character pairs into the training list (cheap
oversampling — identical gradient effect without authoring new
templates), or per-sequence loss weights. Selena does not need 36,000
pairs to hold her voice; the skew is an artifact of her being first, not
a design decision anyone defended. Acceptance: per-character
float-greedy invented-word rates within noise of each other on the
Fix-0 probe. (This is M10's own Shadewrath finding, finally applied at
the ratio scale the gradient actually sees.)

### Fix 3 — Integer min-p sampling (decode with confidence, not against it)

*Cause addressed: sampler amplification. Effort: small; ~10 lines in
`core/`, mirrored in `ref_impl.py`; golden regeneration.*

Keep top-k=5 and T=0.67 exactly as shipped, but after computing the
exp2 sampling weights, drop candidates whose weight is below
`top_weight >> 2` (i.e. probability < 25% of the best candidate — the
published "min-p" rule, in pure integer form; the threshold shift is one
new blob/header constant so it stays tunable without a core change).
Effect, mechanically: mid-word, a well-trained model is near-certain of
the next character, so the gate excludes everything but the right
continuation and decoding is locally greedy — invented words lose their
entry point. At clause boundaries, several continuations genuinely carry
comparable probability, all survive the gate, and the authored variety
still expresses. It is temperature that adapts itself per step, using
information the model already emits for free.

Measured on the probe: 22 → 13 invented words at *unchanged*
temperature, and after Fixes 1–2 sharpen the model's confidence the
gate gets strictly more effective (sharper peaks → tighter gating
exactly where it matters). Bit-exactness discipline is unchanged: the
Python reference gains the same integer gate, goldens regenerate, the
XCHK cross-check proves the C port.

*Literature check (2026-07-23): this gate is exactly published min-p
sampling (arXiv 2407.01082, "Turning Up the Heat" — shown to beat
top-p on coherence/creativity balance). Their recommended base
threshold is 0.05–0.1 of the top probability for word-level LLMs; our
`>> 2` (0.25) is deliberately harsher, on the argument that a
char-level model mid-word should be near-one-hot so aggressive
truncation is the point — and 0.25 is what the 12-prompt probe
validated. Phase 3 should sweep the shift ∈ {1,2,3,4} (0.5, 0.25,
0.125, 0.0625) on the phase-1 probe before freezing the constant,
covering both our empirical point and their recommended range.*

### Fix 4 — The lexicon guard: constrained decoding, zero invented words *by construction*

*Cause addressed: all residual word-coining, regardless of source.
Effort: the one genuinely novel build; `core/`-side, behind the frozen
API. The headline.*

Everything above reduces the invented-word *rate*. This eliminates the
*category*: the corpus has a closed vocabulary (a few thousand distinct
words — the trainer already computes exactly this set for
`invented_word_count`), so **compile that vocabulary into a
character-trie (DAWG) at blob-build time, ship it as a new section
behind the blob's model-type/format seam, and mask the sampler's
candidate set each step to characters that continue a valid corpus word
(or legally end one at a boundary character).** The model still chooses
*which* word — personality, mood, and conditioning all still come from
the logits — but it becomes structurally incapable of emitting a
character sequence that isn't a word its writers wrote. Grammar-
constrained decoding, the same idea llama.cpp's GBNF grammars use on
LLMs, has to my knowledge never been put on an N64 — and this project's
uppercase-plus-punctuation, closed-vocabulary corpus is the ideal
special case for it.

Engineering shape (all within the hard constraints):

- Integer-only, no heap: the trie is a flat, blob-baked node array
  (byte-indexed children over the ≤96-symbol vocab; a few thousand
  words of this corpus's shape ≈ tens of KB — RDRAM noise, and blob
  size was never the constraint). *Prior art to mine before building
  (phase-4 homework, not optional): speech recognition solved "decode
  only real words" decades ago with (weighted) FST lexicons — the
  compact-representation tricks there (suffix sharing via DAWG
  minimization, LOUDS-style succinct tries) are directly stealable if
  the naive flat trie comes out bigger than expected; llama.cpp's GBNF
  grammar sampler and HF's constrained beam search are the modern
  LLM-side references for the candidate-masking loop shape.*
- Decode step: walk the trie in lockstep with emission; at each step,
  intersect the top-k candidate list with the current node's valid
  continuations *before* the min-p gate and the draw. At a word
  boundary the walker resets; EOS is always legal at a boundary and
  never legal mid-word — which incidentally *also* hard-fixes the
  missed-EOS/run-to-cap failure mode.
- Fallback rule (must be explicit for bit-exactness): if the
  intersection is empty — the model wants a word the trie doesn't know
  — take the trie-legal character with the highest logit, full stop.
  Deterministic, and it turns "invented word" into "least-surprising
  real word," the correct failure direction for shipped dialogue.
- Same TDD path as every core change: reference implementation in
  `ref_impl.py` first, host tests red→green, goldens, XCHK on Ares.
- Honest limitation, stated up front: the guard fixes *spelling*, not
  *word order* — "I WANT TO TREAT OF WAIT" is all real words and would
  pass untouched. That's what Fixes 1–3 are for; the layers compose.
  The guard is the floor under them, and it makes the invented-word
  metric a *hard invariant* (a self-test assertion, not a tracked
  number) forever after.

### Fix 5 — Drop back to H=320 (and bank the win)

*Effort: config + retrain; it's a revert.*

M12 exonerated capacity as cleanly as an experiment can: 10.24x the
parameters, same corpus, same seed — same garbling, worse val loss, 44
ch/s → 5 ch/s. Keep the K-chunked kernel (it is proven, and it makes H
a free parameter for whoever needs it later), but ship the rescue on
H=320: every fix above works identically there, the RDRAM pressure
(3MB of `rspWhh` in a 4MB machine) evaporates, and the speed bar M5
set (≥30 ch/s) is honored again instead of waived. If, *after* Fixes
0–4 land, the sharpened metrics show a real capacity ceiling — thin
characters fixed by data but Selena's variety saturating, say — the
kernel is sitting there ready and the experiment finally becomes
interpretable. That is the right order: **de-noise the measurement,
then re-ask the capacity question only if a de-noised metric asks it.**

### Explicitly considered and not chosen

- **Scheduled sampling / student-forcing fine-tune** (train on the
  model's own sampled prefixes to teach recovery from derailment): real
  technique, directly aimed at exposure bias — but it treats the
  symptom at the point where the guard (Fix 4) already provides a hard
  floor and QAT (Fix 1) removes most of the derailment *sources*. Hold
  in reserve; revisit only if the Fix-0 probe still shows fluent-but-
  wrong-order text after Fixes 1–4.
- **Word/subword tokenization**: would attack coherence at the root
  (fewer sampling decisions per word), but `NGPT_GRU_MAX_VOCAB=96` makes
  a real word vocabulary impossible without a new model type and a much
  bigger softmax; the trie guard delivers the same "words are atomic"
  guarantee at decode time without touching the model contract.
- **A fourth corpus-content lever**: three negatives is enough. The
  per-character table says *balance*, not *volume*, is the remaining
  data problem.

---

## Sequencing and acceptance (walking-skeleton, one boot per phase)

| phase | ships | gate that must move |
|---|---|---|
| 1 | Fix 0 probe + gates in trainer; baseline numbers recorded for H=320 retrain (Fix 5 config) | probe exists; per-character baseline table in the milestone doc |
| 2 | Fix 1 QAT + Fix 2 rebalance, retrain, ROM boot | agreement ≥ 0.995; int8-greedy invented ≈ float-greedy invented; per-character rates converge; SELFTEST PASS |
| 3 | Fix 3 integer min-p (core + ref + goldens), ROM boot | sampled invented-word rate at ≤ half of phase-2's, temperature unchanged; XCHK PASS |
| 4 | Fix 4 lexicon trie (blob section + core walker + ref), ROM boot | **invented words = 0 as a self-test invariant**; EOS-before-cap = 100%; XCHK PASS |
| 5 | docs + versioned `.z64` + the honest writeup either way | full DoD per project method |
| 6 *(conditional)* | the capacity question, asked properly this time: H=320 vs H=768 A/B — same corpus, same seed, both through QAT and the phase-1 probe | run ONLY if phase-2–4 metrics show real strain (cast additions degrading voices, variety saturating); a de-noised win must be visible on the probe to justify H>320's speed/RDRAM cost. M12's negative was a mistrial (measured through int8/sampler/imbalance noise), not an acquittal — but the burden of proof stays on the bigger model |

Each phase is a real milestone increment: if the effort stalls after any
phase, the shipped state is still strictly better than today's, with the
improvement pinned by a metric and a booted ROM — the same discipline
that made M12's negative result trustworthy.

## Risks, stated honestly

- **QAT changes training numerics** — determinism of the float phase was
  already declared throwaway (MPS non-reproducibility is accepted since
  M12), but the QAT phase must checkpoint-cache like everything else so
  goldens stay stable between blob rebuilds.
- **The min-p threshold and the 4:1 balance target are calibrated on a
  12-prompt probe.** Phase 1's 50-line probe exists precisely to re-fit
  both before they harden into gates.
- **The trie walker is new core code on the bit-exactness path** — it is
  deliberately last, smallest, and TDD'd, but it is the one fix with ROM
  crash surface. Its fallback rule must be nailed in the reference
  implementation before a line of C exists.
- **Fluent-but-wrong-conditioning output** (M10's "Kragan tells Selena's
  joke") is *reduced* by sharper models but not *guaranteed* away by any
  fix here; the divergence metrics remain the watchdog for that distinct
  failure mode, and scheduled sampling is the named reserve lever.

## One-paragraph version for the standup

Capacity was never the bottleneck: the same M12 model that garbles at
top-k=5 already garbles under pure greedy decode *after quantization*
(5 invented words in float → 17 in int8 → 22 with the shipped sampler,
on a 12-prompt probe), and the errors concentrate in exactly the
characters with 75x less training data than Selena. The metrics never
caught it because val loss is teacher-forced, float, and
prefix-masked — blind to all three causes. Plan: add a shipped-config
coherence probe and real gates (trainer-only), quantization-aware
fine-tuning to kill the int8 flips, rebalance the cast corpus to ≤4:1,
an integer min-p gate in the sampler, and — the new idea — a
blob-baked lexicon trie that constrains decoding to real corpus words,
making invented words structurally impossible while keeping the model's
choice of *which* words. All on H=320, back at 44 ch/s. Five phases,
each ending in a booted ROM and a number that has to move.
