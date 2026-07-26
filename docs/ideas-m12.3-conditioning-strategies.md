# Better conditioning strategies — getting emotion/trust/personality *into* a tiny model

**Status: PROPOSED as M12.3 (2026-07-25).** Forward-looking design
exploration, not yet accepted. Builds directly on M12's honest negative
(capacity doesn't help) and M12.1's diagnosis (`docs/ideas-coherence-
rescue-plan.md`). M12.2 (the corpus voice-polish pass) is option **E**
below and is already in flight. Nothing here changes M13 (portability).

Audience note (per `CLAUDE.md`): written for a good software engineer
with zero ML background — the *why* matters as much as the *what*.

## Executive summary

The game hands the model a bundle of live state — this NPC's
personality, the player's trust level, the current mood and situation —
and expects an in-character line back. Today that bundle is **flattened
into a text prefix** (`M:sassy R:neutral OCC:noble …|`) and the char-GRU
is trained to continue it. That is a real, named technique (**control
codes**, CTRL). It works, but it has a specific weakness at our scale.

**The reframe:** M10, M11.1, and M12 all tried to make the model *learn*
the corpus better (more pairs, more shared content, 10× capacity) and all
came back negative (`docs/ideas-coherence-rescue-plan.md`). M12.1 showed
the garbling was mostly decoding, not the float model. This doc adds the
next lens:

> The bottleneck may not be "too few neurons to store the voices." It may
> be that **the conditioning signal doesn't reach the output strongly or
> persistently.** A 100K-param GRU has to encode the whole `M:/R:/D:/OCC:`
> prefix into its hidden state and *carry it* to the end of a reply. Tiny
> RNNs have short memory, so by the last few characters the conditioning
> has faded — voice drift, off-mood endings, generic closers.

If that's true, the fix isn't more size or more epochs (both tried, both
negative) — it's **delivering the conditioning to the model in a way it
can't forget.** Options A and B below do exactly that.

## How conditioning works today (the baseline)

- `npc_service.prompt_fields()` serializes state into one string, e.g.
  `P:woman D:warm OCC:noble SPECIES:elf R:neutral M:cheerful C:greeting
  AUD:witnessed EV:none|`.
- Training pairs `(prompt_string, response_string)` teach the char-GRU to
  predict the response characters given the prompt as a **prefix**.
- The model has **no "trust" or "emotion" variable** — `M:sassy` is just
  the characters `M`,`:`,`s`,`a`,`s`,`s`,`y`. It learns, statistically,
  that those characters tend to be followed by sassy-flavored text.
- At runtime the game rebuilds the same string and the model continues it,
  one character at a time, until EOS.

This is **CTRL-style control-code conditioning** — proven at 1.6B params.
Our problem is that we have ~100K, int8, running integer-only on an N64.

## The options at a glance

| # | Strategy | Core idea | Fit for a tiny int8 model | Touches N64 core / blob? | Effort |
|---|----------|-----------|---------------------------|--------------------------|--------|
| A | **Per-step attribute embeddings** (persona/speaker embeddings) | Give each field value a small learned vector; concat onto the char embedding at *every* timestep | **Best** — conditioning stays present, can't decay | Yes (model + blob + kernel) | Medium |
| B | **FiLM modulation** | Derive per-channel scale/shift `γ⊙h+β` from the attribute embedding; modulate the GRU hidden state each step | **Excellent** — 2 params/feature, built for small nets | Yes (extra per-channel mul/add) | Medium |
| C | **Attribute logit bias** | Add a learned bias to output logits as a function of (mood, tier) | Good, cheap, **stackable** with A/B | Yes (small, additive) | Low |
| D | **Decode-time steering** (FUDGE / GeDi / DExperts) | Reweight logits at decode with an attribute discriminator | Powerful but needs a **second model** at decode | Partially — we already do decode-time control (trie) | High / mostly out |
| E | **Sharpen the prefix signal** (current M12.2) | Cleaner, more distinct phrase banks per field value | **Zero architecture change**, stacks with all above | No | Done/ongoing |

## The options in detail

### A. Per-step attribute embeddings — *the recommended spike*

**Mechanism.** Instead of (or in addition to) the text prefix, give each
attribute value its own small **learned embedding vector**:
`mood` (5 values), `trust tier` (3, or the full 6 R: tiers), `occupation`
(~10), `descriptor` (~8), `context` (8), etc. At **every** generation
step, concatenate the relevant vectors onto the character embedding before
it enters the GRU. This is the classic **Persona-Based Neural Conversation
Model** (Li et al. 2016): a speaker/attribute vector fed alongside word
embeddings at each timestep.

**Why it fits us.** The conditioning is now a *constant side-input*, not
something the model has to remember from a prefix. It cannot decay across
a long reply — directly targeting the drift failure mode. This is the
single change both the literature and our own negative results point at.

**Cost.** A handful of tiny embedding tables (a few dozen vectors total).
Quantizes cleanly to int8. Integer inference stays simple: table lookup +
concatenation before the existing matvec. Adds a small, fixed number of
input dimensions to the GRU's input-to-hidden matrix.

**Risk.** Changes the model architecture, the `NGPT` blob layout (new
embedding tables to serialize), and the N64 kernel (concat step + wider
input matrix). Must preserve the frozen streaming API and the
bit-exactness proof (host ref == device).

### B. FiLM — feature-wise linear modulation

**Mechanism.** FiLM (Perez et al. 2017): from the attribute embedding,
produce two vectors `γ` (scale) and `β` (shift) and apply them per-channel
to the GRU hidden state each step: `h' = γ ⊙ h + β`. The attributes now
*reshape the model's internal activations* rather than just adding input.

**Why it fits us.** Only **two parameters per modulated feature**, and
FiLM was specifically shown to make *small* networks obey a conditioning
signal. Also keeps conditioning live at every step. Often stronger than
plain concatenation (A) for the same parameter budget.

**Cost.** A per-channel multiply-and-add inside the integer core — modest,
well within the int8 budget, but slightly more kernel work than A. Same
blob/kernel/bit-exactness caveats as A.

### C. Attribute-conditioned logit bias — cheap and stackable

**Mechanism.** Learn a small bias vector added to the **output logits** as
a function of the discrete condition (e.g. one bias table keyed by mood,
another by tier). A poor-man's **DExperts** (Liu et al. 2021): nudge which
characters are favored, conditioned on the attribute.

**Why it fits us.** Dead simple, purely additive, trivially int8. Won't by
itself fix deep voice drift, but it's a cheap *complement* to A or B and a
low-risk standalone experiment.

**Cost.** Small additive table at the output layer. Minimal kernel change.

### D. Decode-time steering (FUDGE / GeDi / DExperts) — mostly out for N64

**Mechanism.** Steer generation at decode by reweighting the model's
logits with a separate **attribute discriminator**: **FUDGE** (Yang &
Klein 2021) predicts whether a prefix will satisfy the attribute and
reweights accordingly; **GeDi** (Krause et al. 2020) uses class-conditional
LMs via Bayes' rule. We *already do decode-time control* — the M12.1
lexicon-trie guard is exactly this shape (structural, not learned).

**Why mostly out.** Full FUDGE/GeDi need a **second model** running at
decode — too heavy for the N64. The cheap, on-device-friendly analogue of
this family is option **C** (a static attribute-bias table), which
captures a slice of the benefit without a second network.

### E. Sharpen the prefix signal — the current M12.2 pass

**Mechanism.** Keep the prefix approach, but make each field value's
training text as **distinct and clean** as possible: villain lines that
are unmistakably menacing, princess lines unmistakably regal, etc. A
stronger correlation is easier for a tiny model to latch onto.

**Why it fits us.** **Zero architecture change**, no blob/kernel/bit-exact
risk, and it *stacks* with every option above. This is literally the
M12.2 corpus voice-polish work. It's the correct low-risk first move — but
it can only strengthen the *existing* channel; it can't fix the channel's
structural memory-decay problem, which is what A/B are for.

## Constraints any change must respect (the guardrails)

From `CLAUDE.md` / `docs/plan.md`:

- **Integer math only in `core/`.** No floats, no heap, no exceptions. New
  ops (concat, per-channel mul/add, bias) must be int8/int32 friendly.
- **`NGPT` blob is big-endian, parsed byte-by-byte.** New embedding/bias
  tables need a versioned format extension (blob model-type field), not a
  silent layout change. Spec: `docs/03-blob-format.md`.
- **Frozen streaming API** (`ngpt_load → ngpt_reset → ngpt_step →
  NGPT_EOS`). Conditioning must arrive through the existing prompt/priming
  path or a deliberately versioned extension — not a new ad-hoc input.
- **Bit-exactness proof.** Host `ref_impl` must stay bit-identical to the
  N64 kernel. Any new op is implemented and golden-tested on the host
  first (`tests/vectors/`), then ported.
- **QAT survives quantization.** Prove the conditioning gain survives int8
  (train-aware quantization already in the pipeline), not just in fp32.

## Recommended sequencing

1. **E — done/ongoing (M12.2).** Measure how much the sharpened voices
   alone buy us once the current retrain lands. If coherence is fixed,
   stop here.
2. **A — host-only spike first.** Prototype per-step attribute embeddings
   in `trainer/` + `ref_impl.py`, no N64 changes yet. Reuse M12.1's
   coherence probe (`trainer/m12_1_coherence_probe.py`): invented-word
   count + cross-set divergence + golden samples, greedy and sampled.
   **Decision gate:** does it beat the M12.2 prefix baseline on the same
   probe? Only then commit to blob + kernel work.
3. **B — if A helps but not enough.** FiLM on top of, or instead of, the
   concatenation in A. Same host-first discipline.
4. **C — cheap add-on.** Fold an attribute logit-bias table into whichever
   of A/B ships; also viable as a standalone low-risk experiment.
5. **D — skip** for N64 (second model too heavy); C is the affordable
   stand-in.

## How we'll know it worked (measurement)

Reuse the existing evaluation rather than inventing one:

- **Invented-word count** on the 12-prompt probe (the M12.1 headline metric).
- **Cross-set divergence** (`trainer/divergence.py`) — do different
  `M:`/`R:` values actually produce measurably different text, including on
  **held-out combos** (the M9 compositionality test)?
- **Golden samples** eyeballed for voice: same NPC, swap `M:sassy` vs
  `M:tender`, does it swing?
- **Long-generation drift** specifically — measure voice consistency at the
  *end* of a reply, since that's the failure A/B target.

## Open questions

- Do we keep the text prefix *and* add embeddings (belt-and-suspenders), or
  replace it? Keeping both is safer but wider input.
- Which fields deserve their own embedding vs. staying in the prefix?
  (Mood and trust tier are the highest-value first candidates.)
- Full 6 R: tiers vs. the 3 the demo's D-pad currently produces.
- Does FiLM (B) justify its extra kernel cost over plain concat (A) at
  this scale, or is A already enough?

## References

- CTRL — control-code conditioning (what we do today), Keskar et al. 2019:
  https://arxiv.org/abs/1909.05858
- Persona-Based Neural Conversation Model (per-step speaker embeddings),
  Li et al. 2016: https://arxiv.org/abs/1603.06155
- FiLM — Feature-wise Linear Modulation, Perez et al. 2017 (overview):
  https://www.emergentmind.com/topics/feature-wise-linear-modulation-film
- FUDGE — Controlled Text Generation with Future Discriminators, Yang &
  Klein 2021: https://arxiv.org/pdf/2104.05218
- DExperts — Decoding-Time Controlled Generation, Liu et al. 2021:
  https://arxiv.org/pdf/2105.03023
- PPLM — Plug-and-Play Language Models, Dathathri et al. 2019:
  https://arxiv.org/pdf/1912.02164
