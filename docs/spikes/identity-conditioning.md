# Spike: identity-conditioning divergence — VERDICT: GATE CLEARS

**Script:** `trainer/spike_identity.py` (run only, not shipped — no blob,
no ROM, no quantization). **Question:** does text-priming through the
shared H=128 GRU (the existing `ngpt_reset`-prompt mechanism, unchanged)
actually carry an identity signal, or does the model learn to ignore an
`N:`-style tag the way M4's WIZARD/CALM/THEFT miss suggested it might?
This gates all of `docs/milestones/m7.md` — Selena's corpus wasn't
authored until this cleared. Same discipline `docs/spikes/rsp-matvec.md`
used before M6.1 bet a milestone on the RSP kernel.

## Verdict (2026-07-16, host-only, no ROM)

| run | metric | identity-swap div | mood-swap div | gate |
|---|---|---|---|---|
| 1 — greedy decode, 360-line corpus | argmax completion, single sample | 0.000 | 0.935 | **FAILS (artifact)** |
| 2 — sampled (T=0.7, 5 draws), same corpus, val loss 0.121 | cross-set trigram Jaccard | 0.913 | 0.942 | fails, narrow margin |
| 3 — sampled, 540-line corpus, val loss 0.082 | cross-set trigram Jaccard | **0.966** | 0.816 | **PASSES** |

**GATE CLEARS on run 3** — identity-swap divergence (0.966) exceeds
mood-swap divergence (0.816), the in-mechanism baseline already known to
work since M3. Selena's real corpus and H=256 training proceed as scoped
in `m7.md`, with the mechanism now evidenced, not just assumed.

## Two false starts, and why they don't count

**Run 1 (greedy decode) was measuring the wrong thing.** The spike
corpus deliberately mirrors M4's grammar: each combo's response is drawn
from a small opener × body mixture (`rng.choice` at corpus-gen time,
`ngpt_trainer/corpus_gen.py`'s own pattern), so the *true* target given
a prompt is a distribution over several strings, not one canonical
answer. Greedy (argmax) decoding on a model that has correctly learned
such a mixture doesn't sample from it — it walks the single
highest-probability path, which for a well-fit multi-modal distribution
tends to collapse onto whichever global string has the most cumulative
character-level support, regardless of which mode of the mixture the
conditioning was pointing at. M4 hit this exact issue and built a real
temperature+top-k sampler (`ngpt_trainer/ref_impl.py generate_sampled`)
for its own acceptance goldens for precisely this reason — the spike
script had reused the wrong decode helper (`model.py`'s
`generate_greedy_prompted`, meant for M2/M3's single-canonical-answer
corpora) out of convenience. Fixed by adding
`generate_sampled_prompted` (ancestral sampling, fixed seed per draw,
T=0.7) and measuring divergence across 5-draw sample sets per condition
instead of single strings.

**Run 2 (sampled, but undertrained) was a real signal, just a weak
one.** Val loss 0.121 with visible garbled completions in a meaningful
fraction of samples (`'WHIT ASGL TALE AREVELTEND|HO DO CO AIL...'`)
meant the model hadn't converged — noise depresses divergence on both
axes and narrows real gaps. Bumped `per_combo` 20→30 (360→540 lines,
still "a few hundred," per the spike's own budget) and training patience
8→20 / max_epochs 80→200. Val loss dropped to 0.082, garbled completions
disappeared, and the gap opened decisively (0.966 vs 0.816, up from
0.913 vs 0.942). Cheap to rule out — a few minutes of extra training —
and worth doing before trusting a near-miss result this close to a
milestone-shaping decision.

**Methodology lesson for M7/M8's real evaluation protocol:** always
measure conditioning-ablation divergence with the project's real sampler
semantics (temperature + multiple draws), never greedy, whenever the
underlying corpus has multiple valid completions per condition — which
every M4-style generated corpus does by construction. Worth stating
explicitly in `m7.md`'s Evaluation Protocol since the same mistake is
easy to repeat there.

## What run 3's samples actually look like

Each identity keeps its own vocabulary consistently under repeated
sampling, with mood layered on top of it — exactly the "identity is
orthogonal to mood, not entangled with it" property the corpus was
designed to test:

- `N=ID_A M=CHEERFUL C=GREETING` draws: *"GREAT TO SEE YOU! WHAT ARE WE
  DOING TODAY?"*, *"HEY THERE! WHAT ARE WE DOING TODAY?"*
- `N=ID_B M=CHEERFUL C=GREETING` draws: *"SPLENDID TIMING! STATE YOUR
  INTENTIONS, TRAVELER."*, *"MY SPIRITS ARE HIGH! SHALL WE PROCEED WITH
  THE DAY'S BUSINESS?"*
- `N=ID_A M=WORRIED C=FAREWELL`: *"WAIT, WHAT HAPPENED? DON'T BE A
  STRANGER!"* vs `N=ID_B` same combo: *"I MUST CONFESS UNEASE. FAREWELL,
  AND TAKE CARE."*

Full per-condition sample dump: `trainer/spike_identity.py`'s stdout
(not committed — rerun with `uv run python spike_identity.py` from
`trainer/` to reproduce; deterministic given the pinned seeds).

## What adoption looks like (per m7.md, now unblocked)

- Selena's real corpus + H=256 training proceeds as scoped in
  `docs/milestones/m7.md`'s "Building the magic zone model" section.
- The Evaluation Protocol's conditioning-ablation divergence table
  reuses this run's method (sampled, not greedy) at Selena's real scale,
  as a re-check against real per-axis numbers (identity/mood/trust/
  context), not a first proof — this spike was the first proof.
- Fixed at n-gram (character-trigram Jaccard) divergence per the M7 open
  questions resolution — cheap, no probability distributions needed,
  matches what this spike already validated works as a discriminating
  signal.
