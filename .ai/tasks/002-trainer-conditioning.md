# Task 002 — trainer-conditioning

## CONTRACT

```yaml
id: 002-trainer-conditioning
goal: >
  Extend the trainer so a single GRU memorizes all 12 prompt->response
  pairs from ngpt_trainer.corpus, and prompted greedy generation
  reproduces each response exactly. New pytest file proves it.
background: >
  M2 shipped a one-line overfit (ngpt_trainer/model.py). M3 adds
  conditioning by TEXT-PROMPT PRIMING (docs/milestones/m3.md): training
  sequence per pair is inputs EOS+prompt+response, targets
  prompt+response+EOS, cross-entropy over the full sequence. Prompted
  generation: one forward over EOS+prompt (batch of one) primes the
  hidden state AND the last position's logits give the first prediction;
  argmax-generate from there until EOS, exactly like M2's loop.
constraints: |
  - Existing functions in model.py (CharGRU, one_hot, overfit,
    generate_greedy) must keep working — the M2 test suite runs
    unchanged and must stay green.
  - torch + stdlib only. Deterministic: torch.manual_seed(seed) first.
  - No prints. Type hints on new public functions.
  - New functions in ngpt_trainer/model.py:
    overfit_corpus(pairs, vocab, hidden=64, seed=0, lr=5e-3,
      max_steps=8000, target_loss=1e-3) -> CharGRU
      trains on all pairs each step (sum of per-sequence CE losses for
      the backward pass), early-stops when the MAX per-sequence loss is
      below target_loss, sets model.final_loss to that max, eval mode.
    generate_greedy_prompted(model, vocab, prompt, max_len=256) -> str
      primes as described in background; ties in argmax break toward
      the lowest id; returns response text only (no prompt, no EOS).
allowed_files:
  - trainer/ngpt_trainer/model.py
  - trainer/tests/test_conditioning.py
acceptance_criteria:
  - tests/test_conditioning.py trains ONCE (module-scoped fixture) via
    overfit_corpus(corpus.pairs(), Vocab.from_text(corpus.corpus_text()))
    and asserts: every one of the 12 prompts generates its exact
    response; model.final_loss < 1e-3; a second overfit_corpus with the
    same seed generates identical strings for all 12 prompts.
  - The full existing pytest suite stays green.
verification:
  - scripts/verify.sh
```

## COMPLETION

```yaml
status: escalated
summary: worker's overfit_corpus deviated from the summed-backward
  constraint and generate_greedy_prompted re-primed the prompt on every
  step; test file hallucinated APIs (trainer.* import root, nonexistent
  Corpus class, use-before-assignment). Lead rewrote both per contract.
files_changed: [trainer/ngpt_trainer/model.py, trainer/tests/test_conditioning.py]
verification: |
  scripts/verify.sh FAIL x3 (worker attempts, ImportError at collection);
  lead rewrite verified separately (see M3 commit).
risks: []
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 3 / 2 / yes
- shakedown notes: worker loop itself worked exactly as designed
  (sanity gate passed FILE blocks, applied, ran verify.sh, escalated
  with a clean review package after cap). Model quality was the limit,
  not the harness. Contract lesson: inline exact import lines for the
  test file, not just API summaries.

