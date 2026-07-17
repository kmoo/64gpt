# Task 008 — test coverage for train_corpus / train_corpus_conditioned

## CONTRACT

```yaml
id: 008-train-corpus-tests
goal: >
  Write pytest tests for train_corpus and train_corpus_conditioned in
  trainer/ngpt_trainer/model.py -- the two real training functions that
  make_m4_blob.py, make_m7_blob.py, and make_m8_blob.py all actually call.
  They currently have zero dedicated test coverage; only the toy
  overfit()/overfit_corpus() variants are tested (test_overfit.py,
  test_conditioning.py).
background: >
  Both functions are already implemented and already shipped (M4/M7/M8
  all trained real models through them) -- this is test-writing against
  EXISTING behavior, not TDD-first spec work. Read the real function
  bodies in reference_files before writing assertions; do not guess
  signatures or behavior.
constraints: |
  - Python 3.12, stdlib + pytest + torch (already dev dependencies, same
    as test_overfit.py and test_conditioning.py).
  - HARD PROJECT RULE (from CLAUDE.md): trainer/ PyTorch code must train
    with device="cpu", NEVER the default (which auto-selects MPS when
    available). Every single call to train_corpus(...) or
    train_corpus_conditioned(...) in your tests MUST pass device="cpu"
    explicitly. Never omit the device= argument.
  - Keep every test FAST: use a tiny hand-written corpus (3-5 short
    prompt/response pairs, similar in spirit to test_overfit.py's
    CORPUS), hidden=8, max_epochs=3, patience=2, batch_size=4. Do not
    use any real project corpus (selena_corpus.py, guard_corpus.py,
    corpus_gen.py) -- those are large and would make tests slow.
  - train_corpus(pairs, vocab, hidden=128, seed=0, lr=3e-3, batch_size=64,
    max_epochs=60, patience=5, device=None) -> CharGRU. Build vocab with
    Vocab.from_text() over the full corpus text (prompts + responses),
    same pattern as test_overfit.py.
  - train_corpus_conditioned(train_pairs, val_pairs, vocab, hidden=256,
    seed=0, lr=3e-3, batch_size=64, max_epochs=60, patience=5,
    device=None) -> CharGRU. Requires BOTH train_pairs and val_pairs to
    be non-empty lists of (prompt, response) tuples -- do not test with
    an empty val_pairs list, that hits an unhandled edge case in the
    real implementation, out of scope for this task.
  - For each function, test exactly two things:
    1. Returned model is a CharGRU instance, in eval mode
       (model.training is False), and has a `final_loss` attribute that
       is a float.
    2. Determinism: calling the function twice with identical arguments
       including the same seed=0 produces models whose state_dict()
       tensors are all torch.equal(...) to each other, key by key (loop
       over state_dict().items() from both models and compare).
  - Do not test _batchify or _batchify_masked directly (leading
    underscore = private helpers, already exercised indirectly through
    the two public functions above).
  - Do not assert exact loss VALUES (non-portable across torch/hardware
    versions) -- only assert final_loss is a float, nothing about its
    magnitude.
allowed_files:
  - trainer/tests/test_train_corpus.py
reference_files:
  - trainer/ngpt_trainer/model.py
  - trainer/tests/test_overfit.py
  - trainer/tests/test_conditioning.py
verification:
  - cd trainer && uv run pytest tests/test_train_corpus.py -v
```

## COMPLETION

```yaml
status: done (escalated -> finished by lead)
summary: |
  qwen-worker never actually dispatched: the local model infra itself was
  broken, not a model-quality issue. OPENCODER_TYPE=code's configured
  models (deepseek-coder-6.7b, deepseek-coder-33b) are NOT in the local
  HF cache (only Qwen2.5-Coder-{0.5B,7B,14B} are) and fail to download
  with 401 Unauthorized/RepositoryNotFoundError from HuggingFace -- a
  prior memory claim that "both models are downloaded and cached" was
  never actually verified and was wrong. Killed the wedged dispatch +
  server, recorded the finding, and wrote the test file directly against
  the same contract spec instead of re-dispatching against a broken
  model. One real bug found while writing it (not in model.py, in my own
  contract): train_corpus() carves its OWN internal val split
  (encoded[9::10]) rather than taking one as an argument, so it needs
  >=10 pairs or that split is empty and val_loss() divides by zero --
  the contract's suggested "3-5 pairs" was wrong for train_corpus (still
  fine for train_corpus_conditioned, which takes explicit val_pairs).
  Fixed by using a 10-pair corpus for the train_corpus tests only.
files_changed:
  - trainer/tests/test_train_corpus.py
verification: |
  cd trainer && uv run pytest tests/test_train_corpus.py -v
  4 passed in 0.78s
risks: []
needs_review:
  - ~/bin/opencoder's OPENCODER_TYPE=code/text model references are
    currently broken on this machine (deepseek-coder/Mistral not
    downloadable) -- needs an HF auth/repo-ID fix before qwen-worker can
    be trusted for code contracts again; see memory
    opencoder-model-setup.md.
```

## METRICS

- dispatches / retries / escalated: 0 / 1 / yes (escalated before any dispatch reached the model -- infra failure, not a model output failure)
- claude tokens spent (contract + review, est.) vs doing it directly: contract-authoring cost was wasted this time since it never got to run against qwen; wrote the file directly instead
- defects: caught in review = 1 (contract's own "3-5 pairs" suggestion was wrong for train_corpus's internal split), slipped past review = 0
