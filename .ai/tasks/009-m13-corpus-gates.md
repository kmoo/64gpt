# Task 009 — m13-corpus-gates

## CONTRACT

```yaml
id: 009-m13-corpus-gates
goal: >
  Implement two pure, no-LLM, no-training corpus-quality gates for M13's
  phrase-bank distillation pipeline (docs/milestones/m13.md mechanisms 2
  and 3): a structural-compatibility gate that rejects candidate bank
  fragments with the wrong vocabulary/invented-word signature before any
  LLM judging, and a density-reuse gate that only keeps a fragment once
  it has been reused across at least K distinct generated corpus combos.
  Both operate on plain strings/counts -- no model, no LLM call, no
  training involved anywhere in this module.
background: >
  This project (64GPT) generates NPC dialogue training corpora from
  template-grammar banks (openers/body-skeletons/catchphrases). M13's
  plan document found that LLM-authored fragments must be checked BEFORE
  being trusted, because a prior milestone (M9) found LLM-judged "good"
  text trained badly due to low repetition/reuse density.
  `invented_word_count`/`build_corpus_vocab` (trainer/make_m12_1_blob.py,
  near line 319) is the existing pattern for detecting invented/
  out-of-vocabulary words against a known corpus vocabulary:
  tokenize with regex `[A-Z']+` after `.upper()`, flag any token of
  length > 1 not present in the known vocab set. `jaccard_distance`/
  `trigrams` (trainer/ngpt_trainer/divergence.py) is this project's
  existing trigram-overlap similarity metric, shown here only as a style
  reference for how this codebase writes small pure text-analysis
  functions -- not required by this task's functions directly. This task
  creates a new standalone module with its own copy of the
  tokenization/vocab logic (no import dependency on file-specific corpus
  builders) so it can be called generically by any future corpus
  generator or gate-checking script.
constraints: |
  - Pure Python 3.12, stdlib only plus whatever trainer/pyproject.toml
    already depends on (no new third-party packages, no new deps added
    to pyproject.toml).
  - No LLM calls, no `opencoder`/`subprocess` dispatch, no PyTorch, no
    model loading of any kind -- this module runs BEFORE any LLM judging
    step and must have zero such dependency.
  - No global mutable state; every function takes its inputs as
    arguments and returns a plain value (bool/tuple/int/set). Any
    fragment-usage counter must be an explicit object/dict the CALLER
    owns and passes in -- never a module-level dict.
  - Plain functions with type hints on every public function/method.
  - No docstrings that just restate the signature -- only add one where
    the WHY is non-obvious (e.g. why a token-length filter exists).
allowed_files:
  - trainer/ngpt_trainer/corpus_gates.py
  - trainer/tests/test_corpus_gates.py
reference_files:
  - trainer/make_m12_1_blob.py
  - trainer/ngpt_trainer/divergence.py
  - trainer/m11_1_lore_bank_experiment.py
test_files:
  - trainer/tests/test_corpus_gates.py
acceptance_criteria:
  - >
    `build_fragment_vocab(*texts: str) -> set[str]`: tokenizes each text
    with regex `[A-Z']+` after `.upper()` (identical rule to
    `build_corpus_vocab` in make_m12_1_blob.py) and returns the union
    vocab set.
  - >
    `structural_gate(fragment: str, corpus_vocab: set[str], max_invented: int = 0) -> tuple[bool, str]`:
    tokenizes `fragment` with the same rule, filters tokens of length > 1,
    and counts how many are NOT in `corpus_vocab` (an "invented word",
    same definition as `invented_word_count` in make_m12_1_blob.py).
    Returns `(True, "")` if the invented count is <= max_invented.
    Returns `(False, reason)` otherwise, where `reason` is a string that
    names the FIRST invented word found, e.g.
    `"invented word: FOOBAR"`.
  - >
    A fragment-usage tracker: either a `FragmentUsageTracker` class with
    `record(fragment: str, combo_text: str) -> None` and
    `usage_count(fragment: str) -> int` methods, OR a pair of free
    functions operating on a `dict[str, int]` passed by the caller --
    pick ONE approach and say which in a short module-level comment.
    Either way, "usage" means: this exact fragment string appears as a
    substring inside `combo_text`. Recording the same (fragment,
    combo_text) pair twice must not double count if `combo_text` is
    literally the same string passed twice (idempotent per distinct
    combo) -- track by a set of combos seen per fragment internally if
    needed, but expose only the counting API described above.
  - >
    `density_gate(usage_count: int, min_reuse: int) -> tuple[bool, str]`:
    returns `(True, "")` if `usage_count >= min_reuse`, else
    `(False, reason)` with a reason string stating the actual count and
    the required minimum, e.g. `"used 2 times, need >= 3"`.
  - >
    trainer/tests/test_corpus_gates.py covers, at minimum: (a) a
    fragment using only known vocab passes structural_gate; (b) a
    fragment with one invented word fails structural_gate with a reason
    string containing that word; (c) the same invented-word fragment
    passes when max_invented is raised to tolerate it; (d) usage
    tracking counts a fragment appearing in 3 distinct combo strings as
    3, and recording the exact same combo string twice for the same
    fragment does not inflate the count past the number of distinct
    combos actually passed in; (e) density_gate passes at
    usage_count == min_reuse and fails at usage_count == min_reuse - 1
    (exact boundary, not approximate).
verification:
  - "cd trainer && uv run pytest tests/test_corpus_gates.py -v"
```

## COMPLETION

```yaml
status: pending
summary:
files_changed: []
verification: |
risks: []
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 0 / 0 / no
- claude tokens spent (contract + review, est.) vs doing it directly:
- defects: caught in review = 0, slipped past review = 0
