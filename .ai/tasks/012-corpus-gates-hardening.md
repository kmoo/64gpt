# Task 012 — corpus-gates-hardening

## CONTRACT

```yaml
id: 012-corpus-gates-hardening
goal: >
  Add edge-case test coverage to trainer/tests/test_corpus_gates.py for
  trainer/ngpt_trainer/corpus_gates.py (build_fragment_vocab,
  structural_gate, FragmentUsageTracker, density_gate) -- the module
  already ships with 5 passing tests covering the documented happy
  paths and one regression case, but several real edge cases in its
  own logic have no coverage yet.
background: >
  This is test-writing against EXISTING, already-shipped behavior, not
  new feature work. Read trainer/ngpt_trainer/corpus_gates.py's actual
  source (it's short, 37 lines) before writing assertions -- do not
  guess what it does.
constraints: |
  - Python 3.12, stdlib only (matches the module under test -- no new
    dependencies).
  - Add tests to the EXISTING file trainer/tests/test_corpus_gates.py,
    do not create a new test file and do not modify
    trainer/ngpt_trainer/corpus_gates.py itself -- this task is
    test-only, on already-shipped code.
  - Do not delete or modify any of the 5 existing test functions
    already in the file.
allowed_files:
  - trainer/tests/test_corpus_gates.py
reference_files:
  - trainer/ngpt_trainer/corpus_gates.py
acceptance_criteria:
  - >
    `build_fragment_vocab()` called with zero arguments returns an
    empty set (no texts to tokenize).
  - >
    `structural_gate("", some_nonempty_vocab)` (empty fragment string)
    returns (True, "") -- no words to tokenize means no invented words
    possible.
  - >
    `structural_gate()` is case-insensitive: a fragment containing
    lowercase text whose uppercase form IS in corpus_vocab (e.g.
    fragment "hello" when corpus_vocab contains "HELLO") passes,
    because the implementation calls .upper() on the fragment before
    tokenizing.
  - >
    Apostrophe handling: a fragment containing a contraction like
    "DON'T" is tokenized as the single token "DON'T" (matching the
    module's `_WORD_RE = re.compile(r"[A-Z']+")` regex, which includes
    the apostrophe character in the word-character class) -- verify
    "DON'T" passes structural_gate when "DON'T" (with the apostrophe)
    is itself in corpus_vocab, and FAILS when only "DONT" (no
    apostrophe) is in corpus_vocab (i.e. "DON'T" != "DONT" as tokens).
  - >
    `density_gate(usage_count=0, min_reuse=0)` returns (True, "") --
    zero usages still satisfies a zero minimum (0 >= 0).
  - >
    `FragmentUsageTracker().usage_count("never recorded")` (a fragment
    that was never passed to record()) returns 0, not an error.
  - >
    Single-character words: structural_gate's real code filters tokens
    of length > 1 before counting invented words (per the module's own
    docstring/logic) -- verify a fragment containing a single-letter
    "word" not in corpus_vocab (e.g. fragment "I SEE X" where "X" is
    not in corpus_vocab but "I" and "SEE" are) still passes, because
    single-character tokens are never counted as invented.
verification:
  - "cd trainer && uv run pytest tests/test_corpus_gates.py -v"
```

## COMPLETION

```yaml
status: done
summary: |
  Clean first-try dispatch (Qwen2.5-Coder-7B), 7 new tests, all passed
  immediately. Lead review caught one real defect the automated
  verification couldn't: test_structural_gate_single_character_words
  built corpus_vocab FROM the fragment itself, so "X" ended up in-vocab
  trivially -- the test would still pass with the single-character
  exemption deleted from structural_gate, i.e. it verified nothing.
  Fixed directly (vocab now excludes X, with an explicit assertion
  confirming that).
files_changed:
  - trainer/tests/test_corpus_gates.py
verification: |
  cd trainer && uv run pytest tests/test_corpus_gates.py -v
  12 passed in 0.01s
risks: []
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 1 / 0 / no
- claude tokens spent (contract + review, est.) vs doing it directly: low -- clean first-try dispatch, only the review pass needed real attention
- defects: caught in review = 1 (vacuous test), slipped past review = 0

## WORKER RESULT (qwen-worker)

- status: done
- attempt: verification PASS
- verification tail:

```
$ cd trainer && uv run pytest tests/test_corpus_gates.py -v
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- <repo>/trainer/.venv/bin/python
cachedir: .pytest_cache
rootdir: <repo>/trainer
configfile: pyproject.toml
collecting ... collected 12 items

tests/test_corpus_gates.py::test_build_fragment_vocab PASSED             [  8%]
tests/test_corpus_gates.py::test_structural_gate PASSED                  [ 16%]
tests/test_corpus_gates.py::test_structural_gate_counts_occurrences_not_distinct_words PASSED [ 25%]
tests/test_corpus_gates.py::test_fragment_usage_tracker PASSED           [ 33%]
tests/test_corpus_gates.py::test_density_gate PASSED                     [ 41%]
tests/test_corpus_gates.py::test_build_fragment_vocab_empty PASSED       [ 50%]
tests/test_corpus_gates.py::test_structural_gate_empty_fragment PASSED   [ 58%]
tests/test_corpus_gates.py::test_structural_gate_case_insensitivity PASSED [ 66%]
tests/test_corpus_gates.py::test_structural_gate_apostrophe_handling PASSED [ 75%]
tests/test_corpus_gates.py::test_density_gate_zero_usages PASSED         [ 83%]
tests/test_corpus_gates.py::test_fragment_usage_tracker_never_recorded PASSED [ 91%]
tests/test_corpus_gates.py::test_structural_gate_single_character_words PASSED [100%]

============================== 12 passed in 0.01s ==============================
```
