# Task 006 — test coverage for divergence.py

## CONTRACT

```yaml
id: 006-divergence-tests
goal: >
  Write pytest tests for the already-implemented, already-committed
  trainer/ngpt_trainer/divergence.py module (trigram-Jaccard divergence),
  which currently has zero test coverage — every other module in
  ngpt_trainer/ has a matching test_*.py except this one.
background: >
  divergence.py is frozen (used by make_m7_blob.py's acceptance gates and
  the identity-conditioning spike, both already landed). This is pure
  test-writing against EXISTING behavior, not TDD-first spec work — read
  the real implementation in reference_files and test what it actually
  does, don't guess.
constraints: |
  - Python 3.12, stdlib only + pytest (already a dev dependency, same as
    every other trainer/tests/test_*.py file).
  - Test the three public functions: trigrams(s), jaccard_distance(a, b),
    cross_set_divergence(samples_a, samples_b).
  - trigrams(s) behavior to verify: for len(s) < 3 it returns a set
    containing exactly ONE element, s itself — e.g. trigrams("") == {""},
    trigrams("ab") == {"ab"}. For len(s) >= 3 it returns the set of all
    contiguous 3-character sliding-window substrings, e.g.
    trigrams("abcd") == {"abc", "bcd"} (duplicates collapse since it's a
    set — pick a test string with a repeated trigram to verify this,
    e.g. "aaaa" -> {"aaa"}).
  - jaccard_distance(a, b) = 1 - |trigrams(a) & trigrams(b)| /
    |trigrams(a) | trigrams(b)|. IMPORTANT: because trigrams() never
    actually returns an empty set (the len<3 branch always returns a
    1-element set, even trigrams("") == {""}), the union of two trigram
    sets is never empty in practice — do not write a test that expects
    jaccard_distance to hit its "empty union -> 0.0" branch via two empty
    strings; trigrams("") == {""} so jaccard_distance("", "") == 0.0 via
    the normal identical-sets path (distance 0.0), not the empty-union
    guard. Cover: identical strings -> 0.0, completely disjoint trigram
    sets -> 1.0, and one partial-overlap case where you hand-compute the
    expected fraction.
  - cross_set_divergence(samples_a, samples_b) returns the mean of
    jaccard_distance(a, b) over the full cross product (every a in
    samples_a paired with every b in samples_b) -- NOT deduplicated, NOT
    symmetric-pair-only. For single-element lists it equals
    jaccard_distance of that one pair. Do NOT test empty-list inputs --
    the real implementation divides by len(divs) with no guard, so an
    empty samples_a or samples_b raises ZeroDivisionError; that's
    existing, accepted behavior, not something to add coverage for or
    "fix" — leave it alone, this task is test-writing only, no
    implementation changes.
  - Do not modify trainer/ngpt_trainer/divergence.py itself.
allowed_files:
  - trainer/tests/test_divergence.py
reference_files:
  - trainer/ngpt_trainer/divergence.py
  - trainer/tests/test_effective_diversity.py
verification:
  - cd trainer && uv run pytest tests/test_divergence.py -v
```

## COMPLETION

```yaml
status: done (escalated -> finished by lead)
summary: |
  qwen-worker escalated after 2 verification failures + 2 degenerate
  retries (root cause: dispatched under critical system memory pressure,
  14b model auto-upgraded while other work was competing for RAM). The
  first attempt's test file was 7/8 correct; the one failure was
  test_jaccard_distance_partial_overlap, which used jaccard_distance("abc",
  "bca") == 0.5 as the expected value -- but that example was wrong in
  the CONTRACT itself (written by lead, not qwen's error): 3-char strings
  produce exactly one trigram each, so two different 3-char strings are
  always fully disjoint (distance 1.0), never partially overlapping.
  Fixed by replacing with jaccard_distance("abc", "abcd") == 0.5, a real
  partial-overlap case (trigrams("abc")={"abc"} is a strict subset of
  trigrams("abcd")={"abc","bcd"}), hand-verified against the actual
  implementation. No other changes needed.
files_changed:
  - trainer/tests/test_divergence.py
verification: |
  cd trainer && uv run pytest tests/test_divergence.py -v
  8 passed in 0.01s
risks: []
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 1 / 2 / yes
- claude tokens spent (contract + review, est.) vs doing it directly: contract + 1-line fix + review, cheap either way for a module this small
- defects: caught in review = 1 (bad example baked into the contract by lead, not a qwen defect), slipped past review = 0

## WORKER RESULT (qwen-worker)

- status: escalated
- attempt: verification FAIL
- attempt: verification FAIL
- attempt: rejected before apply: degenerate output (same line repeated >25x)
- attempt: rejected before apply: no FILE blocks in output
- verification tail:

```
(escalated at sanity gate: no FILE blocks in output)
```
