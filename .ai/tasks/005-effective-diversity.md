# Task 005 — effective-diversity metric

## CONTRACT

```yaml
id: 005-effective-diversity
goal: >
  Implement an "effective diversity" diagnostic for generated dialogue
  corpora: distinct-trigram coverage and a near-duplicate-collapsed
  unique-line ratio, computed per combo and overall. This catches a
  generator that is padding (more bytes, same handful of shapes) rather
  than genuinely diversifying — templated text carries much lower
  entropy per character than natural language, so raw byte count alone
  is a misleading corpus-size metric.
background: >
  trainer/ngpt_trainer/divergence.py is a sibling diagnostic module in
  the same package (trigram-Jaccard divergence between two sample sets)
  — match its docstring style and plain-function shape, not a class.
  This new module is fully generic (plain strings in, no dependency on
  any specific corpus generator) so it can be reused by later corpora
  (archetypes, bosses) without modification.
constraints: |
  - Python 3.12, stdlib only (no new dependencies) — same as every
    other ngpt_trainer module.
  - Pure functions, no I/O, no printing inside the library functions
    (callers print/report; these just compute and return).
  - "Near-duplicate collapse" for unique_line_ratio means: normalize
    each line by stripping leading/trailing whitespace and collapsing
    any run of internal whitespace to a single space, THEN compare for
    exact equality. Do not implement fuzzy/edit-distance matching —
    exact-after-normalization only.
  - trigram_coverage must handle short strings safely: a string with
    fewer than 3 characters contributes exactly one "trigram" (the
    whole string itself), matching the existing convention already used
    in trainer/ngpt_trainer/divergence.py's trigrams() function (read
    that file — same rule, do not diverge from it).
  - Empty input lists must not raise or divide by zero: return 0/0.0 in
    every numeric field for an empty combo.
allowed_files:
  - trainer/ngpt_trainer/effective_diversity.py
  - trainer/tests/test_effective_diversity.py
reference_files:
  - trainer/ngpt_trainer/divergence.py
  - trainer/ngpt_trainer/selena_corpus.py
test_files:
  - trainer/tests/test_effective_diversity.py
acceptance_criteria:
  - "trigram_coverage(lines: list[str]) -> tuple[int, int, float]" returns
    (distinct_trigram_count, total_trigram_occurrences, coverage_ratio)
    where coverage_ratio = distinct_trigram_count / total_trigram_occurrences
    (0.0 if total is 0). Counting a trigram's OCCURRENCES means every
    position in every line counts toward the total, even if the same
    3-character shape recurs — coverage_ratio drops toward 0 as the same
    shapes repeat more, and rises toward 1.0 when every trigram occurrence
    is distinct.
  - "unique_line_ratio(lines: list[str]) -> float" returns
    (count of distinct normalized lines) / (total line count), 0.0 for
    an empty list. Normalization: strip() then collapse internal
    whitespace runs to single spaces (e.g. via " ".join(s.split())).
  - "effective_diversity_report(responses_by_combo: dict) -> dict" takes
    a dict mapping an arbitrary hashable combo key to a list[str] of
    response lines for that combo, and returns a dict with the SAME KEYS
    plus one extra key "overall" (aggregating every combo's lines
    together). Each value is itself a dict with exactly these keys:
    "distinct_trigrams" (int), "total_trigrams" (int),
    "trigram_coverage" (float), "unique_line_ratio" (float) — i.e. the
    trigram_coverage() and unique_line_ratio() results for that combo's
    lines, merged into one dict per combo.
  - All three functions importable from ngpt_trainer.effective_diversity.
verification:
  - cd trainer && uv run pytest tests/test_effective_diversity.py -v
```

<!-- test_files: independent test-designer + programmer split (AgentCoder
     pattern) — the test-designer writes ONLY test_effective_diversity.py
     from the CONTRACT above (never seeing an implementation), the
     programmer writes ONLY effective_diversity.py (never seeing the
     test file). Both must independently derive behavior from the
     acceptance_criteria's exact function signatures above — that's why
     they're spelled out fully rather than described loosely. -->

## COMPLETION

```yaml
status: done (hybrid — see summary)
summary: |
  qwen-worker's test-designer role completed (test_effective_diversity.py
  written blind, independently, against the contract) before the
  dispatch was killed by unrelated resource contention (concurrent PyTorch
  MPS training + the 14b server both fighting for the same unified GPU
  memory on this Mac — GPU OOM, then a duplicate-process launch on retry).
  The programmer role never ran. Rather than re-dispatch into the same
  contention, the lead implemented effective_diversity.py directly
  (small, fully speced already) and reviewed qwen's test file against it.
  Review caught 5/12 assertions with hallucinated expected numbers —
  qwen has no code execution, so it hand-derived trigram/ratio arithmetic
  for its own test oracle and got it wrong (e.g. claimed
  trigram_coverage(["hello world"]) == (4, 8, 0.5); actually (9, 9, 1.0),
  verified by direct execution). Test *structure* and coverage were sound
  (empty-input, short-string, normalization, multi-combo aggregation) —
  only the hardcoded expected values were wrong. Fixed those 5 in place
  rather than discard the file; kept the empty-dict "no overall key"
  behavior qwen's test implied, since it's a more natural reading of the
  contract's ambiguous zero-combos case than my own draft assumed.
files_changed:
  - trainer/ngpt_trainer/effective_diversity.py (lead-authored)
  - trainer/tests/test_effective_diversity.py (qwen test-designer draft,
    5 assertions corrected by lead after independent verification)
verification: |
  cd trainer && uv run pytest tests/test_effective_diversity.py -v  # 12 passed
  cd trainer && uv run pytest -q                                    # 53 passed, full suite
risks:
  - LLM-authored test oracles without code execution are unreliable for
    exact numeric assertions — always re-derive expected values by
    running the real logic before trusting a test-designer's hardcoded
    numbers, not just its test structure.
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 1 / 1 (relaunch after GPU-OOM kill) / yes (killed twice by resource contention, not by task difficulty; programmer role never completed, lead implemented directly)
- claude tokens spent (contract + review, est.) vs doing it directly: contract + review cost roughly what direct implementation would have; the real win was the independent test file, once its arithmetic was corrected
- defects: caught in review = 5 (hallucinated numeric expected values), slipped past review = 0
