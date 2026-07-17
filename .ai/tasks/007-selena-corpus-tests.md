# Task 007 — test coverage for selena_corpus.py

## CONTRACT

```yaml
id: 007-selena-corpus-tests
goal: >
  Write pytest tests for the already-implemented, already-committed
  trainer/ngpt_trainer/selena_corpus.py module (M7's corpus generator for
  Selena), which currently has zero test coverage.
background: >
  selena_corpus.py is frozen (already consumed by a successful M7 training
  run — model.bin, golden vectors, and the conditioning-ablation divergence
  table are all already generated from it). This is pure test-writing
  against EXISTING behavior, not TDD-first spec work — read the real
  implementation in reference_files and test what it actually does.
constraints: |
  - Python 3.12, stdlib only + pytest, same as every other
    trainer/tests/test_*.py file.
  - Do not modify trainer/ngpt_trainer/selena_corpus.py itself.
  - Test these specific behaviors (read the real function bodies in
    reference_files before writing assertions -- do not guess signatures):
    1. prompt_for(trust_tier, mood, context, event, npc_id="selena")
       returns the exact frozen format
       f"N:{npc_id} TR:{trust_tier} M:{mood} C:{context} EV:{ev}|"
       where ev is event if event is truthy, else the literal string
       "none". E.g. prompt_for(2, "cheerful", "item-found", "found_gem")
       == "N:selena TR:2 M:cheerful C:item-found EV:found_gem|", and
       prompt_for(0, "worried", "greeting", "") ==
       "N:selena TR:0 M:worried C:greeting EV:none|" (empty event ->
       "none").
    2. combo_key(prompt) parses a prompt string produced by prompt_for()
       back into the tuple (trust_tier: int, mood: str, context: str) --
       verify it round-trips: for any (tier, mood, context, event) inputs,
       combo_key(prompt_for(tier, mood, context, event)) ==
       (tier, mood, context). Test at least 2 different combos.
    3. generate_pairs(seed=0, per_combo=N) is deterministic: calling it
       twice with the same seed and per_combo produces byte-identical
       results (same list of (prompt, response) tuples). Use a small
       per_combo (e.g. 2) to keep the test fast.
    4. generate_pairs returns exactly per_combo * 120 pairs (120 = the
       fixed TRUST_TIERS x MOODS x CONTEXTS grid: 3 x 5 x 8). Verify the
       count directly with a small per_combo (e.g. 1 -> expect 120 pairs,
       or 2 -> expect 240).
    5. Every prompt produced by generate_pairs is parseable by combo_key
       without raising, and the parsed (trust, mood, context) is a member
       of the actual TRUST_TIERS x MOODS x CONTEXTS product (import those
       three tuples from the module rather than hardcoding the values).
    6. generate_thin_identity_pairs(seed=1000, combos_used=20,
       lines_per_combo=12) returns exactly combos_used * lines_per_combo
       pairs, and every prompt in the output uses npc_id=THIN_ID (import
       THIN_ID from the module) rather than the default "selena" --
       check this by asserting THIN_ID appears in the prompt and
       "selena" does not (use the module's own THIN_ID constant, don't
       hardcode "kip").
  - Do not test _fill, _response, _OPENERS, _BODIES, _CLOSERS,
    EVENTS_FOR_CONTEXT directly -- those are private/internal (leading
    underscore or content dicts) and covered indirectly through
    generate_pairs; testing them directly would break on harmless content
    edits.
allowed_files:
  - trainer/tests/test_selena_corpus.py
reference_files:
  - trainer/ngpt_trainer/selena_corpus.py
  - trainer/tests/test_divergence.py
  - trainer/tests/test_corpus_gen.py
verification:
  - cd trainer && uv run pytest tests/test_selena_corpus.py -v
```

## COMPLETION

```yaml
status: done (escalated -> finished by lead)
summary: |
  qwen-worker escalated after 2 attempts, both rejected pre-apply for
  producing no FILE blocks (empty/malformed output) -- second escalation
  in a row with this shape, both under sustained memory pressure from the
  same session (14b server loaded alongside other work). Wrote the test
  file directly against the contract's 6 specified behaviors instead of
  retrying a third time.
files_changed:
  - trainer/tests/test_selena_corpus.py
verification: |
  cd trainer && uv run pytest tests/test_selena_corpus.py -v
  6 passed in 0.01s
risks: []
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 1 / 2 / yes
- claude tokens spent (contract + review, est.) vs doing it directly: contract + full manual write, no net savings this time
- defects: caught in review = 0, slipped past review = 0

## WORKER RESULT (qwen-worker)

- status: escalated
- attempt: rejected before apply: no FILE blocks in output
- attempt: rejected before apply: no FILE blocks in output
- verification tail:

```
(escalated at sanity gate: no FILE blocks in output)
```
