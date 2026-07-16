# Task 004 — corpus-generator

## CONTRACT

```yaml
id: 004-corpus-generator
goal: >
  Deterministic M4 corpus generator: ngpt_trainer/corpus_gen.py produces
  a large set of varied (prompt, response) pairs from a template grammar
  covering every NPC x MOOD x EVENT combination, plus pytest proving the
  invariants. This becomes the training data for the ~100K-param model.
background: >
  M3's corpus.py has 12 hand-written pairs with prompt format
  'NPC=<n> MOOD=<m> EV=<e>|' (NPCS=GUARD/MERCHANT/WIZARD,
  MOODS=ANGRY/CALM, EVENTS=THEFT/FESTIVAL — reuse these exact tuples by
  importing from ngpt_trainer.corpus). M4 needs variety: many distinct
  responses per combination, generated from per-NPC voice templates with
  slot fillers, so the model generalizes instead of memorizing.
constraints: |
  - stdlib only (random.Random(seed) for determinism; no numpy/torch).
  - Uppercase printable ASCII 32..126 only, in prompts AND responses
    (the N64 debug font has no lowercase). No newlines inside a response.
  - Responses 8..120 chars. Prompt format EXACTLY as M3 (import
    prompt_for from ngpt_trainer.corpus).
  - Public API:
    generate_pairs(seed: int = 0, per_combo: int = 400) ->
      list[tuple[str, str]]  # deterministic, combos interleaved
    corpus_text(seed: int = 0, per_combo: int = 400) -> str
      # concatenated prompt+response text (vocab source)
  - Grammar: per NPC a distinct voice (guard: commands/threats/watch
    idiom; merchant: wares/prices/coin idiom; wizard: stars/spells/omen
    idiom), modulated by MOOD (angry: exclamations, short; calm: longer,
    measured) and EVENT (theft/festival vocabulary). At least 6 response
    templates per (npc, mood, event) with 2-3 filler slots each drawing
    from lists of 4+ options — enough that per_combo=400 yields >=200
    DISTINCT responses per combo.
allowed_files:
  - trainer/ngpt_trainer/corpus_gen.py
  - trainer/tests/test_corpus_gen.py
acceptance_criteria:
  - test_corpus_gen.py proves: same seed -> byte-identical output;
    different seeds differ; every combo has exactly per_combo pairs;
    >=200 distinct responses per combo at per_combo=400; every char of
    corpus_text() is printable ASCII 32..126; all responses 8..120
    chars; every prompt matches the M3 format via corpus.prompt_for;
    total corpus_text(per_combo=400) length is between 1_000_000 and
    2_500_000 chars (grow templates/fillers if short).
  - Existing suite stays green (scripts/verify.sh).
verification:
  - scripts/verify.sh
```

Exact import lines for the test file (use verbatim):

    import pytest
    from ngpt_trainer import corpus
    from ngpt_trainer.corpus_gen import generate_pairs, corpus_text

## COMPLETION

```yaml
status: pending
```

## METRICS

- dispatches / retries / escalated: 0 / 0 / no
