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
status: done (escalated)
contract_amendment: >
  The original acceptance criteria were internally inconsistent: 12
  combos x 400 pairs x <=120-char responses caps at ~734K chars, below
  the required 1M. Amended: default per_combo=1200 (the >=200-distinct
  criterion still holds at per_combo=400 and is tested there).
summary: >
  qwen-worker made 3 attempts, all failed verification: fixed
  whole-sentence templates (6 distinct per combo, no filler slots),
  mixed-case text, newline joins, and it reimplemented prompt_for
  instead of importing. Lead rewrote both files: opener/body/closer
  grammar with 2-slot bodies, mood-modulated punctuation, per-NPC
  voices. Shipped: 1,503,198 chars at default seed, charset 34,
  762-1129 distinct responses per combo.
verification: scripts/verify.sh — host 4/4, trainer 34 passed, PASS
lesson: >
  Multi-constraint generative-grammar contracts exceed the local
  model's ceiling even at 14b: it satisfies the API shape and drops
  the combinatorial requirements. Contracts must also be checked for
  arithmetic consistency BEFORE dispatch — an unsatisfiable criterion
  burns all retries.
```

## METRICS

- dispatches / retries / escalated: 3 / 2 / yes

## WORKER RESULT (qwen-worker)

- status: escalated
- attempt: verification FAIL
- attempt: verification FAIL
- attempt: verification FAIL
- verification tail:

```
$ scripts/verify.sh
# Review package — 2026-07-15, HEAD 4f7202c

## Working tree
?? docs/milestones/m4.md
?? trainer/ngpt_trainer/corpus_gen.py
?? trainer/tests/test_corpus_gen.py

## Host tests (core/ integer engine)
Test project ~/GitHub/64gpt/build
    Start 1: test_blob_parser
1/4 Test #1: test_blob_parser .................   Passed    0.23 sec
    Start 2: test_canned_model
2/4 Test #2: test_canned_model ................   Passed    0.11 sec
    Start 3: test_gru_model
3/4 Test #3: test_gru_model ...................   Passed    0.11 sec
    Start 4: test_prompted_model
4/4 Test #4: test_prompted_model ..............   Passed    0.14 sec

100% tests passed out of 4

Total Test time (real) =   0.60 sec
HOST TESTS: PASS

## Trainer tests (pytest via uv)
F..F............................                                         [100%]
=================================== FAILURES ===================================
________________________ TestCorpusGen.test_corpus_text ________________________

self = <test_corpus_gen.TestCorpusGen testMethod=test_corpus_text>

    def test_corpus_text(self):
        text = corpus_text()
>       self.assertTrue(all(32 <= ord(c) <= 126 for c in text))
E       AssertionError: False is not true

tests/test_corpus_gen.py:34: AssertionError
____________________ TestCorpusGen.test_distinct_responses _____________________

self = <test_corpus_gen.TestCorpusGen testMethod=test_distinct_responses>

    def test_distinct_responses(self):
        seed = 42
        per_combo = 400
        pairs = generate_pairs(seed, per_combo)
        for npc in NPCS:
            for mood in MOODS:
                for event in EVENTS:
                    combo_pairs = [pair for pair in pairs if pair[0].startswith(f"NPC={npc} MOOD={mood} EV={event}|")]
>                   self.assertGreaterEqual(len(set(pair[1] for pair in combo_pairs)), 200)
E                   AssertionError: 6 not greater than or equal to 200

tests/test_corpus_gen.py:30: AssertionError
=========================== short test summary info ============================
FAILED tests/test_corpus_gen.py::TestCorpusGen::test_corpus_text - AssertionE...
FAILED tests/test_corpus_gen.py::TestCorpusGen::test_distinct_responses - Ass...
2 failed, 30 passed, 3 deselected in 3.63s
TRAINER TESTS: FAIL

VERDICT: FAIL
```
