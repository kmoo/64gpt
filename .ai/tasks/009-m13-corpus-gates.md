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
  text trained badly due to low repetition/reuse density. This task
  creates a new standalone module with its own copy of the
  tokenization/vocab logic (no import dependency on file-specific corpus
  builders) so it can be called generically by any future corpus
  generator or gate-checking script. The existing pattern this task's
  tokenization must match EXACTLY (from trainer/make_m12_1_blob.py,
  which this task does not import from or modify) is:

    import re
    _WORD_RE = re.compile(r"[A-Z']+")

    def build_corpus_vocab(*texts: str) -> set[str]:
        vocab = set()
        for text in texts:
            vocab.update(_WORD_RE.findall(text.upper()))
        return vocab

    def invented_word_count(response: str, corpus_vocab: set[str]) -> int:
        words = _WORD_RE.findall(response.upper())
        return sum(1 for w in words if w not in corpus_vocab and len(w) > 1)
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
test_files:
  - trainer/tests/test_corpus_gates.py
acceptance_criteria:
  - >
    trainer/tests/test_corpus_gates.py MUST import the module under test
    with exactly `from ngpt_trainer.corpus_gates import ...` (NOT
    `from corpus_gates import ...`, which will fail with
    ModuleNotFoundError -- ngpt_trainer is the installed package name,
    trainer/tests/test_overfit.py and test_conditioning.py already use
    this same `from ngpt_trainer.<module> import ...` pattern).
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
status: done
summary: |
  4 dispatch attempts, 3 escalated for real reasons (2 were qwen-worker/
  opencoder tooling bugs found and fixed with TDD along the way; see
  WORKER RESULT sections above). Attempt 4 (Qwen2.5-Coder-7B) produced a
  working structural_gate/density_gate/FragmentUsageTracker/
  build_fragment_vocab module that passed its own verification, but lead
  review caught a real semantic bug the tests never exercised:
  structural_gate deduped invented words through a set before counting,
  so a fragment with ONE invented word repeated 3x scored as "1 invented
  word" instead of matching invented_word_count's occurrence-counting
  definition (a real gate-evasion path for exactly the kind of repetitive
  bad fragment this module exists to catch). Fixed directly + added a
  regression test; all 5 tests pass.
files_changed:
  - trainer/ngpt_trainer/corpus_gates.py
  - trainer/tests/test_corpus_gates.py
verification: |
  cd trainer && uv run pytest tests/test_corpus_gates.py -v
  5 passed in 0.01s
risks: []
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 4 / 3 discarded (2 tooling bugs, 1 real contract gap) / no (done on attempt 4, with one post-hoc lead fix)
- claude tokens spent (contract + review, est.) vs doing it directly: high this time -- 2 real infra bugs in ~/bin/opencoder and ~/bin/qwen-worker had to be found, TDD'd, and fixed before the pipeline worked at all; those fixes are reusable for tasks 010/011 and beyond, so the cost isn't sunk to this task alone
- defects: caught in review = 1 (occurrence-vs-distinct invented-word counting), slipped past review = 0

## WORKER RESULT (qwen-worker)

Discarded: this run's 3 verification FAILs were a false signal from a
qwen-worker bug (mini-yaml parser didn't strip quotes from the
`verification:` list item, so the shell tried to exec the literal
quoted string as a command name). Fixed in ~/bin/qwen-worker
(_unquote helper + regression test in --selftest); re-dispatching
against clean state.

## WORKER RESULT (qwen-worker) — attempt 2, discarded

Escalated for a real reason this time: test-designer wrote
`from corpus_gates import ...` instead of `from ngpt_trainer.corpus_gates
import ...`, a ModuleNotFoundError the programmer couldn't fix (test_files
are frozen once applied). Contract now pins the exact import line under
acceptance_criteria. Also switched tier 14b -> 7b before re-dispatch:
the first attempt's verify step (uv run pytest, which imports trainer's
torch dependency into a fresh venv) combined with 14b's 8.7GB GPU-wired
weights drove system free memory down to 5% -- recovered on its own,
no crash, but too close to a known near-OOM incident to keep risking.

```
$ cd trainer && uv run pytest tests/test_corpus_gates.py -v
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- <repo>/trainer/.venv/bin/python
cachedir: .pytest_cache
rootdir: <repo>/trainer
configfile: pyproject.toml
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
_________________ ERROR collecting tests/test_corpus_gates.py __________________
ImportError while importing test module '<repo>/trainer/tests/test_corpus_gates.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Cellar/python@3.12/3.12.13_4/Frameworks/Python.framework/Versions/3.12/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_corpus_gates.py:2: in <module>
    from corpus_gates import build_fragment_vocab, structural_gate, FragmentUsageTracker, density_gate
E   ModuleNotFoundError: No module named 'corpus_gates'
=========================== short test summary info ============================
ERROR tests/test_corpus_gates.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.05s ===============================
```

## WORKER RESULT (qwen-worker) — attempt 3, discarded

Switched model to Qwen2.5-Coder-7B-Instruct-4bit (already cached locally,
no download) after DeepSeek-Coder-6.7b failed the FILE-block output
format twice in a row. Found two more real ~/bin/opencoder /
~/bin/qwen-worker bugs along the way, both fixed with TDD (red/green
confirmed, regression tests added to each tool's --selftest):
(1) opencoder's active_model() let QWEN_MODEL unconditionally outrank
the on-disk state file, so `QWEN_MODEL=... opencoder start` compared
the override against itself and silently skipped restarting a
different already-running model; (2) qwen-worker's check_output()
required an exact `=== END FILE ===` line match, but Qwen2.5-Coder
leaked its `<|im_end|>` stop token onto that same line with no
newline, so a complete, correctly-formed dispatch got misdiagnosed as
"truncated output" and discarded. Both are fixed now; re-dispatching
against clean state.

## WORKER RESULT (qwen-worker)

- status: escalated
- attempt: [test-designer] rejected before apply: no FILE blocks in output
- attempt: [test-designer] rejected before apply: no FILE blocks in output
- verification tail:

```
(escalated at sanity gate, test-designer: no FILE blocks in output)
```

## WORKER RESULT (qwen-worker)

- status: escalated
- attempt: [test-designer] applied 1 file(s)
- attempt: [programmer] rejected before apply: no FILE blocks in output
- attempt: [programmer] rejected before apply: no FILE blocks in output
- verification tail:

```
(escalated at sanity gate, programmer: no FILE blocks in output)
```

## WORKER RESULT (qwen-worker)

- status: escalated
- attempt: [test-designer] rejected before apply: truncated output (unclosed FILE block — hit token limit?)
- attempt: [test-designer] rejected before apply: truncated output (unclosed FILE block — hit token limit?)
- verification tail:

```
(escalated at sanity gate, test-designer: truncated output (unclosed FILE block — hit token limit?))
```

## WORKER RESULT (qwen-worker)

- status: done
- attempt: [test-designer] applied 1 file(s)
- attempt: [programmer] verification PASS
- verification tail:

```
$ cd trainer && uv run pytest tests/test_corpus_gates.py -v
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- <repo>/trainer/.venv/bin/python
cachedir: .pytest_cache
rootdir: <repo>/trainer
configfile: pyproject.toml
collecting ... collected 4 items

tests/test_corpus_gates.py::test_build_fragment_vocab PASSED             [ 25%]
tests/test_corpus_gates.py::test_structural_gate PASSED                  [ 50%]
tests/test_corpus_gates.py::test_fragment_usage_tracker PASSED           [ 75%]
tests/test_corpus_gates.py::test_density_gate PASSED                     [100%]

============================== 4 passed in 0.01s ===============================
```
