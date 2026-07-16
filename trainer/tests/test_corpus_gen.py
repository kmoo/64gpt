"""Corpus-generator invariants (contract 004, amended per_combo=1200).

These are the properties training relies on; if one breaks, the corpus
silently degrades and the 100K model trains on garbage."""
import pytest
from ngpt_trainer import corpus
from ngpt_trainer.corpus_gen import generate_pairs, corpus_text

COMBOS = [(n, m, e) for n in corpus.NPCS for m in corpus.MOODS
          for e in corpus.EVENTS]


def test_deterministic_and_seed_sensitive():
    assert generate_pairs(seed=7, per_combo=50) == generate_pairs(seed=7, per_combo=50)
    assert generate_pairs(seed=7, per_combo=50) != generate_pairs(seed=8, per_combo=50)


def test_coverage_and_interleaving():
    pairs = generate_pairs(seed=0, per_combo=25)
    assert len(pairs) == 12 * 25
    # interleaved: every window of 12 covers all combos, in cycle order
    expected = [corpus.prompt_for(n, m, e) for n, m, e in COMBOS]
    for i in range(0, len(pairs), 12):
        assert [p for p, _ in pairs[i:i + 12]] == expected


def test_prompts_use_m3_protocol():
    pairs = generate_pairs(seed=0, per_combo=2)
    prompts = {p for p, _ in pairs}
    assert prompts == {corpus.prompt_for(n, m, e) for n, m, e in COMBOS}


def test_distinct_responses_per_combo():
    pairs = generate_pairs(seed=0, per_combo=400)
    for n, m, e in COMBOS:
        want = corpus.prompt_for(n, m, e)
        distinct = {r for p, r in pairs if p == want}
        assert len(distinct) >= 200, (n, m, e, len(distinct))


def test_response_shape():
    for _, r in generate_pairs(seed=0, per_combo=400):
        assert 8 <= len(r) <= 120
        assert "\n" not in r
        assert r == r.upper()  # N64 debug font: no lowercase


def test_charset_printable_ascii():
    text = corpus_text(seed=0, per_combo=200)
    assert all(32 <= ord(c) <= 126 for c in text)


def test_default_corpus_size():
    text = corpus_text()  # seed=0, per_combo=1200
    assert 1_000_000 <= len(text) <= 2_500_000, len(text)
