"""Tests for cast_corpus.py -- M9 attempt #2's template-grammar generator
for the curated named cast (Bram/Fergus/Kragan). See docs/milestones/
m9.md section 4 for why attempt #1 (freeform LLM per-persona) failed and
why this design replaces it."""
import pytest

from ngpt_trainer.cast_corpus import (
    CHARACTERS,
    HOLDOUT_COMBOS,
    _DESCRIPTOR_TICS,
    assert_no_holdout_leak,
    combo_key,
    corpus_text,
    generate_pairs,
    holdout_pairs,
)
from ngpt_trainer.npc_service import personality_descriptor

_CANON_DESCRIPTOR = {"guard": "gruff", "innkeeper": "cheerful", "bandit": "cold"}


def test_character_traits_calibrate_to_intended_descriptor():
    for name, profile in CHARACTERS.items():
        expected = _CANON_DESCRIPTOR[profile["occupation"]]
        assert personality_descriptor(profile["traits"]) == expected, (
            f"{name}'s traits map to the wrong descriptor")


def test_generate_pairs_deterministic_with_same_seed():
    a = generate_pairs(seed=0)
    b = generate_pairs(seed=0)
    assert a == b


def test_generate_pairs_different_seed_differs():
    a = generate_pairs(seed=0)
    b = generate_pairs(seed=1)
    assert a != b


def test_every_prompt_token_has_a_colon():
    pairs = generate_pairs(seed=0)
    for prompt, _ in pairs:
        for tok in prompt.rstrip("|").split(" "):
            assert ":" in tok, f"bare token {tok!r} in {prompt!r}"


def test_combo_key_round_trips():
    pairs = generate_pairs(seed=0)
    prompt, _ = pairs[0]
    person, descriptor, occupation, tier, mood, context = combo_key(prompt)
    assert f"P:{person} " in prompt
    assert f"D:{descriptor} " in prompt
    assert f"OCC:{occupation} " in prompt
    assert f"R:{tier} " in prompt
    assert f"M:{mood} " in prompt
    assert f"C:{context} " in prompt


def test_density_per_character_matches_guard_benchmark():
    # Guard's own proven-working precedent (docs/milestones/m9.md section
    # 4): ~123K chars/instance. Each named character here should land in
    # the same ballpark, not attempt #1's ~1,300 chars/persona.
    pairs = generate_pairs(seed=0)
    per_char_chars: dict[str, int] = {}
    for prompt, response in pairs:
        occ = prompt.split("OCC:")[1].split(" ")[0]
        per_char_chars[occ] = per_char_chars.get(occ, 0) + len(prompt) + len(response)
    assert set(per_char_chars) == {"guard", "innkeeper", "bandit"}
    for occ, chars in per_char_chars.items():
        assert 80_000 <= chars <= 200_000, (
            f"{occ}: {chars} chars is far from guard's ~123K benchmark")


def test_axis_crossing_produces_internally_consistent_examples():
    # A crossed line's D: must match a REAL descriptor value (not the
    # character's canonical one for every line) -- proving OCC:/D: are
    # exercised as independent axes, not perfectly correlated.
    pairs = generate_pairs(seed=0, cross_fraction=0.2)
    crossed = 0
    for prompt, _ in pairs:
        occ = prompt.split("OCC:")[1].split(" ")[0]
        d = prompt.split("D:")[1].split(" ")[0]
        canon = _CANON_DESCRIPTOR.get(occ)
        if canon and d != canon:
            crossed += 1
            assert d in _DESCRIPTOR_TICS
    assert crossed > 0, "no axis-crossing examples were generated"


def test_zero_cross_fraction_never_crosses():
    pairs = generate_pairs(seed=0, cross_fraction=0.0)
    for prompt, _ in pairs:
        occ = prompt.split("OCC:")[1].split(" ")[0]
        d = prompt.split("D:")[1].split(" ")[0]
        canon = _CANON_DESCRIPTOR.get(occ)
        if canon:
            assert d == canon


def test_corpus_text_matches_generate_pairs():
    pairs = generate_pairs(seed=0, per_combo=1)
    text = corpus_text(seed=0, per_combo=1)
    assert text == "".join(p + r for p, r in pairs)


def test_holdout_combos_never_appear_in_generated_corpus():
    pairs = generate_pairs(seed=0)
    assert_no_holdout_leak(pairs)  # no exception
    for prompt, _ in pairs:
        occ = prompt.split("OCC:")[1].split(" ")[0]
        d = prompt.split("D:")[1].split(" ")[0]
        assert (occ, d) not in HOLDOUT_COMBOS


def test_assert_no_holdout_leak_raises_when_leaked():
    occ, d = sorted(HOLDOUT_COMBOS)[0]
    fake_pairs = [(f"P:man D:{d} OCC:{occ} R:friend M:cheerful C:greeting EV:none|", "HI.")]
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_holdout_leak(fake_pairs)


def test_holdout_pairs_returns_sorted_list():
    assert holdout_pairs() == sorted(HOLDOUT_COMBOS)
