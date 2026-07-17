"""Tests for m9_corpus.py's assembly logic. See docs/milestones/m9.md."""
import pytest

from ngpt_trainer.m9_corpus import (
    _TIER_MIDPOINT,
    _relationship_state,
    assert_no_holdout_leak,
    combo_key,
    generate_pairs,
)
from ngpt_trainer.npc_service import relationship_label

_PROFILE = {"occupation": "guard", "age": 30, "gender": "male",
            "traits": {"warmth": 20, "humor": 20, "impulsivity": 20,
                       "bravery": 80, "focus": 85}}


def _entry(tier="best_friend", occupation=None, traits=None):
    persona = dict(_PROFILE)
    if occupation:
        persona["occupation"] = occupation
    if traits:
        persona["traits"] = traits
    return {"persona": persona, "mood": "cheerful", "context": "greeting",
            "tier": tier, "line": "HOLD THE LINE."}


@pytest.mark.parametrize("tier", list(_TIER_MIDPOINT.keys()))
def test_relationship_state_midpoint_round_trips_to_same_tier(tier):
    state = _relationship_state(tier)
    _, resolved = relationship_label(state)
    assert resolved == tier


def test_generate_pairs_produces_prompt_fields_output():
    pairs = generate_pairs([_entry()])
    assert len(pairs) == 1
    prompt, response = pairs[0]
    assert prompt == "P:man AGE:30 D:gruff OCC:guard R:best_friend M:cheerful C:greeting EV:none|"
    assert response == "HOLD THE LINE."


def test_combo_key_is_occupation_and_descriptor():
    entry = _entry(occupation="wizard")
    assert combo_key(entry) == ("wizard", "gruff")


def test_assert_no_holdout_leak_passes_when_clean(tmp_path, monkeypatch):
    import ngpt_trainer.m9_corpus as m9c
    monkeypatch.setattr(m9c, "holdout_pairs", lambda: [("bandit", "sassy")])
    raw = [_entry(occupation="guard")]  # not the held-out combo
    assert_no_holdout_leak(raw)  # no exception


def test_assert_no_holdout_leak_raises_when_leaked(monkeypatch):
    import ngpt_trainer.m9_corpus as m9c
    monkeypatch.setattr(m9c, "holdout_pairs", lambda: [("guard", "gruff")])
    raw = [_entry(occupation="guard")]  # matches the held-out combo
    with pytest.raises(AssertionError, match="leaked"):
        assert_no_holdout_leak(raw)
