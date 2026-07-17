"""Tests for llm_judge.py's pure logic (duplicate detection, prompt
building, response parsing) -- the parts that don't require a live
opencoder dispatch. See docs/milestones/m9.md section 5."""
import json

from llm_judge import (
    _JSON_ARRAY_RE,
    build_judge_prompt,
    find_near_duplicates,
    persona_key,
    persona_label,
)

_SELENA_PROFILE = {"occupation": "villager", "age": 12, "gender": "female",
                    "traits": {"warmth": 90, "humor": 85, "impulsivity": 70,
                               "bravery": 55, "focus": 30}}


def _entry(line, persona=None, mood="cheerful", context="greeting", tier="best_friend"):
    return {"persona": persona or _SELENA_PROFILE, "mood": mood,
            "context": context, "tier": tier, "line": line}


def test_persona_key_groups_identical_profiles_together():
    a = _entry("HELLO THERE!")
    b = _entry("A DIFFERENT LINE.")
    assert persona_key(a["persona"]) == persona_key(b["persona"])


def test_persona_key_distinguishes_different_profiles():
    other = dict(_SELENA_PROFILE, occupation="guard")
    a = _entry("X", persona=_SELENA_PROFILE)
    b = _entry("X", persona=other)
    assert persona_key(a["persona"]) != persona_key(b["persona"])


def test_persona_label_format():
    label = persona_label(_SELENA_PROFILE)
    assert "sassy" in label
    assert "12-year-old" in label
    assert "girl" in label
    assert "villager" in label


def test_find_near_duplicates_flags_identical_lines_same_persona():
    corpus = [_entry("WATCH THE GOBLIN, IT'S FAST."),
              _entry("WATCH THE GOBLIN, IT'S FAST.")]
    dups = find_near_duplicates(corpus)
    assert len(dups) == 1
    assert dups[0]["distance"] == 0.0


def test_find_near_duplicates_ignores_different_personas():
    other = dict(_SELENA_PROFILE, occupation="guard")
    corpus = [_entry("SAME LINE HERE.", persona=_SELENA_PROFILE),
              _entry("SAME LINE HERE.", persona=other)]
    dups = find_near_duplicates(corpus)
    assert dups == []


def test_find_near_duplicates_ignores_genuinely_different_lines():
    corpus = [_entry("THE WEATHER IS LOVELY TODAY."),
              _entry("GET AWAY FROM THAT COLLAPSING BRIDGE!")]
    dups = find_near_duplicates(corpus)
    assert dups == []


def test_build_judge_prompt_includes_every_line_and_persona_context():
    batch = [_entry("A LINE ABOUT GOBLINS."), _entry("A SECOND LINE.")]
    prompt = build_judge_prompt(batch)
    assert "A LINE ABOUT GOBLINS." in prompt
    assert "A SECOND LINE." in prompt
    assert "sassy" in prompt  # persona descriptor shows up in context
    assert "1." in prompt and "2." in prompt
    assert "JSON array" in prompt


def test_json_array_regex_extracts_array_from_noisy_response():
    noisy = 'Sure, here is the result:\n[{"i":1,"coherence":4,"voice":3}]\nHope that helps!'
    m = _JSON_ARRAY_RE.search(noisy)
    assert m is not None
    parsed = json.loads(m.group(0))
    assert parsed == [{"i": 1, "coherence": 4, "voice": 3}]


def test_json_array_regex_none_when_no_array_present():
    assert _JSON_ARRAY_RE.search("no json here at all") is None
