"""Tests for korrath_corpus.py -- M10's mid-tier talking boss, same
discipline as test_shadewrath_corpus.py."""
from ngpt_trainer import korrath_corpus as kc


def test_prompt_matches_context_builder_format():
    p = kc.prompt_for(1, "sassy", "greeting", "none")
    assert p == "N:korrath TR:1 M:sassy C:greeting EV:none|"


def test_prompt_defaults_missing_event_to_none():
    p = kc.prompt_for(0, "cheerful", "farewell", "")
    assert p.endswith("EV:none|")


def test_generate_pairs_deterministic_with_same_seed():
    a = kc.generate_pairs(seed=0)
    b = kc.generate_pairs(seed=0)
    assert a == b


def test_generate_pairs_covers_full_grid():
    pairs = kc.generate_pairs(seed=0, per_combo=1)
    assert len(pairs) == len(kc.TRUST_TIERS) * len(kc.MOODS) * len(kc.CONTEXTS)


def test_all_responses_non_empty_and_reasonable_length():
    pairs = kc.generate_pairs(seed=0)
    for prompt, response in pairs:
        assert 1 <= len(response) <= 300, f"{prompt!r} -> {response!r}"


def test_every_mood_and_context_produces_distinct_content():
    assert len(kc._BODIES) == len(kc.CONTEXTS)
    assert len(kc._OPENERS) == len(kc.MOODS)
    for context, bodies in kc._BODIES.items():
        assert len(set(bodies)) == len(bodies), f"{context}: duplicate body line"
    for mood, openers in kc._OPENERS.items():
        assert len(set(openers)) == len(openers), f"{mood}: duplicate opener line"


def test_closers_escalate_by_trust_tier_and_dont_repeat_across_tiers():
    all_closers = [c for tier in kc.TRUST_TIERS for c in kc._CLOSERS[tier]]
    assert len(set(all_closers)) == len(all_closers), (
        "a closer line is reused across trust tiers, flattening the arc")


def test_voice_is_distinct_from_shadewrath():
    # Korrath and Shadewrath are both on the old N: scheme with similar
    # structural shape -- guard against accidentally sharing content
    # between the two (would be a real voice-bleed bug, not a feature).
    from ngpt_trainer import shadewrath_corpus as swc
    korrath_lines = set()
    for bank in kc._OPENERS.values():
        korrath_lines.update(bank)
    for bank in kc._BODIES.values():
        korrath_lines.update(bank)
    for bank in kc._CLOSERS.values():
        korrath_lines.update(bank)
    shadewrath_lines = set()
    for bank in swc._OPENERS.values():
        shadewrath_lines.update(bank)
    for bank in swc._BODIES.values():
        shadewrath_lines.update(bank)
    for bank in swc._CLOSERS.values():
        shadewrath_lines.update(bank)
    shadewrath_lines.update(swc._SHADEWRATH_CATCHPHRASES)
    assert korrath_lines.isdisjoint(shadewrath_lines)


def test_density_is_smaller_than_shadewrath_matching_mid_tier():
    # Mid tier means less density than full tier, not parity with it
    # (docs/milestones/m10.md's tier table) -- verify this stays true
    # rather than silently drifting as content gets edited.
    from ngpt_trainer import shadewrath_corpus as swc
    korrath_chars = len(kc.corpus_text(seed=0, per_combo=4))
    shadewrath_chars = len(swc.corpus_text(seed=0, per_combo=8))
    assert korrath_chars < shadewrath_chars
