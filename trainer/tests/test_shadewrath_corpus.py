"""Tests for shadewrath_corpus.py -- M10's recurring necromancer villain,
full tier, old N:<id> scheme (ContextBuilder), same discipline as
selena_corpus's own tests."""
from ngpt_trainer import shadewrath_corpus as sw


def test_prompt_matches_context_builder_format():
    p = sw.prompt_for(1, "sassy", "greeting", "none")
    assert p == "N:shadewrath TR:1 M:sassy C:greeting EV:none|"


def test_prompt_defaults_missing_event_to_none():
    p = sw.prompt_for(0, "cheerful", "farewell", "")
    assert p.endswith("EV:none|")


def test_generate_pairs_deterministic_with_same_seed():
    a = sw.generate_pairs(seed=0)
    b = sw.generate_pairs(seed=0)
    assert a == b


def test_generate_pairs_covers_full_grid():
    pairs = sw.generate_pairs(seed=0, per_combo=1)
    assert len(pairs) == len(sw.TRUST_TIERS) * len(sw.MOODS) * len(sw.CONTEXTS)


def test_all_responses_non_empty_and_reasonable_length():
    pairs = sw.generate_pairs(seed=0)
    for prompt, response in pairs:
        assert 1 <= len(response) <= 300, f"{prompt!r} -> {response!r}"


def test_every_mood_and_context_produces_distinct_content():
    # Catches an accidental copy-paste where two moods/contexts share a
    # bank -- every _BODIES/_OPENERS key must actually differ from every
    # other, not just exist.
    assert len(sw._BODIES) == len(sw.CONTEXTS)
    assert len(sw._OPENERS) == len(sw.MOODS)
    for context, bodies in sw._BODIES.items():
        assert len(set(bodies)) == len(bodies), f"{context}: duplicate body line"
    for mood, openers in sw._OPENERS.items():
        assert len(set(openers)) == len(openers), f"{mood}: duplicate opener line"


def test_closers_escalate_by_trust_tier_and_dont_repeat_across_tiers():
    # The whole point of the trust-tier arc: tier 2 reveals the alliance
    # offer, which must not leak into tier 0/1 (would flatten the arc).
    all_closers = [c for tier in sw.TRUST_TIERS for c in sw._CLOSERS[tier]]
    assert len(set(all_closers)) == len(all_closers), (
        "a closer line is reused across trust tiers, flattening the arc")


def test_density_in_guard_benchmark_ballpark():
    # ~123K chars/instance is the proven-working precedent (docs/
    # milestones/m9.md section 4); Shadewrath is bespoke content authored
    # at a similar scale, not a thin per-combo=3 draw.
    text = sw.corpus_text(seed=0, per_combo=8)
    assert 80_000 <= len(text) <= 220_000, f"{len(text)} chars"
