"""Tests for shadewrath_corpus.py -- M10's recurring necromancer villain,
full tier, genericized M11.1 onto NpcService's compositional scheme
(docs/milestones/m11.1.md Part 1), same discipline as selena_corpus's
own tests."""
from ngpt_trainer import shadewrath_corpus as sw


def test_prompt_matches_npc_service_format():
    p = sw.prompt_for(1, "sassy", "greeting", "none")
    assert p == ("P:man D:gruff OCC:villain SPECIES:shade R:neutral "
                 "BOND:rival M:sassy C:greeting AUD:witnessed EV:none|")


def test_prompt_defaults_missing_event_to_none():
    p = sw.prompt_for(0, "cheerful", "farewell", "")
    assert p.endswith("EV:none|")


def test_prompt_alone_moods_get_aud_alone():
    # tender/embarrassed are his private-register moods -- see module header.
    assert "AUD:alone" in sw.prompt_for(2, "tender", "quiet-moment", "")
    assert "AUD:witnessed" in sw.prompt_for(2, "cheerful", "greeting", "")


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


def test_lore_bank_toggle_is_rng_isolated():
    # M11.1 Part 2: disabling lore_bank_enabled must change ONLY whether
    # the lore clause appears -- prompts and every non-lore response must
    # be byte-identical, or a baseline/treatment retrain wouldn't be an
    # isolated-variable comparison.
    enabled = sw.generate_pairs(seed=0, per_combo=2, lore_bank_enabled=True)
    disabled = sw.generate_pairs(seed=0, per_combo=2, lore_bank_enabled=False)
    assert len(enabled) == len(disabled)
    assert all(p1 == p2 for (p1, _), (p2, _) in zip(enabled, disabled)), (
        "prompts must be identical regardless of lore_bank_enabled")
    assert not any(any(line in r for line in sw.RAVENDALE_LORE) for _, r in disabled), (
        "lore_bank_enabled=False must never produce a lore line")
    diffs = [(r1, r2) for (_, r1), (_, r2) in zip(enabled, disabled) if r1 != r2]
    assert diffs, "expected at least one lore-bearing response to differ"
    for r1, r2 in diffs:
        assert any(line in r1 for line in sw.RAVENDALE_LORE), (
            "every diff must be explained by a lore line, not RNG drift")


def test_density_in_guard_benchmark_ballpark():
    # ~123K chars/instance is the proven-working precedent (docs/
    # milestones/m9.md section 4); Shadewrath is bespoke content authored
    # at a similar scale, not a thin per-combo=3 draw.
    text = sw.corpus_text(seed=0, per_combo=8)
    assert 80_000 <= len(text) <= 220_000, f"{len(text)} chars"
