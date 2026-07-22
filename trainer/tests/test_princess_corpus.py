"""Tests for princess_corpus.py -- M11's rescued elf princess (Elowen),
same discipline as test_korrath_corpus.py/test_shadewrath_corpus.py."""
from ngpt_trainer import korrath_corpus as kc
from ngpt_trainer import princess_corpus as pc
from ngpt_trainer import shadewrath_corpus as swc
from ngpt_trainer.ravendale_lore import RAVENDALE_LORE


def test_prompt_matches_context_builder_format():
    p = pc.prompt_for(1, "sassy", "greeting", "none")
    assert p == "N:elowen TR:1 M:sassy C:greeting EV:none|"


def test_prompt_defaults_missing_event_to_none():
    p = pc.prompt_for(0, "cheerful", "farewell", "")
    assert p.endswith("EV:none|")


def test_generate_pairs_deterministic_with_same_seed():
    a = pc.generate_pairs(seed=0)
    b = pc.generate_pairs(seed=0)
    assert a == b


def test_generate_pairs_covers_full_grid():
    pairs = pc.generate_pairs(seed=0, per_combo=1)
    assert len(pairs) == len(pc.TRUST_TIERS) * len(pc.MOODS) * len(pc.CONTEXTS)


def test_all_responses_non_empty_and_reasonable_length():
    pairs = pc.generate_pairs(seed=0)
    for prompt, response in pairs:
        assert 1 <= len(response) <= 300, f"{prompt!r} -> {response!r}"


def test_every_mood_and_context_produces_distinct_content():
    assert len(pc._BODIES) == len(pc.CONTEXTS)
    assert len(pc._OPENERS) == len(pc.MOODS)
    for context, bodies in pc._BODIES.items():
        assert len(set(bodies)) == len(bodies), f"{context}: duplicate body line"
    for mood, openers in pc._OPENERS.items():
        assert len(set(openers)) == len(openers), f"{mood}: duplicate opener line"


def test_closers_escalate_by_trust_tier_and_dont_repeat_across_tiers():
    all_closers = [c for tier in pc.TRUST_TIERS for c in pc._CLOSERS[tier]]
    assert len(set(all_closers)) == len(all_closers), (
        "a closer line is reused across trust tiers, flattening the arc")


def test_voice_is_distinct_from_shadewrath_and_korrath():
    # All three are on the old N: scheme with similar structural shape --
    # guard against accidentally sharing bespoke voice content (would be
    # a real voice-bleed bug, not a feature). The shared RAVENDALE_LORE
    # bank is the ONE deliberate exception -- excluded from this check.
    princess_lines = set()
    for bank in pc._OPENERS.values():
        princess_lines.update(bank)
    for bank in pc._BODIES.values():
        princess_lines.update(bank)
    for bank in pc._CLOSERS.values():
        princess_lines.update(bank)

    other_lines = set()
    for bank in kc._OPENERS.values():
        other_lines.update(bank)
    for bank in kc._BODIES.values():
        other_lines.update(bank)
    for bank in kc._CLOSERS.values():
        other_lines.update(bank)
    for bank in swc._OPENERS.values():
        other_lines.update(bank)
    for bank in swc._BODIES.values():
        other_lines.update(bank)
    for bank in swc._CLOSERS.values():
        other_lines.update(bank)
    other_lines.update(swc._SHADEWRATH_CATCHPHRASES)

    assert princess_lines.isdisjoint(other_lines)


def test_density_is_comparable_to_korrath_matching_mid_tier():
    # Both are mid tier -- same per_combo default, comparable size, not
    # parity with Shadewrath's full-tier density.
    princess_chars = len(pc.corpus_text(seed=0, per_combo=4))
    shadewrath_chars = len(swc.corpus_text(seed=0, per_combo=8))
    assert princess_chars < shadewrath_chars


def test_shared_ravendale_lore_bank_reused_by_all_three():
    # The M11 quality-push lever (docs/plan.md Known follow-ups): all
    # three narratively-linked characters draw from the SAME lore bank,
    # not their own copies -- verify import identity, not just similar
    # content, so a future edit to ravendale_lore.py actually propagates.
    from ngpt_trainer import shadewrath_corpus as swc_mod
    from ngpt_trainer import korrath_corpus as kc_mod
    assert swc_mod.RAVENDALE_LORE is RAVENDALE_LORE
    assert kc_mod.RAVENDALE_LORE is RAVENDALE_LORE
    assert pc.RAVENDALE_LORE is RAVENDALE_LORE


def test_ravendale_lore_lines_actually_appear_in_generated_output():
    # Not just wired in principle -- confirm at least one RAVENDALE_LORE
    # line surfaces in each character's generated corpus at a real
    # sample size, proving the splice point in _response() actually
    # fires, not just that the import exists.
    for corpus_module, kwargs in (
        (pc, {"per_combo": 4}),
        (kc, {"per_combo": 4}),
        (swc, {"per_combo": 8}),
    ):
        pairs = corpus_module.generate_pairs(seed=0, **kwargs)
        found = any(
            any(line in response for line in RAVENDALE_LORE)
            for _, response in pairs
        )
        assert found, f"{corpus_module.__name__}: no RAVENDALE_LORE line found in sample"
