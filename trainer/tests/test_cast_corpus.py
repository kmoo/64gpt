"""Tests for cast_corpus.py -- M9 attempt #2's template-grammar generator
for the curated named cast (Bram/Fergus/Kragan). See docs/milestones/
m9.md section 4 for why attempt #1 (freeform LLM per-persona) failed and
why this design replaces it."""
import pytest

from ngpt_trainer.cast_corpus import (
    CHARACTERS,
    GOSSIP_EVENTS,
    GOSSIP_HUB_OCCUPATIONS,
    HOLDOUT_COMBOS,
    _CATCHPHRASES,
    _DESCRIPTOR_TICS,
    _FERGUS_CATCHPHRASES,
    _GOSSIP_LINES,
    _KRAGAN_CATCHPHRASES,
    assert_no_holdout_leak,
    combo_key,
    corpus_text,
    generate_pairs,
    holdout_pairs,
)
from ngpt_trainer.npc_service import personality_descriptor

_CANON_DESCRIPTOR = {
    "guard": "gruff", "innkeeper": "cheerful", "bandit": "cold",
    # M10 town-archetype representatives (pub_patron/blacksmith/wizard/
    # villager keyed by occupation the same as the original three).
    "pub_patron": "cheerful", "blacksmith": "gruff",
    "wizard": "measured", "villager": "playful",
}


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
    assert set(per_char_chars) == {
        "guard", "innkeeper", "bandit",
        "pub_patron", "blacksmith", "wizard", "villager",
    }
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


def test_combo_count_narrows_to_a_seeded_subset():
    # m9.1 density-structure experiment: combo_count should restrict each
    # character to that many DISTINCT combos (repeated per_combo times
    # each), not just cap total pairs.
    pairs = generate_pairs(seed=0, combo_count=10, per_combo=5)
    by_char: dict[str, set] = {}
    for prompt, _ in pairs:
        occ = prompt.split("OCC:")[1].split(" ")[0]
        tier, mood, ctx = combo_key(prompt)[3:]
        by_char.setdefault(occ, set()).add((tier, mood, ctx))
    for occ, combos in by_char.items():
        assert len(combos) == 10, f"{occ}: expected 10 distinct combos, got {len(combos)}"


def test_combo_count_deterministic_with_same_seed():
    a = generate_pairs(seed=0, combo_count=10, per_combo=5)
    b = generate_pairs(seed=0, combo_count=10, per_combo=5)
    assert a == b


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


def test_kragan_catchphrases_appear_in_generated_corpus():
    # M9.2: Kragan gets his own small catchphrase bank, same mechanism as
    # Fergus's (docs/milestones/m9.2.md) -- targets the coherence gap M9
    # flagged live on hardware for Kragan specifically.
    pairs = generate_pairs(seed=0)
    kragan_responses = [r for p, r in pairs if "OCC:bandit" in p]
    assert any(any(cp in r for cp in _KRAGAN_CATCHPHRASES) for r in kragan_responses), (
        "no Kragan catchphrase found across the generated corpus")


def test_catchphrase_banks_are_disjoint_per_character():
    assert set(_FERGUS_CATCHPHRASES).isdisjoint(_KRAGAN_CATCHPHRASES)
    assert set(_CATCHPHRASES) == {"fergus", "kragan"}


def test_holdout_pairs_returns_sorted_list():
    assert holdout_pairs() == sorted(HOLDOUT_COMBOS)


# ---- M11 gossip (docs/milestones/m11.md section 2) -----------------------

def test_gossip_events_match_worldstate_cpp():
    # MUST match game/src/user/WorldState.cpp's GOSSIP_EVENTS exactly,
    # same order -- these are the only EV: tags a gossip-hub occupation
    # was ever shown in training, and WorldState.cpp is the runtime side
    # that actually publishes them (no automated cross-check possible
    # across the language boundary, same limitation test_npc_service.cpp's
    # own header comment notes for its Python/C++ parity -- pinned by
    # literal value here instead, same discipline test_guard_instances.py
    # uses for its EXPECTED dict).
    assert GOSSIP_EVENTS == ("shadewrath_allied", "korrath_pleaded")


def test_gossip_hub_occupations_are_pub_patron_and_villager():
    assert GOSSIP_HUB_OCCUPATIONS == {"pub_patron", "villager"}


def test_gossip_lines_defined_for_every_gossip_event():
    assert set(_GOSSIP_LINES) == set(GOSSIP_EVENTS)
    for tag, lines in _GOSSIP_LINES.items():
        assert len(lines) >= 3, f"{tag}: too few lines for real variety"
        for line in lines:
            assert line == line.upper(), f"{tag} line not uppercase: {line!r}"


def test_gossip_events_appear_only_for_hub_occupations():
    pairs = generate_pairs(seed=0)
    for prompt, _ in pairs:
        occ = prompt.split("OCC:")[1].split(" ")[0]
        ev = prompt.split("EV:")[1].split("|")[0]
        if ev in GOSSIP_EVENTS:
            assert occ in GOSSIP_HUB_OCCUPATIONS, (
                f"gossip tag {ev!r} leaked into non-hub occupation {occ!r}")


def test_gossip_tag_produces_a_gossip_line_in_the_response():
    # When a combo's EV: is a gossip tag, the response MUST contain one of
    # that tag's lines -- gossip_tag is appended unconditionally in
    # _response(), not probabilistically, since this combo's whole point
    # is to teach the EV:<tag> -> secondhand-reaction association.
    pairs = generate_pairs(seed=0)
    gossip_pairs = [(p, r) for p, r in pairs
                    if p.split("EV:")[1].split("|")[0] in GOSSIP_EVENTS]
    assert gossip_pairs, "no gossip-tagged combos generated at seed=0"
    for prompt, response in gossip_pairs:
        tag = prompt.split("EV:")[1].split("|")[0]
        assert any(line in response for line in _GOSSIP_LINES[tag]), (
            f"gossip response missing a {tag} line: {response!r}")


def test_gossip_hub_occupations_still_get_direct_events_too():
    # GOSSIP_FRACTION < 1.0 -- pub_patron/villager must still see their
    # ordinary direct EVENTS_FOR_CONTEXT values most of the time, not be
    # entirely converted to gossip-only content.
    pairs = generate_pairs(seed=0)
    hub_events = [
        prompt.split("EV:")[1].split("|")[0]
        for prompt, _ in pairs
        if prompt.split("OCC:")[1].split(" ")[0] in GOSSIP_HUB_OCCUPATIONS
    ]
    non_gossip = [ev for ev in hub_events if ev not in GOSSIP_EVENTS]
    assert non_gossip, "gossip-hub occupations lost all direct-event coverage"


def test_non_hub_occupations_never_see_gossip_events():
    pairs = generate_pairs(seed=0)
    for prompt, _ in pairs:
        occ = prompt.split("OCC:")[1].split(" ")[0]
        if occ not in GOSSIP_HUB_OCCUPATIONS:
            ev = prompt.split("EV:")[1].split("|")[0]
            assert ev not in GOSSIP_EVENTS
