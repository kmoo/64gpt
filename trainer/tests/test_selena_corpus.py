from ngpt_trainer.selena_corpus import (
    TRUST_TIERS, MOODS, CONTEXTS, THIN_ID,
    prompt_for, combo_key, generate_pairs, generate_thin_identity_pairs,
)


def test_prompt_for_format():
    assert (prompt_for(2, "cheerful", "item-found", "found_gem")
            == "N:selena TR:2 M:cheerful C:item-found EV:found_gem|")
    # empty event -> "none"
    assert (prompt_for(0, "worried", "greeting", "")
            == "N:selena TR:0 M:worried C:greeting EV:none|")


def test_combo_key_round_trips_prompt_for():
    for tier, mood, context, event in [
        (1, "sassy", "combat-banter", "tough_fight"),
        (2, "tender", "quiet-moment", ""),
    ]:
        prompt = prompt_for(tier, mood, context, event)
        assert combo_key(prompt) == (tier, mood, context)


def test_generate_pairs_deterministic():
    a = generate_pairs(seed=0, per_combo=2)
    b = generate_pairs(seed=0, per_combo=2)
    assert a == b


def test_generate_pairs_count():
    assert len(generate_pairs(seed=0, per_combo=1)) == 120
    assert len(generate_pairs(seed=0, per_combo=2)) == 240


def test_generate_pairs_all_combos_parseable_and_in_grid():
    pairs = generate_pairs(seed=0, per_combo=1)
    for prompt, _response in pairs:
        tier, mood, context = combo_key(prompt)
        assert tier in TRUST_TIERS
        assert mood in MOODS
        assert context in CONTEXTS


def test_generate_thin_identity_pairs_count_and_identity():
    pairs = generate_thin_identity_pairs(seed=1000, combos_used=20, lines_per_combo=12)
    assert len(pairs) == 20 * 12
    for prompt, _response in pairs:
        assert f"N:{THIN_ID} " in prompt
        assert "N:selena " not in prompt
