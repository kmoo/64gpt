from ngpt_trainer.selena_corpus import (
    TRUST_TIERS, MOODS, CONTEXTS,
    prompt_for, combo_key, generate_pairs,
)


def test_prompt_for_format():
    assert (prompt_for(2, "cheerful", "item-found", "found_gem")
            == "P:girl D:sassy OCC:companion SPECIES:human R:best_friend "
               "BOND:ally M:cheerful C:item-found AUD:witnessed EV:found_gem|")
    # empty event -> "none"; "worried" is not an _ALONE_MOODS entry -> AUD:witnessed
    assert (prompt_for(0, "worried", "greeting", "")
            == "P:girl D:sassy OCC:companion SPECIES:human R:stranger "
               "BOND:ally M:worried C:greeting AUD:witnessed EV:none|")


def test_prompt_for_alone_moods_get_aud_alone():
    # tender/embarrassed are Selena's vulnerable-register moods (module
    # header) -- the only two that carry AUD:alone.
    assert "AUD:alone" in prompt_for(2, "tender", "quiet-moment", "")
    assert "AUD:alone" in prompt_for(2, "embarrassed", "joke", "")
    assert "AUD:witnessed" in prompt_for(2, "cheerful", "greeting", "")


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
