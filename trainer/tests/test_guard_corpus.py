from ngpt_trainer.selena_corpus import MOODS, TRUST_TIERS
from ngpt_trainer.guard_corpus import (
    GUARD_IDS, GUARD_CONTEXTS,
    prompt_for, combo_key, generate_pairs,
)


def test_prompt_for_format():
    assert (prompt_for("guard#1001", 2, "cheerful", "greeting")
            == "N:guard#1001 TR:2 M:cheerful C:greeting EV:none|")
    assert (prompt_for("guard#1002", 0, "worried", "combat-banter", "tough_fight")
            == "N:guard#1002 TR:0 M:worried C:combat-banter EV:tough_fight|")


def test_combo_key_round_trips_prompt_for():
    for gid, tier, mood, ctx in [
        ("guard#1003", 1, "sassy", "quiet-moment"),
        ("guard#1004", 2, "tender", "combat-banter"),
    ]:
        prompt = prompt_for(gid, tier, mood, ctx)
        assert combo_key(prompt) == (gid, tier, mood, ctx)


def test_generate_pairs_deterministic():
    a = generate_pairs(seed=0, per_combo=2)
    b = generate_pairs(seed=0, per_combo=2)
    assert a == b


def test_generate_pairs_count():
    combo_count = len(GUARD_IDS) * len(TRUST_TIERS) * len(MOODS) * len(GUARD_CONTEXTS)
    assert combo_count == 4 * 3 * 5 * 3
    assert len(generate_pairs(seed=0, per_combo=1)) == combo_count
    assert len(generate_pairs(seed=0, per_combo=2)) == combo_count * 2


def test_generate_pairs_all_combos_parseable_and_in_grid():
    pairs = generate_pairs(seed=0, per_combo=1)
    for prompt, _response in pairs:
        gid, tier, mood, ctx = combo_key(prompt)
        assert gid in GUARD_IDS
        assert tier in TRUST_TIERS
        assert mood in MOODS
        assert ctx in GUARD_CONTEXTS


def test_responses_are_uppercase_and_nonempty():
    pairs = generate_pairs(seed=1, per_combo=2)
    for _prompt, response in pairs:
        assert response
        assert response == response.upper()
