from ngpt_trainer.selena_corpus import MOODS, TRUST_TIERS
from ngpt_trainer.guard_corpus import (
    GUARD_IDS, GUARD_CONTEXTS, GUARD_PROFILES,
    prompt_for, combo_key, generate_pairs,
)


def test_prompt_for_format():
    assert (prompt_for("guard#1001", 2, "cheerful", "greeting")
            == "P:man D:gruff OCC:guard SPECIES:human R:best_friend "
               "BOND:stranger M:cheerful C:greeting AUD:witnessed EV:none|")
    assert (prompt_for("guard#1002", 0, "worried", "combat-banter", "tough_fight")
            == "P:man D:stoic OCC:guard SPECIES:human R:stranger "
               "BOND:stranger M:worried C:combat-banter AUD:witnessed EV:tough_fight|")


def test_guard_profiles_match_engine_ground_truth():
    # Cross-checked against the compiled NPCDatabase.cpp, same discipline
    # test_guard_instances.py's EXPECTED dict uses.
    want = {
        "guard#1001": (38, "male", {"warmth": 43, "humor": 5, "impulsivity": 35, "bravery": 72, "focus": 73}),
        "guard#1002": (37, "male", {"warmth": 42, "humor": 24, "impulsivity": 16, "bravery": 60, "focus": 80}),
        "guard#1003": (30, "male", {"warmth": 33, "humor": 7, "impulsivity": 19, "bravery": 72, "focus": 56}),
        "guard#1004": (28, "female", {"warmth": 32, "humor": 18, "impulsivity": 24, "bravery": 84, "focus": 76}),
    }
    for gid, (age, gender, traits) in want.items():
        p = GUARD_PROFILES[gid]
        assert p["age"] == age, gid
        assert p["gender"] == gender, gid
        assert p["traits"] == traits, gid
        assert p["occupation"] == "guard"
        assert p["species"] == "human"
        assert p["bond"] == "stranger"


def test_combo_key_round_trips_prompt_for():
    for gid, tier, mood, ctx in [
        ("guard#1003", 1, "sassy", "quiet-moment"),
        ("guard#1004", 2, "tender", "combat-banter"),
    ]:
        prompt = prompt_for(gid, tier, mood, ctx)
        assert combo_key(prompt) == (tier, mood, ctx)


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
        tier, mood, ctx = combo_key(prompt)
        assert tier in TRUST_TIERS
        assert mood in MOODS
        assert ctx in GUARD_CONTEXTS


def test_responses_are_uppercase_and_nonempty():
    pairs = generate_pairs(seed=1, per_combo=2)
    for _prompt, response in pairs:
        assert response
        assert response == response.upper()
