"""Tests for npc_service.py -- M9's compositional conditioning mapping.
See docs/milestones/m9.md for the design rationale."""
from ngpt_trainer.npc_service import (
    OCCUPATIONS,
    TRAITS,
    age_gender_token,
    closeness,
    conditioning_features,
    generate_sample_population,
    personality_descriptor,
    prompt_fields,
    random_npc_profile,
    random_relationship_state,
    relationship_label,
    xorshift32,
)


def test_selena_calibration_maps_to_sassy():
    # Selena's real personality, NPCDatabase.cpp -- the exact worked
    # example this module's design is built from.
    selena = {"warmth": 90, "humor": 85, "impulsivity": 70, "bravery": 55, "focus": 30}
    assert personality_descriptor(selena) == "sassy"


def test_personality_descriptor_requires_all_traits():
    incomplete = {"warmth": 50, "humor": 50}
    try:
        personality_descriptor(incomplete)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_age_gender_token_buckets():
    assert age_gender_token(12, "female") == "girl"
    assert age_gender_token(12, "male") == "boy"
    assert age_gender_token(19, "female") == "girl"  # teen still girl/boy
    assert age_gender_token(20, "female") == "woman"
    assert age_gender_token(20, "male") == "man"
    assert age_gender_token(60, "female") == "elderly woman"
    assert age_gender_token(60, "male") == "elderly man"


def test_age_gender_token_rejects_unknown_gender():
    try:
        age_gender_token(30, "unknown")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_relationship_label_boundaries():
    stranger = {"familiarity": 0.0, "affection": 0.0, "trust": 0.0, "respect": 0.0}
    best_friend = {"familiarity": 1.0, "affection": 1.0, "trust": 1.0, "respect": 1.0}
    c, tier = relationship_label(stranger)
    assert c == 0.0
    assert tier == "stranger"
    c, tier = relationship_label(best_friend)
    assert c == 1.0
    assert tier == "best_friend"


def test_closeness_ignores_fear():
    # fear is a separate modifier, not averaged into closeness -- a high
    # fear value must not change the closeness score.
    base = {"familiarity": 0.5, "affection": 0.5, "trust": 0.5, "respect": 0.5}
    with_fear = dict(base, fear=0.9)
    assert closeness(base) == closeness(with_fear) == 0.5


def test_xorshift32_matches_seed_zero_remap():
    # seed 0 remaps to 1 -- same fixed-point rule as core/ngpt_sample.cpp
    # and M8's spawnInstance (0 ^ anything == 0, xorshift32's fixed point).
    assert xorshift32(0) == 0
    assert xorshift32(1) != 0


def test_random_relationship_state_deterministic():
    a = random_relationship_state(42)
    b = random_relationship_state(42)
    assert a == b
    for axis in ("familiarity", "affection", "trust", "respect", "fear"):
        assert 0.0 <= a[axis] <= 1.0


def test_random_relationship_state_seed_zero_remaps_to_one():
    assert random_relationship_state(0) == random_relationship_state(1)


def test_random_npc_profile_deterministic():
    a = random_npc_profile(7)
    b = random_npc_profile(7)
    assert a == b
    assert a["occupation"] in OCCUPATIONS
    assert a["gender"] in ("female", "male")
    assert 5 <= a["age"] <= 84
    assert set(a["traits"].keys()) == set(TRAITS)
    for v in a["traits"].values():
        assert 0 <= v <= 100


def test_conditioning_features_format():
    profile = {"occupation": "guard", "age": 30, "gender": "male",
               "traits": {"warmth": 90, "humor": 85, "impulsivity": 70,
                          "bravery": 55, "focus": 30}}
    relationship = {"familiarity": 1.0, "affection": 1.0, "trust": 1.0, "respect": 1.0}
    features = conditioning_features(profile, relationship)
    assert features == "man age:30 sassy GUARD R:best_friend"


def test_generate_sample_population_deterministic_and_covers_vocab():
    pop_a = generate_sample_population(300, seed=0xC0FFEE)
    pop_b = generate_sample_population(300, seed=0xC0FFEE)
    assert pop_a == pop_b
    occupations_seen = {s["profile"]["occupation"] for s in pop_a}
    assert occupations_seen == set(OCCUPATIONS)


def test_prompt_fields_format():
    profile = {"occupation": "guard", "age": 30, "gender": "male",
               "traits": {"warmth": 90, "humor": 85, "impulsivity": 70,
                          "bravery": 55, "focus": 30}}
    relationship = {"familiarity": 1.0, "affection": 1.0, "trust": 1.0, "respect": 1.0}
    prompt = prompt_fields(profile, relationship, "cheerful", "greeting")
    assert prompt == "P:man AGE:30 D:sassy OCC:guard R:best_friend M:cheerful C:greeting EV:none|"
    # every space-separated token carries its own colon (ContextBuilder's
    # existing N:/TR:/M:/C:/EV: parsing convention)
    for tok in prompt.rstrip("|").split(" "):
        assert ":" in tok


def test_prompt_fields_multiword_person_token_underscored():
    # age_gender_token can return "elderly woman"/"elderly man" -- must not
    # break the one-token-per-space rule.
    profile = {"occupation": "healer", "age": 70, "gender": "female",
               "traits": {"warmth": 50, "humor": 50, "impulsivity": 50,
                          "bravery": 50, "focus": 50}}
    relationship = {"familiarity": 0.0, "affection": 0.0, "trust": 0.0, "respect": 0.0}
    prompt = prompt_fields(profile, relationship, "worried", "farewell", "heading_home")
    assert "P:elderly_woman " in prompt
    for tok in prompt.rstrip("|").split(" "):
        assert ":" in tok


def test_generate_sample_population_different_seed_differs():
    pop_a = generate_sample_population(20, seed=1)
    pop_b = generate_sample_population(20, seed=2)
    assert pop_a != pop_b
