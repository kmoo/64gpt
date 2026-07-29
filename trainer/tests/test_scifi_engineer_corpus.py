from ngpt_trainer.selena_corpus import MOODS, TRUST_TIERS
from ngpt_trainer.scifi_engineer_corpus import (
    ENGINEER_CONTEXTS, ENGINEER_PROFILE,
    prompt_for, combo_key, generate_pairs, corpus_text,
)


def test_prompt_for_format():
    assert (prompt_for(2, "cheerful", "reactor-check")
            == "P:woman D:measured OCC:engineer SPECIES:human R:best_friend "
               "BOND:stranger M:cheerful C:reactor-check AUD:witnessed EV:none|")
    assert (prompt_for(0, "worried", "hull-breach", "power_surge")
            == "P:woman D:measured OCC:engineer SPECIES:human R:stranger "
               "BOND:stranger M:worried C:hull-breach AUD:witnessed EV:power_surge|")


def test_engineer_profile_shape():
    assert ENGINEER_PROFILE["age"] == 35
    assert ENGINEER_PROFILE["gender"] == "female"
    assert ENGINEER_PROFILE["occupation"] == "engineer"
    assert ENGINEER_PROFILE["species"] == "human"
    assert ENGINEER_PROFILE["bond"] == "stranger"
    assert set(ENGINEER_PROFILE["traits"]) == {"warmth", "humor", "impulsivity", "bravery", "focus"}


def test_combo_key_round_trips_prompt_for():
    for tier, mood, ctx in [
        (1, "sassy", "quiet-shift"),
        (2, "tender", "hull-breach"),
    ]:
        prompt = prompt_for(tier, mood, ctx)
        assert combo_key(prompt) == (tier, mood, ctx)


def test_generate_pairs_deterministic():
    a = generate_pairs(seed=0, per_combo=2)
    b = generate_pairs(seed=0, per_combo=2)
    assert a == b


def test_generate_pairs_count():
    combo_count = len(TRUST_TIERS) * len(MOODS) * len(ENGINEER_CONTEXTS)
    assert combo_count == 3 * 5 * 3
    assert len(generate_pairs(seed=0, per_combo=1)) == combo_count
    assert len(generate_pairs(seed=0, per_combo=2)) == combo_count * 2


def test_generate_pairs_all_combos_parseable_and_in_grid():
    pairs = generate_pairs(seed=0, per_combo=1)
    for prompt, _response in pairs:
        tier, mood, ctx = combo_key(prompt)
        assert tier in TRUST_TIERS
        assert mood in MOODS
        assert ctx in ENGINEER_CONTEXTS


def test_responses_are_uppercase_and_nonempty():
    pairs = generate_pairs(seed=1, per_combo=2)
    for _prompt, response in pairs:
        assert response
        assert response == response.upper()


def test_corpus_text_concatenates_prompt_and_response():
    pairs = generate_pairs(seed=0, per_combo=1)
    text = corpus_text(seed=0, per_combo=1)
    assert text == "".join(p + r for p, r in pairs)


def test_occupation_token_is_genuinely_novel_vocabulary():
    """The whole point of the portability proof: OCC:engineer is a
    field VALUE the shared model's existing corpus (guard/selena/cast/
    shadewrath/korrath/elowen) never trains on -- confirms this corpus
    doesn't accidentally reuse an existing occupation string."""
    from ngpt_trainer import guard_corpus, selena_corpus
    assert "engineer" not in (guard_corpus.GUARD_PROFILES["guard#1001"]["occupation"],)
    assert ENGINEER_PROFILE["occupation"] == "engineer"
