"""Compositional NPC conditioning (M9): NPC Profile + Relationship State
+ World Context -> a small set of reusable feature tokens, replacing
M8's opaque per-character identity tag (N:selena, N:guard#1001) with
tokens that generalize across characters sharing traits.

Why: M8's density table (docs/milestones/m8.md, Data Science Review)
shows guard corpus density and Selena's val loss moving in opposite
directions as guard density rises -- every new opaque identity tag
competes for its own slice of a fixed, shared parameter budget, because
nothing about the tags themselves is shared or reusable. A second sassy
12-year-old girl NPC should condition on nearly the same tokens Selena
already does, so the model learns feature->voice associations that
generalize instead of per-id memorization that only grows.

Design doc: docs/milestones/m9.md, docs/ideas-m7-living-npcs.md Part 3
(refined). Python-only this milestone -- the C++ port (ContextBuilder
extension) is real M9 follow-on work, gated on this vocabulary being
validated first, same sequencing M8 used (guard_instances.py prototyped
in Python before NPCDatabase.cpp's guard bits existed).

"Randomize" in this module means seed-deterministic, matching every
other RNG use in this project (core/ngpt_sample.cpp, M8's
spawnInstance): same seed -> byte-identical output, always.
"""

MASK32 = 0xFFFFFFFF

# Matches NPCDatabase::TRAITS exactly (game/src/user/NPCDatabase.cpp) --
# no new trait vocabulary invented, this is a second interpretation
# layer over the one M8 already trained.
TRAITS = ("warmth", "humor", "impulsivity", "bravery", "focus")

# Pulled from docs/milestones/m11.md's (formerly m10.md's) already-
# written town-cast list, plus villager/farmer/innkeeper -- not invented
# fresh for this module. M11.1 adds "villain"/"knight"/"companion": real
# manifest-schema decisions (docs/milestones/m11.1.md Part 1) made because
# genericizing Shadewrath/Korrath/Selena onto this scheme found no
# existing entry that fit without merging their voice into an unrelated
# occupation's bank (e.g. Shadewrath into "wizard", which the town
# tinker-wizard archetype deliberately contrasts against).
OCCUPATIONS = (
    "villager", "guard", "merchant", "wizard", "damsel", "pub_patron",
    "blacksmith", "healer", "noble", "bandit", "farmer", "innkeeper",
    "villain", "knight", "companion",
)

GENDERS = ("female", "male")

# M11.1: SPECIES: and BOND: (Part 3's new axes, plus two of the recorded-
# but-deferred candidates from docs/milestones/m11.1.md, pulled forward
# on Luke's direction 2026-07-22). Not every value has a trained
# character this pass -- "dwarf"/"beast" (SPECIES:) and "captor"/
# "family"/"mentor"/"romantic" (BOND:) are declared vocabulary with no
# corpus yet, same "declared, not yet exercised" status OCCUPATIONS
# values commonly start in (e.g. "damsel" before M11's town cast). Do
# not condition live gameplay on an unused value until a character
# trains it.
SPECIES_TYPES = ("human", "elf", "dwarf", "beast", "shade")
BOND_TYPES = (
    "stranger", "ally", "rival", "enemy", "captor", "captive", "family",
    "mentor", "romantic",
)

# M11.1 Part 3: AUD: (audience) operationalizes the bible's public/
# private/secret fields (docs/08-manifest-schema.md, existed since M7 as
# pure authoring guidance) into something the model can act on directly.
# "trusted" (a closeness threshold unlocking private content even when
# witnessed) is a recorded v2 candidate, not built this pass.
AUDIENCE_TYPES = ("alone", "witnessed")

# (name, min_age, max_age) -- inclusive bounds.
AGE_BUCKETS = (
    ("child", 0, 12),
    ("teen", 13, 19),
    ("adult", 20, 59),
    ("elder", 60, 120),
)

RELATIONSHIP_TIERS = (
    (0.0, "stranger"),
    (0.2, "acquaintance"),
    (0.4, "neutral"),
    (0.6, "friend"),
    (0.8, "close_friend"),
    (0.95, "best_friend"),
)

def xorshift32(x: int) -> int:
    """Same RNG discipline as core/ngpt_sample.cpp and M8's spawnInstance."""
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= x >> 17
    x ^= (x << 5) & MASK32
    return x & MASK32


def _age_bucket_name(age: int) -> str:
    for name, lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return name
    return AGE_BUCKETS[-1][0]


def age_gender_token(age: int, gender: str) -> str:
    """child/teen -> girl/boy, adult -> woman/man, elder -> elderly
    woman/elderly man. Reproduces the "12 years old" -> "girl" example."""
    if gender not in GENDERS:
        raise ValueError(f"unknown gender: {gender!r}")
    bucket = _age_bucket_name(age)
    female = gender == "female"
    if bucket in ("child", "teen"):
        return "girl" if female else "boy"
    if bucket == "adult":
        return "woman" if female else "man"
    return "elderly woman" if female else "elderly man"


# Ordered blend rules over the 5 trait sliders (0-100 each) -- first
# match wins. Selena's real personality {90,85,70,55,30} (NPCDatabase.cpp)
# MUST map to "sassy" -- the exact word from the worked example this
# module is built from. Covered by test_npc_service.py.
_BLENDS = (
    (lambda t: t["humor"] >= 70 and t["impulsivity"] >= 60, "sassy"),
    (lambda t: t["warmth"] >= 70 and t["bravery"] < 45, "gentle"),
    (lambda t: t["bravery"] >= 70 and t["warmth"] < 45, "gruff"),
    (lambda t: t["focus"] >= 70 and t["warmth"] < 45, "stoic"),
    (lambda t: t["bravery"] < 40 and t["focus"] < 40, "anxious"),
    (lambda t: t["warmth"] >= 70 and t["humor"] >= 60, "cheerful"),
    (lambda t: t["impulsivity"] < 35 and t["focus"] >= 60, "measured"),
    (lambda t: t["bravery"] >= 75 and t["impulsivity"] >= 60, "reckless"),
    (lambda t: t["warmth"] < 35 and t["humor"] < 35, "cold"),
)

# Fallback: (high_word, low_word) per trait, used when no blend matches.
_DOMINANT_WORDS = {
    "warmth": ("warm", "cold"),
    "humor": ("playful", "serious"),
    "impulsivity": ("impulsive", "careful"),
    "bravery": ("bold", "timid"),
    "focus": ("focused", "distracted"),
}


def personality_descriptor(traits: dict) -> str:
    """traits: dict with all 5 TRAITS keys, 0-100 each."""
    missing = set(TRAITS) - set(traits)
    if missing:
        raise ValueError(f"missing trait(s): {sorted(missing)}")
    for check, label in _BLENDS:
        if check(traits):
            return label
    # No blend matched: single most-extreme-from-neutral (50) trait wins.
    dominant = max(TRAITS, key=lambda k: abs(traits[k] - 50))
    high_word, low_word = _DOMINANT_WORDS[dominant]
    return high_word if traits[dominant] >= 50 else low_word


def closeness(state: dict) -> float:
    """Mean of familiarity/affection/trust/respect -- fear is a separate
    modifier, not averaged in (a feared-but-trusted relationship reads
    differently than a low-trust one and shouldn't cancel out)."""
    return (state["familiarity"] + state["affection"]
            + state["trust"] + state["respect"]) / 4.0


def relationship_label(state: dict) -> tuple[float, str]:
    """-> (closeness, tier_name), e.g. (1.0, "best_friend")."""
    c = closeness(state)
    tier = RELATIONSHIP_TIERS[0][1]
    for threshold, name in RELATIONSHIP_TIERS:
        if c >= threshold:
            tier = name
    return c, tier


def random_relationship_state(seed: int) -> dict:
    """Seed-deterministic. Five axes, each 0.0-1.0."""
    rng = seed if seed != 0 else 1
    out = {}
    for axis in ("familiarity", "affection", "trust", "respect", "fear"):
        rng = xorshift32(rng)
        out[axis] = (rng % 1001) / 1000.0
    return out


def random_npc_profile(seed: int) -> dict:
    """Seed-deterministic. occupation/age/gender/species/bond/personality
    traits."""
    rng = seed if seed != 0 else 1

    rng = xorshift32(rng)
    occupation = OCCUPATIONS[rng % len(OCCUPATIONS)]

    rng = xorshift32(rng)
    age = 5 + (rng % 80)  # 5..84

    rng = xorshift32(rng)
    gender = GENDERS[rng % len(GENDERS)]

    rng = xorshift32(rng)
    species = SPECIES_TYPES[rng % len(SPECIES_TYPES)]

    rng = xorshift32(rng)
    bond = BOND_TYPES[rng % len(BOND_TYPES)]

    traits = {}
    for name in TRAITS:
        rng = xorshift32(rng)
        traits[name] = rng % 101  # 0..100

    return {
        "occupation": occupation,
        "age": age,
        "gender": gender,
        "species": species,
        "bond": bond,
        "traits": traits,
    }


def conditioning_features(profile: dict, relationship: dict) -> str:
    """The compositional feature string, e.g.
    "girl age:12 sassy VILLAGER human R:best_friend rival"."""
    person = age_gender_token(profile["age"], profile["gender"])
    descriptor = personality_descriptor(profile["traits"])
    _, tier = relationship_label(relationship)
    return (f"{person} age:{profile['age']} {descriptor} "
            f"{profile['occupation'].upper()} {profile['species']} "
            f"R:{tier} {profile['bond']}")


def prompt_fields(profile: dict, relationship: dict, mood: str, context: str,
                   audience: str = "alone", event: str = "") -> str:
    """The actual training/inference prompt string: colon-delimited tokens
    (matching ContextBuilder's N:/TR:/M:/C:/EV: convention -- every
    space-separated token has a colon, unlike conditioning_features()'s
    human-readable form). Supersedes M7/M8's opaque N:<id> AND its TR:
    trust-tier field -- R: (relationship tier, derived the same way as
    the old TR: dial but from real relationship state, not a flat 0-3
    knob) covers the same closeness concept with one axis instead of two.

    No raw AGE: field -- P:'s age/gender token (girl/boy/woman/man/
    elderly woman/elderly man) already carries the only age signal that
    matters for voice; a bare integer would just be redundant precision
    the model has no way to use (a char-level GRU has no numeric-magnitude
    primitive, so "AGE:63" costs prompt characters for a distinction P:
    already made without teaching the model anything new).

    "P:girl D:sassy OCC:villager SPECIES:human R:best_friend BOND:rival
    M:cheerful C:greeting AUD:alone EV:none|"

    M11.1: SPECIES:/BOND: (Part 1, genericization) and AUD: (Part 3, the
    new audience axis operationalizing the bible's public/private/secret
    fields) grouped with their nearest existing axis -- identity
    (P/D/OCC/SPECIES), relationship (R/BOND), situation (M/C/AUD) -- not
    appended at the end, so a human reading the string still finds
    related fields next to each other.
    """
    # age_gender_token can be two words ("elderly woman") -- every
    # space-separated prompt token must carry its own colon, so multi-word
    # tokens get underscored here (conditioning_features() keeps the
    # natural spaced form for human/LLM-facing text).
    person = age_gender_token(profile["age"], profile["gender"]).replace(" ", "_")
    descriptor = personality_descriptor(profile["traits"])
    _, tier = relationship_label(relationship)
    ev = event if event else "none"
    return (f"P:{person} D:{descriptor} "
            f"OCC:{profile['occupation']} SPECIES:{profile['species']} "
            f"R:{tier} BOND:{profile['bond']} M:{mood} C:{context} "
            f"AUD:{audience} EV:{ev}|")


def parse_prompt_fields(prompt: str) -> dict[str, str]:
    """Inverse of prompt_fields(): colon-delimited tokens back into a
    dict, e.g. {"P": "girl", "D": "sassy", ..., "M": "cheerful", ...}.
    Every corpus generator funnels through prompt_fields(), so this one
    parser covers all six without touching any of them -- same technique
    selena_corpus.combo_key() already uses locally for R:/M:/C:. M12.3
    uses this to recover D:/M: for per-step attribute embeddings without
    threading a parallel structured-data path through corpus generation."""
    fields = {}
    for tok in prompt.rstrip("|").split(" "):
        k, _, v = tok.partition(":")
        fields[k] = v
    return fields


def strip_prompt_fields(prompt: str, keys: set[str]) -> str:
    """Inverse-ish of parse_prompt_fields: removes the given colon-tokens
    (e.g. {"D", "M"}) from a prompt string, keeping every other token in
    its original order and re-attaching the trailing "|". M12.4's
    redundancy-ablation spike uses this to strip D:/M: from what the
    model actually sees as its text prefix, while
    parse_prompt_fields(prompt) on the ORIGINAL (unstripped) prompt still
    supplies the per-step attribute columns -- isolating whether M12.3's
    regression came from the columns being redundant with the prefix."""
    kept = [tok for tok in prompt.rstrip("|").split(" ")
            if tok.partition(":")[0] not in keys]
    return " ".join(kept) + "|"


def generate_sample_population(n: int, seed: int = 0) -> list[dict]:
    """n randomized NPCs (profile + relationship), for coverage-checking
    the vocabulary and grounding docs in real generated examples rather
    than hand-picked cases. Deterministic given seed."""
    samples = []
    rng = seed if seed != 0 else 1
    for i in range(n):
        rng = xorshift32(rng)
        profile_seed = rng
        rng = xorshift32(rng)
        relationship_seed = rng

        profile = random_npc_profile(profile_seed)
        relationship = random_relationship_state(relationship_seed)
        samples.append({
            "profile_seed": profile_seed,
            "relationship_seed": relationship_seed,
            "profile": profile,
            "relationship": relationship,
            "features": conditioning_features(profile, relationship),
        })
    return samples


if __name__ == "__main__":
    import json
    import pathlib

    pop = generate_sample_population(300, seed=0xC0FFEE)
    out_path = pathlib.Path(__file__).parent.parent / "npc_service_samples.json"
    out_path.write_text(json.dumps(pop, indent=2))

    occupations_seen = {s["profile"]["occupation"] for s in pop}
    descriptors_seen = {personality_descriptor(s["profile"]["traits"]) for s in pop}
    tiers_seen = {relationship_label(s["relationship"])[1] for s in pop}
    print(f"wrote {len(pop)} samples to {out_path}")
    print(f"occupations covered: {len(occupations_seen)}/{len(OCCUPATIONS)}")
    print(f"descriptors seen: {sorted(descriptors_seen)}")
    print(f"relationship tiers seen: {sorted(tiers_seen)}")
    print()
    print("first 10 samples:")
    for s in pop[:10]:
        print(f"  {s['features']}")

    selena = {"warmth": 90, "humor": 85, "impulsivity": 70, "bravery": 55, "focus": 30}
    print()
    print(f"calibration check -- Selena's personality -> "
          f"{personality_descriptor(selena)!r} (expect 'sassy')")
