"""Python port of NPCDatabase::spawnInstance (game/src/user/NPCDatabase.cpp)
for the guard archetype -- MUST match the C++ engine bit-for-bit, same
discipline as ref_impl matching core/'s quantized inference. Ports don't
share a build system across game/ and trainer/, so parity is enforced by
test_guard_instances.py asserting against ground truth pulled directly
from the production NPCDatabase.cpp (see that test's docstring for how).

xorshift32 constants (13/17/5 shifts) match core/ngpt_sample.cpp exactly
-- one RNG discipline project-wide, per docs/milestones/m8.md section 2.
"""

MASK32 = 0xFFFFFFFF

TRAITS = ("warmth", "humor", "impulsivity", "bravery", "focus")

# Must match game/src/user/NPCDatabase.cpp's GUARD_ARCHETYPE exactly.
GUARD_RANGES = {
    "warmth": (20, 45),
    "humor": (5, 30),
    "impulsivity": (10, 35),
    "bravery": (60, 90),
    "focus": (55, 85),
}

# Must match game/src/user/NPCDatabase.cpp's GUARD_NAMES exactly.
GUARD_NAMES = ("BRAM", "CORVIN", "DESMOND", "EDRIC",
               "FENWICK", "GARRICK", "HALVOR", "IVOR")

# Must match game/src/user/NPCDatabase.cpp's GUARD_ARCHETYPE.ageRange (M10).
GUARD_AGE_RANGE = (25, 55)


def xorshift32(x: int) -> int:
    x &= MASK32
    x ^= (x << 13) & MASK32
    x ^= (x >> 17)
    x ^= (x << 5) & MASK32
    return x & MASK32


def spawn_guard_instance(seed: int) -> dict:
    """Mirrors NPCDatabase::spawnInstance(GUARD_ARCHETYPE, seed) exactly:
    per-trait jitter in TRAITS order, then one more xorshift32 step to
    pick a name, then age (M10 ageRange jitter), then a gender coin flip,
    then the id as "guard#<4 lowercase hex digits>". age/gender added
    M11.1 (docs/milestones/m11.1.md Part 1) -- guard_corpus.py's real
    Profile dicts need them; verified against the compiled NPCDatabase.cpp
    via the dump-program method this module's test file documents."""
    rng = seed if seed != 0 else 1  # xorshift32's fixed point, same remap
                                     # as core/ngpt.cpp's ngpt_reset
    personality = {}
    for trait in TRAITS:
        lo, hi = GUARD_RANGES[trait]
        rng = xorshift32(rng)
        span = hi - lo + 1
        personality[trait] = lo + (rng % span)

    rng = xorshift32(rng)
    name = GUARD_NAMES[rng % len(GUARD_NAMES)]

    rng = xorshift32(rng)
    age_lo, age_hi = GUARD_AGE_RANGE
    age = age_lo + (rng % (age_hi - age_lo + 1))

    rng = xorshift32(rng)
    is_female = (rng & 1) != 0

    return {
        "id": f"guard#{seed & 0xFFFF:04x}",
        "name": name,
        "personality": personality,
        "age": age,
        "gender": "female" if is_female else "male",
    }
