"""Cross-checks guard_instances.py against ground truth pulled directly
from the production game/src/user/NPCDatabase.cpp (spawnInstance), by
compiling a throwaway one-off dump program against that exact .cpp file
and NOT reimplementing the logic by hand a second time -- if this test
ever fails, it means the Python port drifted from the shipped engine
code, not that the "expected" numbers below are wrong.

Regenerate expected values (only if NPCDatabase.cpp's GUARD_ARCHETYPE or
spawnInstance changes) with:

    clang++ -std=c++20 -I game/src/user <dump_program>.cpp \
      game/src/user/NPCDatabase.cpp -o /tmp/dump && /tmp/dump

where <dump_program>.cpp calls NPCDatabase::spawnInstance(GUARD_ARCHETYPE,
seed) for each seed below and prints id/name/personality.
"""
from ngpt_trainer.guard_instances import spawn_guard_instance

# seed -> (id, name, personality dict, age, gender), verified 2026-07-17
# (personality/name/id) and 2026-07-22 (age/gender, M11.1) against the
# real NPCDatabase.cpp via the dump-program method described above.
EXPECTED = {
    0x1001: ("guard#1001", "BRAM",
             {"warmth": 43, "humor": 5, "impulsivity": 35, "bravery": 72, "focus": 73},
             38, "male"),
    0x1002: ("guard#1002", "EDRIC",
             {"warmth": 42, "humor": 24, "impulsivity": 16, "bravery": 60, "focus": 80},
             37, "male"),
    0x1003: ("guard#1003", "EDRIC",
             {"warmth": 33, "humor": 7, "impulsivity": 19, "bravery": 72, "focus": 56},
             30, "male"),
    0x1004: ("guard#1004", "IVOR",
             {"warmth": 32, "humor": 18, "impulsivity": 24, "bravery": 84, "focus": 76},
             28, "female"),
}


def test_spawn_guard_instance_matches_engine_ground_truth():
    for seed, (expected_id, expected_name, expected_personality,
              expected_age, expected_gender) in EXPECTED.items():
        instance = spawn_guard_instance(seed)
        assert instance["id"] == expected_id, seed
        assert instance["name"] == expected_name, seed
        assert instance["personality"] == expected_personality, seed
        assert instance["age"] == expected_age, seed
        assert instance["gender"] == expected_gender, seed


def test_spawn_guard_instance_deterministic():
    a = spawn_guard_instance(0x1001)
    b = spawn_guard_instance(0x1001)
    assert a == b


def test_spawn_guard_instance_seed_zero_remaps_to_one():
    # id is formatted from the raw seed (0000 vs 0001), but personality
    # and name derive from the remapped rng (0 -> 1), so those match --
    # same split the C++ spawnInstance() makes.
    zero = spawn_guard_instance(0)
    one = spawn_guard_instance(1)
    assert zero["personality"] == one["personality"]
    assert zero["name"] == one["name"]
    assert zero["id"] == "guard#0000"
    assert one["id"] == "guard#0001"
