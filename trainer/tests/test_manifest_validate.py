import json
from pathlib import Path
from ngpt_trainer.manifest_validate import load_manifest, find_undeclared_values, find_orphaned_declarations, validate_manifest

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "manifests" / "dungeon_crawler.json"

def test_no_undeclared_values_in_real_manifest():
    # The real manifest legitimately has ORPHANED declarations today
    # (e.g. bond_types "mentor"/"family"/"captor"/"romantic", species
    # "dwarf"/"beast" -- schema headroom for characters/archetypes that
    # don't exist yet), so asserting validate_manifest() == (True, [])
    # against it would be wrong. UNDECLARED values are the real invariant
    # worth guarding: something USED that isn't in the schema at all is
    # a genuine error (e.g. a typo'd occupation), unlike an as-yet-unused
    # declaration.
    manifest = load_manifest(MANIFEST_PATH)
    undeclared = find_undeclared_values(manifest)
    assert undeclared == {
        "personality_traits": set(),
        "occupations": set(),
        "species_types": set(),
        "bond_types": set(),
    }

def test_find_undeclared_values():
    manifest = {
        "schema_fields": {
            "personality_traits": ["bold", "courageous"],
            "occupations": ["warrior", "mage"],
            "species_types": ["human", "elf"],
            "bond_types": ["friend", "ally"]
        },
        "characters": [
            {"id": "char1", "personality": {"bold": 1}, "occupation": "warrior", "species": "human", "bond": "friend"}
        ],
        "archetypes": [
            {"id": "arch1", "personality_ranges": {"courageous": [1, 10]}, "occupation": "mage", "species": "elf", "bond": "ally"}
        ]
    }
    undeclared = find_undeclared_values(manifest)
    assert undeclared == {
        "personality_traits": set(),
        "occupations": set(),
        "species_types": set(),
        "bond_types": set()
    }

def test_find_orphaned_declarations():
    manifest = {
        "schema_fields": {
            "personality_traits": ["bold", "courageous"],
            "occupations": ["warrior", "mage"],
            "species_types": ["human", "elf"],
            "bond_types": ["friend", "ally"]
        },
        "characters": [
            {"id": "char1", "personality": {"bold": 1}, "occupation": "warrior", "species": "human", "bond": "friend"}
        ],
        "archetypes": [
            {"id": "arch1", "personality_ranges": {"courageous": [1, 10]}, "occupation": "mage", "species": "elf", "bond": "ally"}
        ]
    }
    orphaned = find_orphaned_declarations(manifest)
    assert orphaned == {
        "personality_traits": set(),
        "occupations": set(),
        "species_types": set(),
        "bond_types": set()
    }

def test_validate_dirty_manifest():
    manifest = {
        "schema_fields": {
            "personality_traits": ["bold", "courageous"],
            "occupations": ["warrior", "mage", "ranger"],
            "species_types": ["human", "elf"],
            "bond_types": ["friend", "ally"]
        },
        "characters": [
            {"id": "char1", "personality": {"bold": 1}, "occupation": "warrior", "species": "human", "bond": "friend"}
        ],
        "archetypes": [
            {"id": "arch1", "personality_ranges": {"courageous": [1, 10]}, "occupation": "mage", "species": "elf", "bond": "ally"},
            {"id": "arch2", "personality_ranges": {"courageous": [1, 10]}, "occupation": "warrior", "species": "human", "bond": "foe"}
        ]
    }
    # "ranger" is declared but never used (orphaned occupations); "foe" is
    # used but never declared (undeclared bond_types). "warrior"/"mage" are
    # both declared AND used, so they must NOT show up as problems.
    is_valid, problems = validate_manifest(manifest)
    assert not is_valid
    assert "undeclared bond_types: {'foe'}" in problems
    assert "orphaned occupations: {'ranger'}" in problems

def test_empty_schema_fields_category_defaults_to_empty_declared_set():
    # schema_fields missing a category entirely (not even an empty list)
    # must not raise -- .get(category, []) already handles this, this
    # locks it in as a regression guard.
    manifest = {
        "schema_fields": {
            "personality_traits": ["bold"],
            "species_types": ["human"],
            "bond_types": ["friend"],
            # "occupations" key is entirely absent
        },
        "characters": [
            {"id": "char1", "personality": {"bold": 1}, "species": "human", "bond": "friend"}
        ],
        "archetypes": []
    }
    undeclared = find_undeclared_values(manifest)
    orphaned = find_orphaned_declarations(manifest)
    assert undeclared["occupations"] == set()
    assert orphaned["occupations"] == set()

def test_character_missing_a_field_entirely_is_skipped_not_an_error():
    # character.get(field) returns None for a missing key -- must be
    # skipped (not counted as a used value, not a KeyError), same
    # "field is None: continue" path a missing occupation would hit.
    manifest = {
        "schema_fields": {
            "personality_traits": ["bold"],
            "occupations": ["warrior"],
            "species_types": ["human"],
            "bond_types": ["friend"],
        },
        "characters": [
            {"id": "char1", "personality": {"bold": 1}, "species": "human", "bond": "friend"}
            # no "occupation" key at all
        ],
        "archetypes": []
    }
    undeclared = find_undeclared_values(manifest)
    assert undeclared["occupations"] == set()
    orphaned = find_orphaned_declarations(manifest)
    assert orphaned["occupations"] == {"warrior"}  # declared, never used

def test_empty_characters_and_archetypes_lists():
    manifest = {
        "schema_fields": {
            "personality_traits": [], "occupations": [],
            "species_types": [], "bond_types": [],
        },
        "characters": [],
        "archetypes": []
    }
    is_valid, problems = validate_manifest(manifest)
    assert is_valid
    assert problems == []

def test_load_manifest_accepts_str_and_path(tmp_path):
    manifest_dict = {
        "schema_fields": {"personality_traits": [], "occupations": [],
                          "species_types": [], "bond_types": []},
        "characters": [], "archetypes": []
    }
    p = tmp_path / "test_manifest.json"
    p.write_text(json.dumps(manifest_dict))

    from_str = load_manifest(str(p))
    from_path = load_manifest(p)
    assert from_str == manifest_dict
    assert from_path == manifest_dict
