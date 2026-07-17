import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "manifests" / "dungeon_crawler.json"


def _load():
    return json.loads(MANIFEST_PATH.read_text())


def _declared_traits(manifest):
    return set(manifest["schema_fields"]["personality_traits"])


def test_manifest_loads_and_has_required_top_level_keys():
    manifest = _load()
    for key in ("schema_fields", "characters", "archetypes"):
        assert key in manifest


def test_character_personality_keys_are_declared_traits():
    manifest = _load()
    traits = _declared_traits(manifest)
    for character in manifest["characters"]:
        keys = set(character["personality"].keys())
        assert keys <= traits, (
            f"{character['id']}: undeclared trait(s) {keys - traits}"
        )


def test_archetype_personality_range_keys_are_declared_traits():
    manifest = _load()
    traits = _declared_traits(manifest)
    for archetype in manifest["archetypes"]:
        keys = set(archetype["personality_ranges"].keys())
        assert keys <= traits, (
            f"{archetype['id']}: undeclared trait(s) {keys - traits}"
        )


def test_archetype_personality_ranges_are_ordered_pairs():
    manifest = _load()
    for archetype in manifest["archetypes"]:
        for trait, bounds in archetype["personality_ranges"].items():
            assert len(bounds) == 2
            lo, hi = bounds
            assert lo <= hi, f"{archetype['id']}.{trait}: range {bounds} not ordered"


def test_ids_are_unique_across_characters_and_archetypes():
    manifest = _load()
    ids = [c["id"] for c in manifest["characters"]] + [a["id"] for a in manifest["archetypes"]]
    assert len(ids) == len(set(ids))
