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


def test_tier_is_declared_and_valid_for_every_entry():
    # M10: full/mid/thin is meta-schema (docs/08-manifest-schema.md), not
    # per-project content -- every characters[]/archetypes[] entry must
    # declare one, and it must be a real tier, not an unvalidated string.
    manifest = _load()
    entries = manifest["characters"] + manifest["archetypes"]
    for entry in entries:
        assert "tier" in entry, f"{entry['id']}: missing tier"
        assert entry["tier"] in ("full", "mid", "thin"), (
            f"{entry['id']}: invalid tier {entry['tier']!r}")


def test_characters_are_full_or_mid_tier_not_thin():
    # characters[] entries are hand-authored individuals (a fixed
    # personality point, a bible) -- thin tier is archetypes[]-only by
    # definition (docs/08-manifest-schema.md's tier table).
    manifest = _load()
    for character in manifest["characters"]:
        assert character["tier"] in ("full", "mid"), (
            f"{character['id']}: characters[] entries can't be thin-tier")


def test_archetypes_are_thin_tier():
    # archetypes[] entries are generators (a range + shared corpus), not
    # authored individuals -- full/mid tier implies a fixed personality
    # point and a bible, which archetypes[] doesn't have a shape for yet.
    manifest = _load()
    for archetype in manifest["archetypes"]:
        assert archetype["tier"] == "thin", (
            f"{archetype['id']}: archetypes[] entries must be thin-tier")


def test_archetype_occupation_is_declared_in_npc_service():
    # M10: an archetype's occupation feeds NpcService::buildPromptFields()'s
    # OCC: field directly (NPCDatabase::Archetype.occupation ->
    # spawnInstance() -> profileFor()) -- a typo'd occupation string would
    # silently train an OCC: value the shared vocabulary doesn't recognize.
    from ngpt_trainer.npc_service import OCCUPATIONS
    manifest = _load()
    for archetype in manifest["archetypes"]:
        assert "occupation" in archetype, f"{archetype['id']}: missing occupation"
        assert archetype["occupation"] in OCCUPATIONS, (
            f"{archetype['id']}: occupation {archetype['occupation']!r} not in "
            f"npc_service.OCCUPATIONS")


def test_full_tier_characters_have_a_bible():
    # full tier means hand-authored individual (docs/08-manifest-schema.md)
    # -- a bible is the whole point of that tier, not optional content.
    manifest = _load()
    for character in manifest["characters"]:
        if character["tier"] != "full":
            continue
        bible = character.get("bible")
        assert bible, f"{character['id']}: full-tier character missing a bible"
        for field in ("public", "private", "secret", "fear", "desire"):
            assert bible.get(field), f"{character['id']}: bible missing {field!r}"
