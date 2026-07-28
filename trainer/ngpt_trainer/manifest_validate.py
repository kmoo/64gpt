import json
from pathlib import Path

# (character field, archetype field, value shape) per schema category --
# personality is a dict on both sides (trait NAMES are the keys, not the
# scores); occupation/species/bond are plain string fields on both sides.
# The schema_fields key and the character/archetype field name are NOT
# the same string (e.g. schema key "occupations" vs field "occupation"),
# so this mapping can't be derived generically -- it has to be pinned.
_CATEGORIES = {
    "personality_traits": ("personality", "personality_ranges", "keys"),
    "occupations": ("occupation", "occupation", "value"),
    "species_types": ("species", "species", "value"),
    "bond_types": ("bond", "bond", "value"),
}


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _used_values(manifest: dict, character_field: str, archetype_field: str,
                  shape: str) -> set[str]:
    used = set()
    for character in manifest["characters"]:
        field = character.get(character_field)
        if field is None:
            continue
        used.update(field.keys() if shape == "keys" else [field])
    for archetype in manifest["archetypes"]:
        field = archetype.get(archetype_field)
        if field is None:
            continue
        used.update(field.keys() if shape == "keys" else [field])
    return used


def find_undeclared_values(manifest: dict) -> dict[str, set[str]]:
    undeclared = {}
    for category, (char_field, arch_field, shape) in _CATEGORIES.items():
        declared = set(manifest["schema_fields"].get(category, []))
        used = _used_values(manifest, char_field, arch_field, shape)
        undeclared[category] = used - declared
    return undeclared


def find_orphaned_declarations(manifest: dict) -> dict[str, set[str]]:
    orphaned = {}
    for category, (char_field, arch_field, shape) in _CATEGORIES.items():
        declared = set(manifest["schema_fields"].get(category, []))
        used = _used_values(manifest, char_field, arch_field, shape)
        orphaned[category] = declared - used
    return orphaned


def validate_manifest(manifest: dict) -> tuple[bool, list[str]]:
    undeclared = find_undeclared_values(manifest)
    orphaned = find_orphaned_declarations(manifest)
    problems = [
        f"undeclared {category}: {values}"
        for category, values in undeclared.items() if values
    ] + [
        f"orphaned {category}: {values}"
        for category, values in orphaned.items() if values
    ]
    return not problems, problems
