# Task 011 — manifest-validate

## CONTRACT

```yaml
id: 011-manifest-validate
goal: >
  Implement a reusable manifest schema-consistency checker for M14's
  planned "manifest-update skill" (docs/milestones/m14.md section 1b):
  a function that finds two kinds of schema drift in
  manifests/dungeon_crawler.json -- (1) an undeclared value: a
  character/archetype field using a value not present in the matching
  schema_fields[] list, and (2) an orphaned declaration: a value listed
  in schema_fields[] that no character or archetype actually uses. Pure
  JSON/dict logic, no training, no model loading, no LLM calls.
background: >
  trainer/tests/test_manifest_schema.py already checks ONE direction for
  ONE category: that every character/archetype personality-trait KEY is
  in schema_fields.personality_traits (undeclared-value direction only).
  This task builds a general-purpose, reusable version covering FOUR
  categories (personality_traits, occupations, species_types,
  bond_types) and BOTH directions (undeclared AND orphaned), as an
  importable module -- not more inline test assertions -- because M14's
  actual planned deliverable is a manifest-update skill that calls a
  validate step programmatically, not just a pytest file. The manifest
  shape itself is documented in docs/08-manifest-schema.md and the real
  file being validated is manifests/dungeon_crawler.json (4 characters,
  7 archetypes today).
constraints: |
  - Pure Python 3.12, stdlib only (json, pathlib) -- no new third-party
    dependencies.
  - Do not modify manifests/dungeon_crawler.json or
    trainer/tests/test_manifest_schema.py -- read-only references only.
  - mood/context are NOT covered by this task -- they are runtime
    dialogue axes used by corpus-generation scripts, not per-character
    manifest fields, so there is no manifest-side "used" signal to check
    them against. Only the four categories named in acceptance_criteria
    below are in scope.
  - Functions take a parsed manifest dict as an argument (or a path to
    load) -- no hidden global state, no caching.
  - No docstrings that just restate the signature -- only where the WHY
    is non-obvious.
allowed_files:
  - trainer/ngpt_trainer/manifest_validate.py
  - trainer/tests/test_manifest_validate.py
reference_files:
  - manifests/dungeon_crawler.json
  - docs/08-manifest-schema.md
  - trainer/tests/test_manifest_schema.py
test_files:
  - trainer/tests/test_manifest_validate.py
acceptance_criteria:
  - >
    `load_manifest(path) -> dict`: reads and json.loads()s the file at
    `path` (a `str` or `pathlib.Path`), returns the parsed dict as-is,
    no validation performed by this function itself.
  - >
    The four schema categories in scope, and the exact (schema_fields
    key, character/archetype field name) pairing for each: personality
    traits -- schema_fields["personality_traits"], checked against the
    KEYS of character["personality"] and archetype["personality_ranges"]
    dicts; occupations -- schema_fields["occupations"], checked against
    the VALUE of character["occupation"] and archetype["occupation"];
    species -- schema_fields["species_types"], checked against the VALUE
    of character["species"] and archetype["species"]; bonds --
    schema_fields["bond_types"], checked against the VALUE of
    character["bond"] and archetype["bond"]. Use exactly these four
    category names ("personality_traits", "occupations",
    "species_types", "bond_types") as dict keys in both functions below.
  - >
    `find_undeclared_values(manifest: dict) -> dict[str, set[str]]`:
    for each of the four categories, collects every value actually used
    across all entries in `manifest["characters"]` and
    `manifest["archetypes"]` (per the pairing above) that is NOT present
    in `manifest["schema_fields"][<the matching schema key>]`. Returns a
    dict with all four category names as keys; a category with no
    undeclared values maps to an empty set (always present, never
    omitted).
  - >
    `find_orphaned_declarations(manifest: dict) -> dict[str, set[str]]`:
    the reverse direction -- for each of the four categories, collects
    every value present in `manifest["schema_fields"][<key>]` that is
    NEVER used by any character or archetype entry (per the same
    pairing). Same all-four-keys-always-present shape as
    find_undeclared_values.
  - >
    `validate_manifest(manifest: dict) -> tuple[bool, list[str]]`: calls
    both functions above and combines them into `(True, [])` when both
    return all-empty-sets, or `(False, problems)` otherwise, where
    `problems` is a list of human-readable strings, one per non-empty
    category-direction combination, e.g.
    `"undeclared occupations: {'blacksmith2'}"` or
    `"orphaned bond_types: {'mentor'}"` (exact wording is the
    implementer's choice, but each string must name the direction
    -- undeclared vs orphaned -- and the category, and must be
    constructible/checkable by a test asserting on substrings, not
    exact string equality).
  - >
    trainer/tests/test_manifest_validate.py covers, at minimum: (a)
    running all three functions against the REAL
    manifests/dungeon_crawler.json (loaded via load_manifest) and
    asserting validate_manifest returns `(True, [])` -- the real
    manifest is expected to be clean today, so this is a real
    regression guard, not a synthetic-only test; (b) a synthetic
    manifest dict (small, hand-constructed in the test, not loaded from
    disk) with a character using an occupation NOT in
    schema_fields.occupations, asserting find_undeclared_values reports
    it under "occupations" and validate_manifest returns False with a
    problem string mentioning it; (c) a synthetic manifest with a
    schema_fields.bond_types entry that no character or archetype uses,
    asserting find_orphaned_declarations reports it under "bond_types";
    (d) a synthetic manifest where personality_traits, occupations,
    species_types, and bond_types are all fully consistent (every
    declared value used, every used value declared), asserting all
    three functions report clean/empty results.
verification:
  - "cd trainer && uv run pytest tests/test_manifest_validate.py -v"
```

## COMPLETION

```yaml
status: pending
summary:
files_changed: []
verification: |
risks: []
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 0 / 0 / no
- claude tokens spent (contract + review, est.) vs doing it directly:
- defects: caught in review = 0, slipped past review = 0
