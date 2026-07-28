#!/usr/bin/env python3
"""CLI wrapper around ngpt_trainer.manifest_validate, per M14's planned
manifest-update skill (docs/milestones/m14.md section 1b) -- that skill
needs a validate step it can call programmatically/from a script, not
just a pytest file (trainer/tests/test_manifest_validate.py already
covers the library itself; this is the thin CLI surface on top).

Usage:
  uv run python3 validate_manifest_cli.py [path/to/manifest.json]

Defaults to manifests/dungeon_crawler.json (this project's real, only
manifest) if no path is given. Exits 0 with "clean" if there are no
UNDECLARED values (see manifest_validate.py's own module doc for why
orphaned declarations alone don't fail validation -- they're legitimate
schema headroom, not an error). Exits 1 and prints every problem
(undeclared AND orphaned, both reported for visibility) otherwise.
"""
import sys
from pathlib import Path

from ngpt_trainer.manifest_validate import (find_orphaned_declarations,
                                            find_undeclared_values,
                                            load_manifest)

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "manifests" / "dungeon_crawler.json"


def main(argv: list[str]) -> int:
    manifest_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_MANIFEST
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = load_manifest(manifest_path)
    undeclared = find_undeclared_values(manifest)
    orphaned = find_orphaned_declarations(manifest)

    undeclared_problems = [(cat, vals) for cat, vals in undeclared.items() if vals]
    orphaned_problems = [(cat, vals) for cat, vals in orphaned.items() if vals]

    print(f"validating {manifest_path}")

    if not undeclared_problems and not orphaned_problems:
        print("clean: no undeclared values, no orphaned declarations")
        return 0

    for category, values in undeclared_problems:
        print(f"  UNDECLARED {category}: {sorted(values)}")
    for category, values in orphaned_problems:
        print(f"  orphaned (informational, not an error) {category}: {sorted(values)}")

    if undeclared_problems:
        print(f"\nFAIL: {len(undeclared_problems)} categor"
             f"{'y has' if len(undeclared_problems) == 1 else 'ies have'} "
             f"undeclared values -- something used isn't in the schema at all")
        return 1

    print(f"\nclean: orphaned declarations only (schema headroom, not an error)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
