#!/usr/bin/env python3
"""CLI wrapper around ngpt_trainer.manifest_update.run_manifest_update
(docs/milestones/m14.md section 1b) -- the last real gap the milestone's
own honest scope note flagged: the skill existed as a library function
callers had to script by hand, with no one-command surface matching
validate_manifest_cli.py's existing pattern.

run_manifest_update() takes Python callables (generate_pairs,
divergence_fn, on_ship) that a manifest path alone can't express on a
command line -- corpus generation and shipping are genuinely
project-specific (see manifest_update.py's own module doc). So this CLI
doesn't take a manifest path directly; it takes a PYTHON MODULE that
builds the ManifestUpdateConfig (wiring its own project's corpus
generator, divergence method, and ship step), the same "point this at
your own config" shape a second project would need anyway.

Usage:
  uv run python3 manifest_update_cli.py path/to/config_module.py

The config module must define a module-level CONFIG:
ManifestUpdateConfig. Exit 0 and "SHIPPED" if every gate passed, exit 1
and "REFUSED at <step>" with the specific failing gate(s) otherwise,
exit 2 on a bad path/missing CONFIG (a usage error, not a refusal).
"""
import importlib.util
import sys
from pathlib import Path

from ngpt_trainer.manifest_update import ManifestUpdateConfig, run_manifest_update


def _load_config(module_path: Path) -> ManifestUpdateConfig:
    spec = importlib.util.spec_from_file_location("manifest_update_target", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: manifest_update_cli.py path/to/config_module.py", file=sys.stderr)
        return 2

    module_path = Path(argv[1])
    if not module_path.exists():
        print(f"error: config module not found: {module_path}", file=sys.stderr)
        return 2

    try:
        config = _load_config(module_path)
    except AttributeError:
        print(f"error: {module_path} does not define a module-level CONFIG",
              file=sys.stderr)
        return 2

    print(f"running manifest update for {config.manifest_path}")
    result = run_manifest_update(config)

    if result.refused_at == "validate":
        print("REFUSED at validate -- no training attempted")
        for category, values in result.validation_problems.items():
            print(f"  UNDECLARED {category}: {sorted(values)}")
        return 1

    for gate in result.gates:
        status = "PASS" if gate.passed else "FAIL"
        print(f"  [{status}] {gate.name}: {gate.detail}")

    if result.shipped:
        print(f"SHIPPED (float val loss {result.float_val_loss:.4f}, "
              f"qat val loss {result.qat_val_loss:.4f})")
        return 0

    print("REFUSED at acceptance_gates -- see failing gate(s) above, nothing shipped")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
