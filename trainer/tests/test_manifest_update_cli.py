"""CLI wrapper around run_manifest_update (docs/milestones/m14.md section
1b). Fast toy runs on a tiny synthetic corpus, same scope discipline as
test_manifest_update.py -- these exercise the CLI's argument handling
and exit codes, not model quality."""
import json

from manifest_update_cli import main

CLEAN_MANIFEST = {
    "schema_fields": {
        "personality_traits": ["bold"], "occupations": ["warrior"],
        "species_types": ["human"], "bond_types": ["friend"],
    },
    "characters": [
        {"id": "char1", "personality": {"bold": 1}, "occupation": "warrior",
         "species": "human", "bond": "friend"}
    ],
    "archetypes": [],
}

DIRTY_MANIFEST = {
    "schema_fields": {
        "personality_traits": ["bold"], "occupations": ["warrior"],
        "species_types": ["human"], "bond_types": ["friend"],
    },
    "characters": [
        {"id": "char1", "personality": {"bold": 1}, "occupation": "warrior",
         "species": "human", "bond": "foe"}  # "foe" is undeclared
    ],
    "archetypes": [],
}

_CONFIG_TEMPLATE = '''
from pathlib import Path
from ngpt_trainer.manifest_update import ManifestUpdateConfig

TRAIN_PAIRS = [
    ("N:selena MOOD:happy ", "HELLO THERE FRIEND"),
    ("N:selena MOOD:sad ", "OH NO WHAT HAPPENED"),
    ("N:guard MOOD:happy ", "GOOD DAY CITIZEN"),
    ("N:guard MOOD:sad ", "MOVE ALONG NOW"),
    ("N:selena MOOD:cheerful ", "WHAT A LOVELY DAY"),
    ("N:guard MOOD:cheerful ", "ALL QUIET ON DUTY"),
]

CONFIG = ManifestUpdateConfig(
    manifest_path=Path({manifest_path!r}),
    generate_pairs=lambda: TRAIN_PAIRS,
    hidden=8, seed=0, val_fraction=0.34, device="cpu",
    max_epochs=3, patience=2, qat_max_epochs=2, qat_patience=1,
    agreement_min={agreement_min},
)
'''


def _write_manifest(tmp_path, manifest: dict):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def _write_config(tmp_path, manifest_path, agreement_min: float = 0.0):
    p = tmp_path / "config.py"
    p.write_text(_CONFIG_TEMPLATE.format(
        manifest_path=str(manifest_path), agreement_min=agreement_min))
    return p


def test_ships_prints_shipped_and_exits_0(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, CLEAN_MANIFEST)
    config_path = _write_config(tmp_path, manifest_path)

    exit_code = main(["manifest_update_cli.py", str(config_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "SHIPPED" in out
    assert "[PASS] agreement" in out


def test_refuses_at_validate_exits_1(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, DIRTY_MANIFEST)
    config_path = _write_config(tmp_path, manifest_path)

    exit_code = main(["manifest_update_cli.py", str(config_path)])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "REFUSED at validate" in out
    assert "UNDECLARED bond_types: ['foe']" in out


def test_refuses_at_acceptance_gates_exits_1(tmp_path, capsys):
    manifest_path = _write_manifest(tmp_path, CLEAN_MANIFEST)
    config_path = _write_config(tmp_path, manifest_path, agreement_min=2.0)

    exit_code = main(["manifest_update_cli.py", str(config_path)])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "REFUSED at acceptance_gates" in out
    assert "[FAIL] agreement" in out


def test_missing_config_module_exits_2(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.py"
    exit_code = main(["manifest_update_cli.py", str(missing)])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()


def test_config_module_without_config_exits_2(tmp_path, capsys):
    p = tmp_path / "empty_config.py"
    p.write_text("# no CONFIG defined here\n")
    exit_code = main(["manifest_update_cli.py", str(p)])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "config" in err.lower()
