import json

from validate_manifest_cli import main


def test_real_manifest_exits_clean(capsys):
    exit_code = main(["validate_manifest_cli.py"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "clean" in out.lower()


def test_manifest_with_undeclared_value_exits_1(tmp_path, capsys):
    manifest = {
        "schema_fields": {"personality_traits": [], "occupations": ["warrior"],
                          "species_types": [], "bond_types": []},
        "characters": [{"id": "c1", "occupation": "typo_occupation"}],
        "archetypes": []
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(manifest))

    exit_code = main(["validate_manifest_cli.py", str(p)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "UNDECLARED" in out
    assert "typo_occupation" in out


def test_manifest_with_only_orphaned_still_exits_0(tmp_path, capsys):
    manifest = {
        "schema_fields": {"personality_traits": [], "occupations": ["unused_occ"],
                          "species_types": [], "bond_types": []},
        "characters": [],
        "archetypes": []
    }
    p = tmp_path / "orphaned_only.json"
    p.write_text(json.dumps(manifest))

    exit_code = main(["validate_manifest_cli.py", str(p)])
    assert exit_code == 0  # orphaned alone is not a failure
    out = capsys.readouterr().out
    assert "orphaned" in out.lower()


def test_missing_manifest_file_exits_2(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.json"
    exit_code = main(["validate_manifest_cli.py", str(missing)])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "not found" in err.lower()
