"""M14's manifest-update skill (docs/milestones/m14.md section 1b):
validate -> regenerate corpus -> retrain -> verify (acceptance gates) ->
ship-or-refuse. Fast toy runs on tiny synthetic corpora/manifests, like
every other model.py-adjacent test in this suite -- these prove the
ORCHESTRATION logic (refuse-before-training on a bad manifest, refuse-
before-shipping on a failing gate, ship only when everything passes),
not production-scale training behavior."""
import json

from ngpt_trainer.manifest_update import (
    CapacityCheck,
    ManifestUpdateConfig,
    run_manifest_update,
)

TRAIN_PAIRS = [
    ("N:selena MOOD:happy ", "HELLO THERE FRIEND"),
    ("N:selena MOOD:sad ", "OH NO WHAT HAPPENED"),
    ("N:guard MOOD:happy ", "GOOD DAY CITIZEN"),
    ("N:guard MOOD:sad ", "MOVE ALONG NOW"),
    ("N:selena MOOD:cheerful ", "WHAT A LOVELY DAY"),
    ("N:guard MOOD:cheerful ", "ALL QUIET ON DUTY"),
]

CLEAN_MANIFEST = {
    "schema_fields": {
        "personality_traits": ["bold"],
        "occupations": ["warrior"],
        "species_types": ["human"],
        "bond_types": ["friend"],
    },
    "characters": [
        {"id": "char1", "personality": {"bold": 1}, "occupation": "warrior",
         "species": "human", "bond": "friend"}
    ],
    "archetypes": [],
}

DIRTY_MANIFEST = {
    "schema_fields": {
        "personality_traits": ["bold"],
        "occupations": ["warrior"],
        "species_types": ["human"],
        "bond_types": ["friend"],
    },
    "characters": [
        {"id": "char1", "personality": {"bold": 1}, "occupation": "warrior",
         "species": "human", "bond": "foe"}  # "foe" is undeclared
    ],
    "archetypes": [],
}


def _write(tmp_path, manifest: dict):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def _base_config(tmp_path, manifest: dict, **overrides) -> ManifestUpdateConfig:
    calls = {"generate_pairs": 0}

    def generate_pairs():
        calls["generate_pairs"] += 1
        return TRAIN_PAIRS

    config = ManifestUpdateConfig(
        manifest_path=_write(tmp_path, manifest),
        generate_pairs=generate_pairs,
        hidden=8, seed=0, val_fraction=0.34, device="cpu",
        max_epochs=3, patience=2, qat_max_epochs=2, qat_patience=1,
        agreement_min=0.0,  # always passes -- this suite tests orchestration, not model quality
    )
    for k, v in overrides.items():
        setattr(config, k, v)
    return config, calls


def test_refuses_at_validate_without_training(tmp_path):
    config, calls = _base_config(tmp_path, DIRTY_MANIFEST)
    result = run_manifest_update(config)

    assert result.shipped is False
    assert result.refused_at == "validate"
    assert result.validation_problems == {"bond_types": {"foe"}}
    assert calls["generate_pairs"] == 0  # never even generated the corpus, let alone trained


def test_ships_when_all_gates_pass(tmp_path):
    shipped_with = []

    def on_ship(model, q, vocab):
        shipped_with.append((model, q, vocab))

    config, calls = _base_config(tmp_path, CLEAN_MANIFEST, on_ship=on_ship)
    result = run_manifest_update(config)

    assert result.shipped is True
    assert result.refused_at is None
    assert calls["generate_pairs"] == 1
    assert len(shipped_with) == 1
    assert all(g.passed for g in result.gates)
    assert isinstance(result.float_val_loss, float)
    assert isinstance(result.qat_val_loss, float)


def test_refuses_at_acceptance_gates_without_shipping(tmp_path):
    shipped_with = []

    def on_ship(model, q, vocab):
        shipped_with.append((model, q, vocab))

    config, calls = _base_config(
        tmp_path, CLEAN_MANIFEST, on_ship=on_ship,
        agreement_min=2.0)  # impossible to satisfy -- forces this gate to fail
    result = run_manifest_update(config)

    assert result.shipped is False
    assert result.refused_at == "acceptance_gates"
    assert shipped_with == []  # ship step never runs when a gate fails
    agreement_gates = [g for g in result.gates if g.name == "agreement"]
    assert len(agreement_gates) == 1
    assert agreement_gates[0].passed is False


def test_capacity_check_gate_can_fail_and_blocks_shipping(tmp_path):
    shipped_with = []
    check = CapacityCheck(
        name="selena", predicate=lambda prompt: "N:selena" in prompt,
        baseline_loss=1e-6,  # any real loss will look like enormous degradation vs this
        threshold_pct=5.0)

    config, calls = _base_config(
        tmp_path, CLEAN_MANIFEST, on_ship=lambda m, q, v: shipped_with.append(1),
        capacity_checks=[check])
    result = run_manifest_update(config)

    assert result.shipped is False
    capacity_gates = [g for g in result.gates if g.name == "capacity[selena]"]
    assert len(capacity_gates) == 1
    assert capacity_gates[0].passed is False
    assert shipped_with == []


def test_divergence_gate_can_fail_and_blocks_shipping(tmp_path):
    def fake_divergence(q, vocab):
        return {"mood": 0.10}  # deliberately below any reasonable min

    config, calls = _base_config(
        tmp_path, CLEAN_MANIFEST, divergence_fn=fake_divergence,
        divergence_axis_min=0.90)
    result = run_manifest_update(config)

    assert result.shipped is False
    divergence_gates = [g for g in result.gates if g.name == "divergence[mood]"]
    assert len(divergence_gates) == 1
    assert divergence_gates[0].passed is False
