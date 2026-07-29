"""The manifest-update skill (docs/milestones/m14.md section 1b): validate
an edit -> regenerate the affected corpus -> retrain -> run the full
acceptance suite -> ship (blob/goldens/selftest header) only if every
gate passes, refuse and report the specific failing gate otherwise.

Parameterized on project config/paths, not hardcoded to 64GPT's own
layout: corpus generation, the divergence-table methodology, and
"shipping" (writing a blob/goldens/selftest header) are all genuinely
project-specific -- M9's compositional grammar means corpus content is
hand-authored per archetype (guard_corpus.py, scifi_engineer_corpus.py,
...), not something this module can derive from a manifest generically.
This module owns the ORCHESTRATION (validate -> train -> gate -> ship-
or-refuse) and takes the project-specific pieces as callables via
ManifestUpdateConfig, so a second project (or a second archetype in
this one) can point it at their own manifest/corpus/ship-step without
editing this file -- the same portability discipline M14's own
manifest schema and corpus generator already follow.

Deliberately does NOT implement continued-training-from-checkpoint:
m14.md's own recorded decision (2026-07-28/29) is full retrain, always,
for goldens-never-drift reproducibility -- this always trains from
scratch.
"""
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from ngpt_trainer.capacity_monitor import exceeds_split_trigger, held_out_loss_for_subset
from ngpt_trainer.manifest_validate import find_undeclared_values, load_manifest
from ngpt_trainer.model import one_hot, qat_finetune, train_corpus_conditioned
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import gru_step
from ngpt_trainer.vocab import Vocab

Pairs = list[tuple[str, str]]


@dataclass
class CapacityCheck:
    """One named character's split-trigger check (m14.md's decided rule:
    float-phase held-out loss, NOT QAT -- see capacity_monitor.py for
    why). predicate matches the character's own prompts within val_pairs."""
    name: str
    predicate: Callable[[str], bool]
    baseline_loss: float
    threshold_pct: float = 5.0


@dataclass
class ManifestUpdateConfig:
    manifest_path: Path
    generate_pairs: Callable[[], Pairs]
    hidden: int
    seed: int = 0
    val_fraction: float = 0.15
    device: str = "cpu"
    max_epochs: int = 60
    patience: int = 6
    qat_max_epochs: int = 20
    qat_patience: int = 5
    agreement_probe: Callable[[], Pairs] | None = None
    agreement_min: float = 0.95
    divergence_fn: Callable[[object, Vocab], dict[str, float]] | None = None
    divergence_axis_min: float = 0.90
    capacity_checks: list[CapacityCheck] = field(default_factory=list)
    on_ship: Callable[[object, object, Vocab], None] | None = None


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class ManifestUpdateResult:
    shipped: bool
    refused_at: str | None = None
    validation_problems: dict | None = None
    gates: list[GateResult] = field(default_factory=list)
    float_val_loss: float | None = None
    qat_val_loss: float | None = None


def _held_out_split(pairs: Pairs, seed: int, val_fraction: float) -> tuple[Pairs, Pairs]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def _top1_agreement(model, q, vocab: Vocab, probe: Pairs) -> float:
    """Same methodology as make_m12_1_blob.py's top1_agreement (int8 vs
    float top-1 prediction match on held-out probe pairs) -- duplicated
    rather than imported because that module is a project-specific
    build script, not library code this skill should depend on."""
    match = total = 0
    with torch.no_grad():
        for prompt, response in probe:
            pids = vocab.encode(prompt)
            ids = pids + vocab.encode(response)
            flogits, _ = model(one_hot([vocab.eos_id] + ids, len(vocab)))
            fam = torch.argmax(flogits[0], dim=-1).tolist()
            h = np.zeros(q.H, dtype=np.int64)
            for pos, x in enumerate([vocab.eos_id] + ids):
                h, logits = gru_step(q, h, x)
                if pos >= len(pids):
                    match += int(np.argmax(logits)) == fam[pos]
                    total += 1
    return match / total if total else 1.0


def run_manifest_update(config: ManifestUpdateConfig) -> ManifestUpdateResult:
    manifest = load_manifest(config.manifest_path)
    undeclared = find_undeclared_values(manifest)
    problems = {cat: vals for cat, vals in undeclared.items() if vals}
    if problems:
        return ManifestUpdateResult(shipped=False, refused_at="validate",
                                    validation_problems=problems)

    pairs = config.generate_pairs()
    train_pairs, val_pairs = _held_out_split(pairs, config.seed, config.val_fraction)
    vocab = Vocab.from_text("".join(p + r for p, r in pairs))

    model = train_corpus_conditioned(
        train_pairs, val_pairs, vocab, hidden=config.hidden, seed=config.seed,
        max_epochs=config.max_epochs, patience=config.patience, device=config.device)
    float_val_loss = model.final_loss

    gates: list[GateResult] = []
    for check in config.capacity_checks:
        current = held_out_loss_for_subset(model, val_pairs, vocab, check.predicate)
        if current is None:
            gates.append(GateResult(
                f"capacity[{check.name}]", False,
                "predicate matched no val pairs -- cannot check"))
        else:
            over = exceeds_split_trigger(check.baseline_loss, current, check.threshold_pct)
            gates.append(GateResult(
                f"capacity[{check.name}]", not over,
                f"held-out loss {current:.4f} vs baseline {check.baseline_loss:.4f} "
                f"(threshold {check.threshold_pct}%)"))

    if not all(g.passed for g in gates):
        # A failed capacity gate already decides the outcome -- skip the
        # QAT fine-tune (the single most expensive remaining step) rather
        # than spend real GPU time computing agreement/divergence numbers
        # for a manifest edit that's refused regardless of what they say.
        return ManifestUpdateResult(
            shipped=False, refused_at="acceptance_gates", gates=gates,
            float_val_loss=float_val_loss)

    model = qat_finetune(
        model, train_pairs, val_pairs, vocab, seed=config.seed,
        max_epochs=config.qat_max_epochs, patience=config.qat_patience,
        device=config.device)
    qat_val_loss = model.final_loss

    q = quantize(model)

    probe = config.agreement_probe() if config.agreement_probe else val_pairs
    agreement = _top1_agreement(model, q, vocab, probe)
    gates.append(GateResult(
        "agreement", agreement >= config.agreement_min,
        f"{agreement:.4f} (min {config.agreement_min})"))

    if config.divergence_fn is not None:
        table = config.divergence_fn(q, vocab)
        for axis, score in table.items():
            gates.append(GateResult(
                f"divergence[{axis}]", score >= config.divergence_axis_min,
                f"{score:.4f} (min {config.divergence_axis_min})"))

    if not all(g.passed for g in gates):
        return ManifestUpdateResult(
            shipped=False, refused_at="acceptance_gates", gates=gates,
            float_val_loss=float_val_loss, qat_val_loss=qat_val_loss)

    if config.on_ship is not None:
        config.on_ship(model, q, vocab)

    return ManifestUpdateResult(
        shipped=True, gates=gates,
        float_val_loss=float_val_loss, qat_val_loss=qat_val_loss)
