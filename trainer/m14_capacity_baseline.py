#!/usr/bin/env python3
"""Establishes the real baseline number M14's capacity split-trigger
needs (docs/milestones/m14.md: "not yet operational... the concrete
next step is running held_out_loss_for_subset() against Selena's and
the bad guy's own val-pair subsets on a fresh reproduction of the M11.1
baseline config").

Reproduces make_m12_1_blob.py's corpus config exactly -- M12.1, NOT
M11.1: M12.1 is the actual best/shipped model (QAT + corpus rebalance
+ min-p + lexicon-trie decode guard, docs/milestones/m12.1.md), while
M11.1 predates that entire fix stack and was never shipped as-is.
Reusing M11.1's raw, unrebalanced corpus mix would establish a
baseline against a config this project doesn't actually run today.
Same PER_COMBO values as make_m12_1_blob.py (PER_COMBO=96,
GUARD_PER_COMBO=24, CAST_PER_COMBO=12, SHADEWRATH_PER_COMBO=48,
KORRATH_PER_COMBO=48, PRINCESS_PER_COMBO=48, HOLDOUT_COMBOS=20,
LORE_BANK_ENABLED=False).

One real difference from make_m12_1_blob.py's own combo_split(): that
script's val split holds out combos from Selena ONLY (it only ever
needed to test Selena's own generalization). This script ALSO holds
out combos from Shadewrath ("the bad guy"), because computing HIS
held-out loss needs held-out data of his own.

Float-phase only, deliberately -- m14.md's own recorded decision is
that the split-trigger is defined on FLOAT loss, not QAT (QAT's
~29%-seed-noise floor, M13, would swamp a percentage trigger). This
means the cached `.m12_1_model.pt` checkpoint can't be reused: it
stores POST-QAT weights (get_model()'s cache), and QAT deliberately
lets float weights drift, so a QAT-phase model's per-character loss is
not the float-phase number this baseline needs. This script retrains
the float phase fresh instead -- real, no shortcuts.

Isolated experiment: own results dir, does NOT touch game/rawfs/
model.bin, manifests/dungeon_crawler.json, .m12_1_model.pt, or any
other shipped artifact -- same precedent as every other
m*_experiment.py in this directory.

Run (from trainer/, after confirming via `claude agents --json --cwd
$PWD` and `pgrep -f mlx_lm.server` that nothing else is on the GPU):
  uv run python3 m14_capacity_baseline.py
"""
import json
import random
import time
from pathlib import Path

from ngpt_trainer import cast_corpus as cc
from ngpt_trainer import guard_corpus as gc
from ngpt_trainer import korrath_corpus as kc
from ngpt_trainer import princess_corpus as pc
from ngpt_trainer import selena_corpus as sc
from ngpt_trainer import shadewrath_corpus as swc
from ngpt_trainer.capacity_monitor import held_out_loss_for_subset
from ngpt_trainer.model import train_corpus_conditioned
from ngpt_trainer.npc_service import parse_prompt_fields
from ngpt_trainer.vocab import Vocab

SEED = 0
HIDDEN = 320
# make_m12_1_blob.py's exact shipped values -- M12.1, not M11.1 (see module doc).
PER_COMBO = 96
GUARD_PER_COMBO = 24
CAST_PER_COMBO = 12
SHADEWRATH_PER_COMBO = 48
KORRATH_PER_COMBO = 48
PRINCESS_PER_COMBO = 48
SELENA_HOLDOUT_COMBOS = 20
SHADEWRATH_HOLDOUT_FRACTION = 0.2  # shadewrath has far fewer distinct combos than Selena
DEVICE = "mps"
RESULTS_DIR = Path(__file__).resolve().parent / "m14_capacity_baseline_results"


def generic_combo_key(prompt: str) -> tuple:
    """(R, M, C) straight off the prompt string, no character-specific
    tier-name translation (unlike selena_corpus.combo_key) -- works for
    any character's prompt built via npc_service.prompt_fields(), via
    that module's own parse_prompt_fields() parser rather than a
    reimplementation of it."""
    fields = parse_prompt_fields(prompt)
    return (fields.get("R"), fields.get("M"), fields.get("C"))


def combo_split(pairs: list[tuple[str, str]], n_holdout: int, seed: int):
    all_combos = sorted({generic_combo_key(p) for p, _ in pairs})
    rng = random.Random(seed)
    held = set(rng.sample(all_combos, min(n_holdout, len(all_combos))))
    train, val = [], []
    for p, r in pairs:
        (val if generic_combo_key(p) in held else train).append((p, r))
    return train, val


def main():
    t0 = time.time()

    selena_pairs = sc.generate_pairs(seed=SEED, per_combo=PER_COMBO)
    guard_pairs = gc.generate_pairs(seed=SEED, per_combo=GUARD_PER_COMBO)
    cast_pairs = cc.generate_pairs(seed=SEED, per_combo=CAST_PER_COMBO)
    cc.assert_no_holdout_leak(cast_pairs)
    shadewrath_pairs = swc.generate_pairs(seed=SEED, per_combo=SHADEWRATH_PER_COMBO,
                                          lore_bank_enabled=False)
    korrath_pairs = kc.generate_pairs(seed=SEED, per_combo=KORRATH_PER_COMBO,
                                      lore_bank_enabled=False)
    princess_pairs = pc.generate_pairs(seed=SEED, per_combo=PRINCESS_PER_COMBO,
                                       lore_bank_enabled=False)

    selena_train, selena_val = combo_split(selena_pairs, SELENA_HOLDOUT_COMBOS, SEED + 777)
    n_shadewrath_combos = len({generic_combo_key(p) for p, _ in shadewrath_pairs})
    shadewrath_holdout_n = max(1, int(n_shadewrath_combos * SHADEWRATH_HOLDOUT_FRACTION))
    shadewrath_train, shadewrath_val = combo_split(
        shadewrath_pairs, shadewrath_holdout_n, SEED + 888)

    train_pairs = (selena_train + guard_pairs + cast_pairs + shadewrath_train
                  + korrath_pairs + princess_pairs)
    val_pairs = selena_val + shadewrath_val

    full_text = "".join(p + r for p, r in
                        selena_pairs + guard_pairs + cast_pairs + shadewrath_pairs
                        + korrath_pairs + princess_pairs)
    vocab = Vocab.from_text(full_text)

    print(f"train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)} "
         f"(selena held-out: {len(selena_val)}, shadewrath held-out: {len(shadewrath_val)} "
         f"of {n_shadewrath_combos} distinct combos)")
    print(f"vocab size {len(vocab)}, hidden={HIDDEN}, device={DEVICE}")

    model = train_corpus_conditioned(
        train_pairs, val_pairs, vocab, hidden=HIDDEN, seed=SEED,
        max_epochs=120, patience=12, device=DEVICE)
    aggregate_val_loss = model.final_loss
    print(f"trained in {time.time()-t0:.0f}s, aggregate val loss {aggregate_val_loss:.4f}")

    selena_loss = held_out_loss_for_subset(
        model, val_pairs, vocab, lambda p: "OCC:companion" in p)
    shadewrath_loss = held_out_loss_for_subset(
        model, val_pairs, vocab, lambda p: "OCC:villain" in p)

    print(f"\nSelena (OCC:companion) held-out float val loss:     {selena_loss:.4f}")
    print(f"Shadewrath (OCC:villain) held-out float val loss:   {shadewrath_loss:.4f}")
    print("\nThese are the baseline_loss numbers for CapacityCheck(...) going forward "
         "(docs/milestones/m14.md's 5% split-trigger).")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "seed": SEED, "hidden": HIDDEN, "device": DEVICE,
        "aggregate_val_loss": aggregate_val_loss,
        "selena_held_out_float_val_loss": selena_loss,
        "shadewrath_held_out_float_val_loss": shadewrath_loss,
        "selena_holdout_combos": SELENA_HOLDOUT_COMBOS,
        "shadewrath_holdout_combos": shadewrath_holdout_n,
        "shadewrath_total_combos": n_shadewrath_combos,
        "seconds": time.time() - t0,
    }
    out_path = RESULTS_DIR / "result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
