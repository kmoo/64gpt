#!/usr/bin/env python3
"""M14 portability proof (docs/milestones/m14.md section 2): "port one
archetype to a deliberately different genre and confirm it works
through the unmodified toolkit... a sci-fi 'engineer' instead of a
fantasy 'guard'." This is that experiment.

Deliberately lightweight (m14.md section 2's own framing), not a full
production retrain: guard (existing, familiar baseline) + the new
scifi_engineer_corpus.py archetype, trained together through the PLAIN
(non-attr) train_corpus_conditioned/qat_finetune pair -- the actual
shipped pipeline shape (make_m12_1_blob.py's own path), not the
experimental attribute-embedding mechanism M12.3-M13 use (that's a
separate, orthogonal question from portability). Zero core/,
npc_service.py, or training-pipeline changes; only new corpus content
and new field VALUES go in.

Isolated experiment: own checkpoint dir, NOT the shipped model.bin or
manifests/dungeon_crawler.json -- a mid-experiment failure can't put
the real game's corpus at risk. checkpoint_path (this overnight
session's own new capability) IS used here, both as a real recovery net
and as a dogfooding check that the plain pair's checkpointing actually
works under real, not toy-scale, training.

Held-out split: a simple seeded random 15% holdout per archetype, NOT
M13's rigorous combo-level pre-registered protocol -- this is a
lightweight generalization proof, not a scientific measurement with a
pre-registered bar. Stated plainly rather than overclaimed.

Run from trainer/ (after M13's own GPU work is completely finished --
one MPS job at a time, never alongside opencoder either):
  uv run python3 m14_portability_proof.py
"""
import json
import random
from pathlib import Path

from ngpt_trainer import guard_corpus as gc
from ngpt_trainer import scifi_engineer_corpus as sec
from ngpt_trainer.model import qat_finetune, train_corpus_conditioned
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled
from ngpt_trainer.vocab import Vocab

import make_m12_1_blob as m12

SEED = 0
HIDDEN = 320  # same H as the shipped model -- capacity isn't the question here
VAL_FRACTION = 0.15
PROBE_SEED = 0xC0FFEE
LINES_PER_GROUP = 8
CHECKPOINT_DIR = Path(__file__).resolve().parent / "m14_portability_checkpoints"
RESULTS_DIR = Path(__file__).resolve().parent / "m14_portability_results"


def held_out_split(pairs: list[tuple[str, str]], seed: int, val_fraction: float):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def run_probe(q, vocab, groups: dict, corpus_vocab: set) -> dict:
    results = {}
    tot_inv = tot_lines = 0
    for name, pairs in groups.items():
        rng = random.Random(PROBE_SEED)
        sample = rng.sample(pairs, min(LINES_PER_GROUP, len(pairs)))
        inv = 0
        for i, (prompt, _) in enumerate(sample):
            got = generate_sampled(q, vocab, prompt, seed=PROBE_SEED + i,
                                   inv_t_q8=m12.INV_T_Q8, top_k=m12.TOP_K,
                                   max_len=m12.MAX_GOLDEN_LEN)
            inv += m12.invented_word_count(got, corpus_vocab)
        k = len(sample)
        results[name] = {"lines": k, "invented_per_line": inv / k}
        tot_inv += inv
        tot_lines += k
    results["ALL"] = {"lines": tot_lines, "invented_per_line": tot_inv / tot_lines}
    return results


def main():
    guard_pairs = gc.generate_pairs(seed=SEED, per_combo=3)
    engineer_pairs = sec.generate_pairs(seed=SEED, per_combo=3)

    guard_train, guard_val = held_out_split(guard_pairs, SEED, VAL_FRACTION)
    engineer_train, engineer_val = held_out_split(engineer_pairs, SEED + 1, VAL_FRACTION)

    train_pairs = guard_train + engineer_train
    val_pairs = guard_val + engineer_val

    full_text = "".join(p + r for p, r in guard_pairs + engineer_pairs)
    vocab = Vocab.from_text(full_text)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    float_ckpt = str(CHECKPOINT_DIR / "float.pt")
    qat_ckpt = str(CHECKPOINT_DIR / "qat.pt")

    print(f"training: {len(train_pairs)} train / {len(val_pairs)} val pairs "
          f"(guard {len(guard_pairs)}, engineer {len(engineer_pairs)}), "
          f"vocab size {len(vocab)}, hidden={HIDDEN}, device=mps")
    model = train_corpus_conditioned(
        train_pairs, val_pairs, vocab, hidden=HIDDEN, seed=SEED,
        max_epochs=120, patience=12, device="mps", checkpoint_path=float_ckpt)
    float_val_loss = model.final_loss
    print(f"float phase done: val loss {float_val_loss:.4f}")

    model = qat_finetune(
        model, train_pairs, val_pairs, vocab, seed=SEED, lr=3e-4,
        max_epochs=30, patience=6, device="mps", checkpoint_path=qat_ckpt)
    qat_val_loss = model.final_loss
    print(f"qat done: val loss {qat_val_loss:.4f}")

    q = quantize(model)
    corpus_vocab = m12.build_corpus_vocab(full_text)
    groups = {"guard": guard_pairs, "engineer": engineer_pairs}
    results = run_probe(q, vocab, groups, corpus_vocab)

    print("\nresults:")
    for name, r in results.items():
        print(f"  {name:<10}{r['lines']:>5} lines  {r['invented_per_line']:>6.2f} inv/line")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "seed": SEED, "hidden": HIDDEN,
        "float_val_loss": float_val_loss, "qat_val_loss": qat_val_loss,
        "per_group": results,
    }
    out_path = RESULTS_DIR / "result.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
