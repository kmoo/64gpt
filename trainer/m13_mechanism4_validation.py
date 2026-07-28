#!/usr/bin/env python3
"""M13 mechanism 4 (docs/milestones/m13.md): does judge score predict
real training outcome, on guard/korrath -- M12.4's two laggards
(1.75 / 2.00 inv/line vs the <=1.00 gate). Same architecture/corpus/
training code as trainer/m12_4_attr_ablation_spike.py (attribute
conditioning, D:/M: stripped from the text prefix); this script adds:

  --seed N          override the TRAINING seed only (weight init +
                     batch order). Corpus generation stays on m12.SEED
                     always -- "same corpus, different seed" per the
                     pre-registered protocol's noise-floor measurement.
  --extra-fragments PATH
                     JSON file mapping "GUARD_ID_MOOD" or "KORRATH_MOOD"
                     to a list of new opener lines to splice into that
                     bank before corpus generation (the approved/
                     rejected arms). Omit for the baseline arm.
  --checkpoint-dir DIR
                     mid-run checkpoints (model.py's checkpoint_path),
                     protects a long MPS run against a process crash.

Host-only: fp32 train + int8 quantize()/ref_impl, no blob/kernel/ROM
artifacts, no on-disk model cache reused across runs (each invocation
is a fresh, independent training run by design -- this measures
run-to-run variance, a cache would defeat the purpose).

Run from trainer/:
  uv run python3 m13_mechanism4_validation.py --seed 0 --label baseline_seedA
  uv run python3 m13_mechanism4_validation.py --seed 1 --label baseline_seedB
"""
import argparse
import json
import random
import zlib
from pathlib import Path

import torch

from ngpt_trainer import cast_corpus as cc
from ngpt_trainer import guard_corpus as gc
from ngpt_trainer import korrath_corpus as kc
from ngpt_trainer import princess_corpus as pc
from ngpt_trainer import selena_corpus as sc
from ngpt_trainer import shadewrath_corpus as swc
from ngpt_trainer.model import qat_finetune_attr, train_corpus_conditioned_attr
from ngpt_trainer.npc_service import parse_prompt_fields, strip_prompt_fields
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled_attr
from ngpt_trainer.vocab import Vocab

import make_m12_1_blob as m12

STRIPPED_KEYS = {"D", "M"}
LINES_PER_GROUP = 8
PROBE_SEED = 0xC0FFEE
RESULTS_DIR = Path(__file__).resolve().parent / "m13_mechanism4_results"


def apply_extra_fragments(extra_path: str | None) -> None:
    """Monkeypatches guard_corpus._OPENERS / korrath_corpus._OPENERS
    in-process, BEFORE any corpus_text()/generate_pairs() call reads
    them, so the fragments actually appear in this run's training data.
    Key format: "GUARD_1001_WORRIED" -> guard_corpus._OPENERS["guard#1001"]["worried"],
    "KORRATH_CHEERFUL" -> korrath_corpus._OPENERS["cheerful"]."""
    if extra_path is None:
        return
    extra = json.loads(Path(extra_path).read_text())
    for key, lines in extra.items():
        parts = key.split("_")
        mood = parts[-1].lower()
        who = "_".join(parts[:-1])
        if who.startswith("GUARD_"):
            guard_id = f"guard#{who.split('_')[1]}"
            existing = gc._OPENERS[guard_id][mood]
            gc._OPENERS[guard_id] = {**gc._OPENERS[guard_id],
                                     mood: existing + tuple(lines)}
        elif who == "KORRATH":
            existing = kc._OPENERS[mood]
            kc._OPENERS[mood] = existing + tuple(lines)
        else:
            raise ValueError(f"unrecognized extra-fragments key: {key!r}")


def build_attr_vocabs(pairs):
    descs, moods = set(), set()
    for prompt, _ in pairs:
        fields = parse_prompt_fields(prompt)
        descs.add(fields["D"])
        moods.add(fields["M"])
    return ({v: i for i, v in enumerate(sorted(descs))},
            {v: i for i, v in enumerate(sorted(moods))})


def attrs_for(pairs, desc_by_value, mood_by_value):
    out = []
    for prompt, _ in pairs:
        fields = parse_prompt_fields(prompt)
        out.append((desc_by_value[fields["D"]], mood_by_value[fields["M"]]))
    return out


def strip_pairs(pairs):
    return [(strip_prompt_fields(p, STRIPPED_KEYS), r) for p, r in pairs]


def attr_cols_for(original_prompt, V, n_desc, desc_by_value, mood_by_value):
    fields = parse_prompt_fields(original_prompt)
    return (V + desc_by_value[fields["D"]],
            V + n_desc + mood_by_value[fields["M"]])


def run_probe(q, vocab, groups, corpus_vocab, V, n_desc, desc_by_value, mood_by_value):
    results = {}
    tot_inv = tot_lines = 0
    for name, pairs in groups.items():
        rng = random.Random(PROBE_SEED ^ (zlib.crc32(name.encode()) & 0xFFFF))
        sample = rng.sample(pairs, min(LINES_PER_GROUP, len(pairs)))
        inv = 0
        for i, (prompt, _) in enumerate(sample):
            attr_cols = attr_cols_for(prompt, V, n_desc, desc_by_value, mood_by_value)
            stripped_prompt = strip_prompt_fields(prompt, STRIPPED_KEYS)
            got = generate_sampled_attr(q, vocab, stripped_prompt, attr_cols=attr_cols,
                                        seed=PROBE_SEED + i, inv_t_q8=m12.INV_T_Q8,
                                        top_k=m12.TOP_K, max_len=m12.MAX_GOLDEN_LEN)
            inv += m12.invented_word_count(got, corpus_vocab)
        k = len(sample)
        results[name] = {"lines": k, "invented_per_line": inv / k}
        tot_inv += inv
        tot_lines += k
    results["ALL"] = {"lines": tot_lines, "invented_per_line": tot_inv / tot_lines}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True,
                     help="training seed (weight init + batch order); "
                          "corpus generation always uses m12.SEED")
    ap.add_argument("--label", required=True,
                     help="run label, e.g. baseline_seedA / approved / rejected")
    ap.add_argument("--extra-fragments", default=None)
    ap.add_argument("--checkpoint-dir", default=None)
    args = ap.parse_args()

    apply_extra_fragments(args.extra_fragments)

    (selena_pairs, guard_pairs, cast_pairs, shadewrath_pairs,
     korrath_pairs, princess_pairs) = m12.build_all_pairs()
    train_pairs, val_pairs, held_combos = m12.combo_split(selena_pairs)
    all_train = (train_pairs + guard_pairs + cast_pairs
                 + shadewrath_pairs + korrath_pairs + princess_pairs)
    full_text = (sc.corpus_text(seed=m12.SEED, per_combo=m12.PER_COMBO)
                 + "".join(p + r for p, r in guard_pairs)
                 + cc.corpus_text(seed=m12.SEED, per_combo=m12.CAST_PER_COMBO)
                 + swc.corpus_text(seed=m12.SEED, per_combo=m12.SHADEWRATH_PER_COMBO,
                                   lore_bank_enabled=m12.LORE_BANK_ENABLED)
                 + kc.corpus_text(seed=m12.SEED, per_combo=m12.KORRATH_PER_COMBO,
                                  lore_bank_enabled=m12.LORE_BANK_ENABLED)
                 + pc.corpus_text(seed=m12.SEED, per_combo=m12.PRINCESS_PER_COMBO,
                                  lore_bank_enabled=m12.LORE_BANK_ENABLED))
    vocab = Vocab.from_text(full_text)
    V = len(vocab)

    desc_by_value, mood_by_value = build_attr_vocabs(all_train + val_pairs)
    n_desc, n_mood = len(desc_by_value), len(mood_by_value)

    train_attrs = attrs_for(all_train, desc_by_value, mood_by_value)
    val_attrs = attrs_for(val_pairs, desc_by_value, mood_by_value)
    stripped_train = strip_pairs(all_train)
    stripped_val = strip_pairs(val_pairs)

    ckpt = None
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ckpt = str(Path(args.checkpoint_dir) / f"{args.label}_float.pt")

    print(f"[{args.label}] training seed={args.seed} (corpus seed fixed at "
          f"m12.SEED={m12.SEED}), device=mps, checkpoint={ckpt}")
    model = train_corpus_conditioned_attr(
        stripped_train, stripped_val, train_attrs, val_attrs, vocab,
        n_desc=n_desc, n_mood=n_mood, hidden=m12.HIDDEN, seed=args.seed,
        max_epochs=120, patience=12, device="mps", checkpoint_path=ckpt)
    float_val = model.final_loss
    print(f"[{args.label}] float phase done: val loss {float_val:.4f}")

    qat_ckpt = str(Path(args.checkpoint_dir) / f"{args.label}_qat.pt") if ckpt else None
    model = qat_finetune_attr(
        model, stripped_train, stripped_val, train_attrs, val_attrs, vocab,
        n_desc=n_desc, n_mood=n_mood, seed=args.seed, lr=3e-4,
        max_epochs=30, patience=6, device="mps", checkpoint_path=qat_ckpt)
    print(f"[{args.label}] qat done: val loss {model.final_loss:.4f}")

    q = quantize(model)
    corpus_vocab = m12.build_corpus_vocab(full_text)
    groups = {"selena": selena_pairs, "guard": guard_pairs, "cast": cast_pairs,
              "shadewrath": shadewrath_pairs, "korrath": korrath_pairs,
              "elowen": princess_pairs}
    results = run_probe(q, vocab, groups, corpus_vocab, V, n_desc,
                        desc_by_value, mood_by_value)

    guard_korrath_inv = (results["guard"]["invented_per_line"] * results["guard"]["lines"]
                          + results["korrath"]["invented_per_line"] * results["korrath"]["lines"])
    guard_korrath_lines = results["guard"]["lines"] + results["korrath"]["lines"]
    guard_korrath_rate = guard_korrath_inv / guard_korrath_lines

    print(f"\n[{args.label}] results:")
    for name, r in results.items():
        print(f"  {name:<12}{r['lines']:>5} lines  {r['invented_per_line']:>6.2f} inv/line")
    print(f"  {'guard+korrath':<12}{guard_korrath_lines:>5} lines  {guard_korrath_rate:>6.2f} inv/line  <-- the metric mechanism 4's bar uses")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "label": args.label, "seed": args.seed,
        "extra_fragments_path": args.extra_fragments,
        "float_val_loss": float_val, "qat_val_loss": model.final_loss,
        "per_group": results, "guard_korrath_inv_per_line": guard_korrath_rate,
    }
    out_path = RESULTS_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
