#!/usr/bin/env python3
"""M9.1: does corpus DENSITY or REPETITION STRUCTURE drive the coherence
gap left open by M9? (docs/milestones/m9.1.md)

M9's shipped cast_corpus.py hits ~125-146K chars/character via a WIDE
structure: all 240 (tier x mood x context) combos, each repeated only
per_combo=3 times. Guard's own proven-working M8 corpus hits a similar
per-instance total via the opposite NARROW structure: ~45 combos,
each repeated 24 times. Same total volume, opposite repetition pattern
-- this experiment isolates which one actually matters, training on the
cast corpus ALONE (no Selena/guard) so each variant trains in minutes,
not the ~30min full production run.

Run: uv run python m9_1_density_experiment.py   (from trainer/)
"""
import random

from ngpt_trainer import cast_corpus as cc
from ngpt_trainer.model import train_corpus_conditioned
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled
from ngpt_trainer.vocab import Vocab

SEED = 0
HIDDEN = 320
SAMPLE_SEED = 0xC0FFEE
INV_T_Q8 = 384
TOP_K = 5
# Fraction, not a fixed count -- WIDE has 240 distinct combos, NARROW has
# ~30 (10/character x 3 characters); a fixed holdout count sized for WIDE
# would eat most of NARROW's combos. ~10% held out either way.
HOLDOUT_FRACTION = 0.10


def combo_split(pairs, seed=SEED, fraction=HOLDOUT_FRACTION):
    all_combos = sorted({cc.combo_key(p)[3:] for p, _ in pairs})  # (tier, mood, ctx)
    rng = random.Random(seed + 777)
    holdout = max(1, round(len(all_combos) * fraction))
    held = set(rng.sample(all_combos, min(holdout, len(all_combos))))
    train, val = [], []
    for p, r in pairs:
        (val if cc.combo_key(p)[3:] in held else train).append((p, r))
    return train, val


def run_variant(name: str, pairs: list[tuple[str, str]]):
    total_chars = sum(len(p) + len(r) for p, r in pairs)
    print(f"\n=== {name}: {len(pairs)} pairs, {total_chars} chars "
         f"({total_chars/1e6:.3f} MB) ===")

    train_pairs, val_pairs = combo_split(pairs)
    print(f"  train={len(train_pairs)} val={len(val_pairs)}")

    full_text = "".join(p + r for p, r in pairs)
    vocab = Vocab.from_text(full_text)
    print(f"  vocab={len(vocab)} symbols")

    model = train_corpus_conditioned(train_pairs, val_pairs, vocab, hidden=HIDDEN,
                                     seed=SEED, max_epochs=120, patience=15,
                                     device="cpu")
    print(f"  FINAL val loss: {model.final_loss:.4f}")
    q = quantize(model)

    # Sample a spread of TRAINED prompts.
    rng = random.Random(SAMPLE_SEED)
    sample_prompts = [p for p, _ in rng.sample(pairs, min(8, len(pairs)))]
    print("  sample trained-combo generations:")
    for prompt in sample_prompts:
        got = generate_sampled(q, vocab, prompt, seed=SAMPLE_SEED,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K)
        print(f"    {prompt}{got}")

    # Held-out (occupation, descriptor) generalization probe -- same 3
    # combos cast_corpus.py permanently excludes from training.
    from ngpt_trainer.npc_service import personality_descriptor, prompt_fields, random_relationship_state
    profile_by_occ = {p["occupation"]: p for p in cc.CHARACTERS.values()}
    print("  held-out (occupation, descriptor) generalization probe:")
    for occupation, descriptor in cc.holdout_pairs():
        profile = profile_by_occ[occupation]
        real_descriptor = personality_descriptor(profile["traits"])
        checksum = sum(ord(c) for c in occupation + descriptor)
        rel = random_relationship_state(SAMPLE_SEED + checksum)
        prompt = prompt_fields(profile, rel, "cheerful", "greeting")
        prompt = prompt.replace(f"D:{real_descriptor} ", f"D:{descriptor} ")
        got = generate_sampled(q, vocab, prompt, seed=SAMPLE_SEED,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K)
        print(f"    {prompt}{got!r}")

    return model.final_loss


def main():
    wide = cc.generate_pairs(seed=SEED)  # shipped M9 structure: 240 combos, per_combo=3
    narrow = cc.generate_pairs(seed=SEED, combo_count=10, per_combo=70)  # guard-like

    wide_loss = run_variant("WIDE (shipped M9: 240 combos x per_combo=3)", wide)
    narrow_loss = run_variant("NARROW (guard-like: 10 combos x per_combo=70)", narrow)

    print(f"\n=== SUMMARY ===")
    print(f"WIDE   final val loss: {wide_loss:.4f}")
    print(f"NARROW final val loss: {narrow_loss:.4f}")


if __name__ == "__main__":
    main()
