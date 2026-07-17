#!/usr/bin/env python3
"""M9.2: does a small per-character catchphrase bank for Kragan reduce
the specific coherence gap M9's own DoD flagged live on real hardware
("GOTTAND", "RECAND", "NONDS" -- docs/milestones/m9.md)? (docs/milestones/
m9.2.md)

Prompted by review of an external N64 homebrew project (Legend of Elya,
see docs/milestones/m9.2.md section 1): its persona-conditioned trainer
anchors each character's voice with a dedicated response bank repeated
heavily (~220x) on top of shared conditioning -- structurally the same
idea as this project's per-character catchphrase banks (Fergus's, M9),
just not yet extended to Kragan, the one character M9 shipped with a
known, unresolved garbling problem.

Trains the cast corpus ALONE (no Selena/guard), same methodology as
M9.1's density experiment, so each variant trains in minutes -- the
WITHOUT variant temporarily monkeypatches cast_corpus._CATCHPHRASES to
drop Kragan's bank (added this session) back to Fergus-only, isolating
the one variable this experiment tests.

Run: uv run python m9_2_kragan_catchphrase_experiment.py   (from trainer/)
"""
import random

from ngpt_trainer import cast_corpus as cc
from ngpt_trainer.model import train_corpus_conditioned
from ngpt_trainer.npc_service import personality_descriptor, prompt_fields, random_relationship_state
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled
from ngpt_trainer.vocab import Vocab

SEED = 0
HIDDEN = 320
SAMPLE_SEEDS = (0xC0FFEE, 0xBADA55)
INV_T_Q8 = 384
TOP_K = 5
HOLDOUT_FRACTION = 0.10


def combo_split(pairs, seed=SEED, fraction=HOLDOUT_FRACTION):
    all_combos = sorted({cc.combo_key(p)[3:] for p, _ in pairs})
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

    kragan_profile = cc.CHARACTERS["kragan"]
    kragan_descriptor = personality_descriptor(kragan_profile["traits"])

    # Trained-combo Kragan samples, spread across mood/context, 2 seeds
    # each (stochastic sampling per this project's own convention -- see
    # docs/plan.md's "Model I/O testing" section -- greedy collapses to
    # one output per prompt and would hide exactly the variability this
    # experiment needs to see).
    rng = random.Random(0xDEADBEEF)
    kragan_pairs = [(p, r) for p, r in pairs if "OCC:bandit" in p]
    sample_prompts = [p for p, _ in rng.sample(kragan_pairs, min(6, len(kragan_pairs)))]
    print("  Kragan trained-combo generations:")
    for prompt in sample_prompts:
        for s in SAMPLE_SEEDS:
            got = generate_sampled(q, vocab, prompt, seed=s,
                                   inv_t_q8=INV_T_Q8, top_k=TOP_K)
            print(f"    [seed {s:#x}] {prompt}{got!r}")

    # Held-out generalization probe specific to Kragan's axis: bandit x
    # gruff is one of cast_corpus.HOLDOUT_COMBOS (never seen in training,
    # any structure), independent of whether the catchphrase bank exists.
    print("  Kragan held-out (bandit, gruff) generalization probe:")
    rel = random_relationship_state(0xF00D)
    prompt = prompt_fields(kragan_profile, rel, "cheerful", "greeting")
    prompt = prompt.replace(f"D:{kragan_descriptor} ", "D:gruff ")
    for s in SAMPLE_SEEDS:
        got = generate_sampled(q, vocab, prompt, seed=s,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K)
        print(f"    [seed {s:#x}] {prompt}{got!r}")

    return model.final_loss


def main():
    with_bank = cc.generate_pairs(seed=SEED)  # shipped M9.2: Fergus + Kragan banks

    original_catchphrases = cc._CATCHPHRASES
    cc._CATCHPHRASES = {"fergus": cc._FERGUS_CATCHPHRASES}  # M9 baseline: no Kragan bank
    try:
        without_bank = cc.generate_pairs(seed=SEED)
    finally:
        cc._CATCHPHRASES = original_catchphrases

    without_loss = run_variant("WITHOUT Kragan catchphrase bank (M9 baseline)", without_bank)
    with_loss = run_variant("WITH Kragan catchphrase bank (M9.2)", with_bank)

    print(f"\n=== SUMMARY ===")
    print(f"WITHOUT Kragan bank final val loss: {without_loss:.4f}")
    print(f"WITH    Kragan bank final val loss: {with_loss:.4f}")


if __name__ == "__main__":
    main()
