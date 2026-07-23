#!/usr/bin/env python3
"""M11.1 Part 2: the controlled experiment M11's own combined retrain
made impossible to answer -- does the Ravendale-lore shared bank
(ravendale_lore.py) actually help, in isolation, now that every
character (including guard) shares ONE compositional scheme (Part 1)?

Trains the FULL genericized cast twice, same seed, same everything else,
differing in exactly one variable: whether shadewrath_corpus.py/
korrath_corpus.py/princess_corpus.py's lore-bank splice is enabled.
lore_bank_enabled gates the draw's RESULT, not the RNG draw itself (see
shadewrath_corpus._response()'s docstring), so every prompt and every
non-lore response is byte-identical between the two runs -- the only
possible source of a difference is the lore clause's presence.

Does NOT write to game/rawfs/model.bin or any other shipped artifact --
same precedent as m9_1_density_experiment.py/m9_2_kragan_catchphrase_
experiment.py. Once a winner is decided against the pre-registered bar
below, a separate make_m11_1_blob.py builds the real shipped ROM blob
from the winning configuration.

Run: uv run python m11_1_lore_bank_experiment.py   (from trainer/)
"""
import random
import re
import sys
import time
from pathlib import Path

import torch

from ngpt_trainer import cast_corpus as cc
from ngpt_trainer import guard_corpus as gc
from ngpt_trainer import korrath_corpus as kc
from ngpt_trainer import princess_corpus as pc
from ngpt_trainer import selena_corpus as sc
from ngpt_trainer import shadewrath_corpus as swc
from ngpt_trainer.divergence import cross_set_divergence
from ngpt_trainer.model import train_corpus_conditioned
from ngpt_trainer.npc_service import prompt_fields
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled, gru_step
from ngpt_trainer.vocab import Vocab

TRAINER = Path(__file__).resolve().parent

SEED = 0
HIDDEN = 320
PER_COMBO = 300
GUARD_PER_COMBO = 24
SHADEWRATH_PER_COMBO = 24
KORRATH_PER_COMBO = 12
PRINCESS_PER_COMBO = 4
HOLDOUT_COMBOS = 20
SAMPLE_SEED = 0xC0FFEE
INV_T_Q8 = 384
TOP_K = 5
MIN_AGREEMENT = 0.95
DIVERGENCE_SAMPLES = 5
DIVERGENCE_TEMPERATURE_INV_T_Q8 = 200

# M11's own shipped gossip-only baseline (docs/milestones/m11.md), for
# context in the printed comparison -- NOT the pre-registered bar itself
# (Part 2 point 1: the genericized-cast baseline this script trains is
# the new reference point, since the schema itself changed).
M11_GOSSIP_VAL_LOSS = 0.0992
M11_GOSSIP_AGREEMENT = 0.9771


def build_pairs(lore_bank_enabled: bool):
    selena_pairs = sc.generate_pairs(seed=SEED, per_combo=PER_COMBO)
    guard_pairs = gc.generate_pairs(seed=SEED, per_combo=GUARD_PER_COMBO)
    cast_pairs = cc.generate_pairs(seed=SEED)
    cc.assert_no_holdout_leak(cast_pairs)
    shadewrath_pairs = swc.generate_pairs(seed=SEED, per_combo=SHADEWRATH_PER_COMBO,
                                          lore_bank_enabled=lore_bank_enabled)
    korrath_pairs = kc.generate_pairs(seed=SEED, per_combo=KORRATH_PER_COMBO,
                                      lore_bank_enabled=lore_bank_enabled)
    princess_pairs = pc.generate_pairs(seed=SEED, per_combo=PRINCESS_PER_COMBO,
                                       lore_bank_enabled=lore_bank_enabled)
    return selena_pairs, guard_pairs, cast_pairs, shadewrath_pairs, korrath_pairs, princess_pairs


def combo_split(selena_pairs, seed: int = SEED, holdout: int = HOLDOUT_COMBOS):
    all_combos = sorted({sc.combo_key(p) for p, _ in selena_pairs})
    rng = random.Random(seed + 777)
    held = set(rng.sample(all_combos, holdout))
    train, val = [], []
    for p, r in selena_pairs:
        (val if sc.combo_key(p) in held else train).append((p, r))
    return train, val, held


def top1_agreement(model, q, vocab, probe: list[tuple[str, str]]) -> float:
    from ngpt_trainer.model import one_hot
    import numpy as np
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


def occupation_divergence(q, vocab) -> float:
    """M11.1 replacement for M7-M10's identity-axis check (N:selena vs.
    N:kip, ContextBuilder's old scheme -- both retired, Part 1). Same
    profile, same everything, ONLY occupation swapped -- the compositional
    scheme's own equivalent of an identity-conditioning ablation."""
    def draws(profile, base_seed):
        rel = {"familiarity": 0.975, "affection": 0.975, "trust": 0.975,
              "respect": 0.975, "fear": 0.0}
        prompt = prompt_fields(profile, rel, "cheerful", "item-found",
                               "witnessed", "found_gem")
        return [generate_sampled(q, vocab, prompt, seed=base_seed + i,
                                 inv_t_q8=DIVERGENCE_TEMPERATURE_INV_T_Q8, top_k=TOP_K)
               for i in range(DIVERGENCE_SAMPLES)]

    companion = dict(sc.SELENA_PROFILE)
    guard_occ = dict(sc.SELENA_PROFILE, occupation="guard")
    a = draws(companion, 0x1000)
    b = draws(guard_occ, 0x2000)
    return cross_set_divergence(a, b)


def conditioning_divergence_table(q, vocab) -> dict[str, float]:
    def draws(prompt, base_seed):
        return [generate_sampled(q, vocab, prompt, seed=base_seed + i,
                                 inv_t_q8=DIVERGENCE_TEMPERATURE_INV_T_Q8, top_k=TOP_K)
               for i in range(DIVERGENCE_SAMPLES)]

    trust, mood, context, event = 2, "cheerful", "item-found", "found_gem"

    mood_a = draws(sc.prompt_for(trust, "cheerful", context, event), 0x3000)
    mood_b = draws(sc.prompt_for(trust, "worried", context, event), 0x4000)
    mood_div = cross_set_divergence(mood_a, mood_b)

    trust_a = draws(sc.prompt_for(0, mood, context, event), 0x5000)
    trust_b = draws(sc.prompt_for(2, mood, context, event), 0x6000)
    trust_div = cross_set_divergence(trust_a, trust_b)

    context_a = draws(sc.prompt_for(trust, mood, "greeting", "none"), 0x7000)
    context_b = draws(sc.prompt_for(trust, mood, "farewell", "none"), 0x8000)
    context_div = cross_set_divergence(context_a, context_b)

    occ_div = occupation_divergence(q, vocab)

    return {"occupation": occ_div, "mood": mood_div,
           "trust": trust_div, "context": context_div}


# ---- invented-word proxy ------------------------------------------------
# Metric 6 of docs/ideas-capacity-scaling-evaluation-plan.md calls for
# literally counting invented-non-word occurrences. That's a human-
# eyeball step there; this is an automated proxy for the same idea: any
# whitespace/punctuation-separated token in a generated response that
# never appears ANYWHERE in the authored corpus text is either a novel
# (garbled) character sequence or, rarely, a legitimate word used only
# in the golden set's own draw -- cheap, objective, no new dependencies.
_WORD_RE = re.compile(r"[A-Z']+")


def build_corpus_vocab(*texts: str) -> set[str]:
    vocab = set()
    for text in texts:
        vocab.update(_WORD_RE.findall(text.upper()))
    return vocab


def invented_word_count(response: str, corpus_vocab: set[str]) -> int:
    words = _WORD_RE.findall(response.upper())
    return sum(1 for w in words if w not in corpus_vocab and len(w) > 1)


def golden_probe_prompts() -> list[str]:
    """Fixed set: the model's documented weak spot (Shadewrath/Korrath/
    Elowen) plus a Selena/guard/cast spread, same seed both runs."""
    rng = random.Random(SAMPLE_SEED ^ 0xB055)
    prompts = []
    prompts += [p for p, _ in rng.sample(swc.generate_pairs(seed=SEED, per_combo=1), 6)]
    prompts += [p for p, _ in rng.sample(kc.generate_pairs(seed=SEED, per_combo=1), 6)]
    prompts += [p for p, _ in rng.sample(pc.generate_pairs(seed=SEED, per_combo=1), 6)]
    prompts += [p for p, _ in rng.sample(sc.generate_pairs(seed=SEED, per_combo=1), 4)]
    cast_pairs = cc.generate_pairs(seed=SEED)
    prompts += [p for p, _ in rng.sample(cast_pairs, 4)]
    return prompts


def run_variant(name: str, lore_bank_enabled: bool, corpus_vocab_texts_extra: str = ""):
    print(f"\n{'=' * 70}\n{name} (lore_bank_enabled={lore_bank_enabled})\n{'=' * 70}")
    t0 = time.time()

    (selena_pairs, guard_pairs, cast_pairs, shadewrath_pairs,
     korrath_pairs, princess_pairs) = build_pairs(lore_bank_enabled)
    train_pairs, val_pairs, held_combos = combo_split(selena_pairs)
    all_train = (train_pairs + guard_pairs + cast_pairs
                + shadewrath_pairs + korrath_pairs + princess_pairs)

    full_text = (sc.corpus_text(seed=SEED, per_combo=PER_COMBO)
                + "".join(p + r for p, r in guard_pairs)
                + cc.corpus_text(seed=SEED)
                + swc.corpus_text(seed=SEED, per_combo=SHADEWRATH_PER_COMBO, lore_bank_enabled=lore_bank_enabled)
                + kc.corpus_text(seed=SEED, per_combo=KORRATH_PER_COMBO, lore_bank_enabled=lore_bank_enabled)
                + pc.corpus_text(seed=SEED, per_combo=PRINCESS_PER_COMBO, lore_bank_enabled=lore_bank_enabled))
    vocab = Vocab.from_text(full_text)
    corpus_vocab = build_corpus_vocab(full_text)
    print(f"corpus: {len(full_text)} chars ({len(full_text)/1e6:.2f} MB), "
         f"vocab={len(vocab)} symbols, {len(corpus_vocab)} distinct words")

    model = train_corpus_conditioned(all_train, val_pairs, vocab, hidden=HIDDEN,
                                     seed=SEED, max_epochs=120, patience=12,
                                     device="cpu")
    print(f"trained in {time.time()-t0:.0f}s, val loss {model.final_loss:.4f} "
         f"(M11 gossip-only was {M11_GOSSIP_VAL_LOSS:.4f})")

    q = quantize(model)
    probe = val_pairs[::max(1, len(val_pairs) // 150)][:150]
    agreement = top1_agreement(model, q, vocab, probe)
    print(f"int8-vs-float top-1 agreement (held-out combos): {agreement:.4f} "
         f"(M11 gossip-only was {M11_GOSSIP_AGREEMENT:.4f})")

    div_table = conditioning_divergence_table(q, vocab)
    print("conditioning-ablation divergence (trigram-Jaccard):")
    for axis, d in div_table.items():
        print(f"  {axis}: {d:.4f}")

    probe_prompts = golden_probe_prompts()
    total_invented = 0
    print(f"golden probe ({len(probe_prompts)} prompts, invented-word proxy):")
    for prompt in probe_prompts:
        got = generate_sampled(q, vocab, prompt, seed=SAMPLE_SEED,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K)
        n_invented = invented_word_count(got, corpus_vocab)
        total_invented += n_invented
        flag = f"[{n_invented} invented]" if n_invented else "[clean]"
        print(f"  {flag} {prompt}{got}")

    return {
        "val_loss": model.final_loss,
        "agreement": agreement,
        "divergence": div_table,
        "invented_word_total": total_invented,
        "probe_count": len(probe_prompts),
        "seconds": time.time() - t0,
    }


def main():
    baseline = run_variant("BASELINE (genericized cast, no lore bank)", lore_bank_enabled=False)
    treatment = run_variant("TREATMENT (genericized cast + lore bank)", lore_bank_enabled=True)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'metric':<30}{'baseline':<15}{'treatment':<15}")
    print(f"{'val loss':<30}{baseline['val_loss']:<15.4f}{treatment['val_loss']:<15.4f}")
    print(f"{'agreement':<30}{baseline['agreement']:<15.4f}{treatment['agreement']:<15.4f}")
    for axis in ("occupation", "mood", "trust", "context"):
        b, t = baseline["divergence"][axis], treatment["divergence"][axis]
        print(f"  div.{axis:<25}{b:<15.4f}{t:<15.4f}")
    print(f"{'invented words (of ' + str(baseline['probe_count']) + ')':<30}"
         f"{baseline['invented_word_total']:<15}{treatment['invented_word_total']:<15}")
    print(f"{'train time (s)':<30}{baseline['seconds']:<15.0f}{treatment['seconds']:<15.0f}")

    print("\nPRE-REGISTERED ACCEPTANCE BAR (docs/milestones/m11.1.md Part 2, "
         "methodology from docs/ideas-capacity-scaling-evaluation-plan.md):")
    print("the lore bank counts as a genuine win only if ALL of:")
    ok = True
    c1 = treatment["agreement"] >= MIN_AGREEMENT
    print(f"  [{'PASS' if c1 else 'FAIL'}] treatment agreement >= {MIN_AGREEMENT}: {treatment['agreement']:.4f}")
    ok &= c1
    for axis in ("occupation", "mood", "trust", "context"):
        b, t = baseline["divergence"][axis], treatment["divergence"][axis]
        lo, hi = b * 0.8, b * 1.2
        c = lo <= t <= hi if b > 0 else True
        print(f"  [{'PASS' if c else 'FAIL'}] treatment {axis} divergence within 20% of baseline "
             f"({b:.4f}): {t:.4f}")
        ok &= c
    bar = max(1, baseline["invented_word_total"] // 2)
    c2 = treatment["invented_word_total"] <= bar
    print(f"  [{'PASS' if c2 else 'FAIL'}] treatment invented-word count <= half of baseline "
         f"({baseline['invented_word_total']} -> bar {bar}): {treatment['invented_word_total']}")
    ok &= c2
    print(f"\nRESULT: {'LORE BANK WINS -- ship treatment' if ok else 'LORE BANK DOES NOT CLEAR THE BAR -- ship baseline, record honest negative'}")


if __name__ == "__main__":
    main()
