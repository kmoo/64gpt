#!/usr/bin/env python3
"""Build the M11.1 blob, golden vectors, and ROM self-test header.

M11.1 (docs/milestones/m11.1.md): Part 1 genericized every remaining
old-scheme character (selena/shadewrath/korrath/elowen/guard) onto
NpcService's compositional scheme -- ContextBuilder is gone. Part 3
added AUD:/BOND:/SPECIES:. Part 2's controlled experiment
(m11_1_lore_bank_experiment.py) tested the Ravendale-lore bank in
isolation and found it does NOT clear the pre-registered bar (invented-
word count 69->67 against a required <=34) -- an honest negative,
cleanly isolated this time (M11's own combined retrain couldn't
attribute its regression to any one change). This script ships the
WINNING configuration: the genericized cast, lore bank OFF.

Training is cached in trainer/.m11_1_model.pt (git-ignored); delete it
to retrain.
Run: uv run python make_m11_1_blob.py   (from trainer/)
"""
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from ngpt_trainer import cast_corpus as cc
from ngpt_trainer import guard_corpus as gc
from ngpt_trainer import korrath_corpus as kc
from ngpt_trainer import princess_corpus as pc
from ngpt_trainer import selena_corpus as sc
from ngpt_trainer import shadewrath_corpus as swc
from ngpt_trainer.divergence import cross_set_divergence
from ngpt_trainer.export import build_blob, trace_bytes
from ngpt_trainer.model import CharGRU, train_corpus_conditioned
from ngpt_trainer.npc_service import personality_descriptor, prompt_fields
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled, gru_step, trace_sampled
from ngpt_trainer.sampler_lut import LUT_EXP2
from ngpt_trainer.vocab import Vocab

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / ".m11_1_model.pt"

SEED = 0
HIDDEN = 320
LORE_BANK_ENABLED = False    # M11.1 Part 2 result: does not clear the bar
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
MAX_GOLDEN_LEN = 300
DIVERGENCE_SAMPLES = 5
DIVERGENCE_TEMPERATURE_INV_T_Q8 = 200
GENERALIZATION_SAMPLE_SIZE = 8

# Prior milestones' recorded numbers, for the catastrophic-interference
# comparison. M9-M11 numbers don't transfer cleanly across the M11.1
# schema change (Part 1) -- kept for context, not as the acceptance gate.
M11_GOSSIP_VAL_LOSS = 0.0992
M11_GOSSIP_AGREEMENT = 0.9771
# M11.1's own isolated-experiment baseline (m11_1_lore_bank_experiment.py,
# lore_bank_enabled=False, identical corpus/seed to this script) -- the
# real reference point this schema's numbers should be judged against.
M11_1_EXPERIMENT_VAL_LOSS = 0.1015
M11_1_EXPERIMENT_AGREEMENT = 0.9508


def build_all_pairs():
    selena_pairs = sc.generate_pairs(seed=SEED, per_combo=PER_COMBO)
    guard_pairs = gc.generate_pairs(seed=SEED, per_combo=GUARD_PER_COMBO)
    cast_pairs = cc.generate_pairs(seed=SEED)
    cc.assert_no_holdout_leak(cast_pairs)
    shadewrath_pairs = swc.generate_pairs(seed=SEED, per_combo=SHADEWRATH_PER_COMBO,
                                          lore_bank_enabled=LORE_BANK_ENABLED)
    korrath_pairs = kc.generate_pairs(seed=SEED, per_combo=KORRATH_PER_COMBO,
                                      lore_bank_enabled=LORE_BANK_ENABLED)
    princess_pairs = pc.generate_pairs(seed=SEED, per_combo=PRINCESS_PER_COMBO,
                                       lore_bank_enabled=LORE_BANK_ENABLED)
    return (selena_pairs, guard_pairs, cast_pairs,
            shadewrath_pairs, korrath_pairs, princess_pairs)


def combo_split(selena_pairs, seed: int = SEED, holdout: int = HOLDOUT_COMBOS):
    all_combos = sorted({sc.combo_key(p) for p, _ in selena_pairs})
    rng = random.Random(seed + 777)
    held = set(rng.sample(all_combos, holdout))
    train, val = [], []
    for p, r in selena_pairs:
        (val if sc.combo_key(p) in held else train).append((p, r))
    return train, val, held


def get_model(train_pairs, val_pairs, vocab) -> CharGRU:
    if CACHE.exists():
        d = torch.load(CACHE, weights_only=True)
        model = CharGRU(vocab_size=len(vocab), hidden=d["hidden"])
        model.load_state_dict(d["state"])
        model.eval()
        model.final_loss = d["val_loss"]
        print(f"loaded cached model (val loss {model.final_loss:.4f})")
        return model
    # device="cpu" forced -- MPS training competes with opencoder's MLX
    # server for unified GPU memory and OOM-crashes both if they run at
    # once (same reasoning every prior make_mN_blob.py documents).
    model = train_corpus_conditioned(train_pairs, val_pairs, vocab, hidden=HIDDEN,
                                     seed=SEED, max_epochs=120, patience=12,
                                     device="cpu")
    torch.save({"state": model.state_dict(), "val_loss": model.final_loss,
                "hidden": HIDDEN}, CACHE)
    return model


def top1_agreement(model, q, vocab, probe: list[tuple[str, str]]) -> float:
    match = total = 0
    with torch.no_grad():
        for prompt, response in probe:
            pids = vocab.encode(prompt)
            ids = pids + vocab.encode(response)
            from ngpt_trainer.model import one_hot
            flogits, _ = model(one_hot([vocab.eos_id] + ids, len(vocab)))
            fam = torch.argmax(flogits[0], dim=-1).tolist()
            h = np.zeros(q.H, dtype=np.int64)
            for pos, x in enumerate([vocab.eos_id] + ids):
                h, logits = gru_step(q, h, x)
                if pos >= len(pids):
                    match += int(np.argmax(logits)) == fam[pos]
                    total += 1
    return match / total if total else 1.0


def curated_golden_combos():
    """M7's Selena grid + one representative guard combo + a Shadewrath,
    Korrath, and Elowen spread across all 3 trust tiers (their own
    bespoke arcs) -- every character now round-trips through the SAME
    compositional bridge, unlike M7-M11 which proved two schemas
    separately (M11.1 Part 1)."""
    tm_cycle = [(0, "cheerful"), (2, "embarrassed"), (1, "sassy")]
    combos = [("selena", *tm_cycle[i % len(tm_cycle)], context)
              for i, context in enumerate(sc.CONTEXTS)]
    combos += [(gc.GUARD_IDS[0], 1, "cheerful", "greeting")]
    combos += [("shadewrath", tier, mood, "greeting")
               for tier, mood in ((0, "sassy"), (1, "worried"), (2, "tender"))]
    combos += [("korrath", tier, mood, "greeting")
               for tier, mood in ((0, "sassy"), (1, "worried"), (2, "tender"))]
    combos += [("elowen", tier, mood, "greeting")
               for tier, mood in ((0, "sassy"), (1, "worried"), (2, "tender"))]
    return combos


def korrath_golden_prompts(n: int = 3, seed: int = SAMPLE_SEED) -> list[str]:
    rng = random.Random(seed ^ 0xB055)
    pairs = kc.generate_pairs(seed=SEED, per_combo=1, lore_bank_enabled=LORE_BANK_ENABLED)
    sample = rng.sample(pairs, min(n, len(pairs)))
    return [prompt for prompt, _ in sample]


def princess_golden_prompts(n: int = 3, seed: int = SAMPLE_SEED) -> list[str]:
    rng = random.Random(seed ^ 0xE10E)
    pairs = pc.generate_pairs(seed=SEED, per_combo=1, lore_bank_enabled=LORE_BANK_ENABLED)
    sample = rng.sample(pairs, min(n, len(pairs)))
    return [prompt for prompt, _ in sample]


def shadewrath_golden_prompts(n: int = 3, seed: int = SAMPLE_SEED) -> list[str]:
    rng = random.Random(seed ^ 0xBAD)
    pairs = swc.generate_pairs(seed=SEED, per_combo=1, lore_bank_enabled=LORE_BANK_ENABLED)
    sample = rng.sample(pairs, min(n, len(pairs)))
    return [prompt for prompt, _ in sample]


def cast_golden_prompts(cast_pairs: list[tuple[str, str]], n: int = 6,
                        seed: int = SAMPLE_SEED) -> list[str]:
    rng = random.Random(seed)
    sample = rng.sample(cast_pairs, min(n, len(cast_pairs)))
    return [prompt for prompt, _ in sample]


def gossip_golden_prompts(cast_pairs: list[tuple[str, str]],
                          seed: int = SAMPLE_SEED) -> list[str]:
    rng = random.Random(seed ^ 0x9055)
    prompts = []
    for tag in cc.GOSSIP_EVENTS:
        tag_pairs = [(p, r) for p, r in cast_pairs
                    if p.split("EV:")[1].split("|")[0] == tag]
        prompts.append(rng.choice(tag_pairs)[0])
    return prompts


def conditioning_divergence_table(q, vocab) -> dict[str, float]:
    """M11.1: the identity axis (M7-M10's N:selena vs N:kip check) is
    replaced by an occupation-swap check -- ContextBuilder/N:<id> is
    gone (Part 1), and OCC: is the compositional scheme's own nearest
    equivalent to what "identity" meant in the old scheme."""
    def draws(prompt, base_seed):
        return [generate_sampled(q, vocab, prompt, seed=base_seed + i,
                                 inv_t_q8=DIVERGENCE_TEMPERATURE_INV_T_Q8, top_k=TOP_K)
               for i in range(DIVERGENCE_SAMPLES)]

    trust, mood, context, event = 2, "cheerful", "item-found", "found_gem"

    rel = {"familiarity": 0.975, "affection": 0.975, "trust": 0.975,
          "respect": 0.975, "fear": 0.0}
    occ_a = draws(prompt_fields(sc.SELENA_PROFILE, rel, mood, "item-found",
                               "witnessed", event), 0x1000)
    occ_b = draws(prompt_fields(dict(sc.SELENA_PROFILE, occupation="guard"), rel,
                               mood, "item-found", "witnessed", event), 0x2000)
    occ_div = cross_set_divergence(occ_a, occ_b)

    mood_a = draws(sc.prompt_for(trust, "cheerful", context, event), 0x3000)
    mood_b = draws(sc.prompt_for(trust, "worried", context, event), 0x4000)
    mood_div = cross_set_divergence(mood_a, mood_b)

    trust_a = draws(sc.prompt_for(0, mood, context, event), 0x5000)
    trust_b = draws(sc.prompt_for(2, mood, context, event), 0x6000)
    trust_div = cross_set_divergence(trust_a, trust_b)

    context_a = draws(sc.prompt_for(trust, mood, "greeting", "none"), 0x7000)
    context_b = draws(sc.prompt_for(trust, mood, "farewell", "none"), 0x8000)
    context_div = cross_set_divergence(context_a, context_b)

    return {"occupation": occ_div, "mood": mood_div,
           "trust": trust_div, "context": context_div}


def generalization_check(q, vocab, seed: int = SAMPLE_SEED) -> list[dict]:
    """Same falsification test as M9-M11 -- probes held-out (occupation,
    descriptor) pairs, orthogonal to M11.1's changes."""
    held = cc.holdout_pairs()
    rng = random.Random(seed ^ 0xF00D)
    sample = rng.sample(held, min(GENERALIZATION_SAMPLE_SIZE, len(held)))

    profile_by_occ = {p["occupation"]: p for p in cc.CHARACTERS.values()}

    results = []
    for occupation, descriptor in sample:
        profile = profile_by_occ[occupation]
        real_descriptor = personality_descriptor(profile["traits"])
        combo_checksum = sum(ord(c) for c in occupation + descriptor)
        from ngpt_trainer.npc_service import random_relationship_state
        rel = random_relationship_state(seed + combo_checksum)
        prompt = prompt_fields(profile, rel, "cheerful", "greeting", "witnessed")
        prompt = prompt.replace(f"D:{real_descriptor} ", f"D:{descriptor} ")
        got = generate_sampled(q, vocab, prompt, seed=seed, inv_t_q8=INV_T_Q8, top_k=TOP_K)
        degenerate = not (1 <= len(got) <= MAX_GOLDEN_LEN)
        results.append({"occupation": occupation, "descriptor": descriptor,
                        "prompt": prompt, "output": got, "degenerate": degenerate})
    return results


def goldens_bytes(pairs: list[tuple[str, str]]) -> bytes:
    out = struct.pack(">IHHH", SAMPLE_SEED, INV_T_Q8, TOP_K, len(pairs))
    for prompt, response in pairs:
        for s in (prompt, response):
            b = s.encode("ascii")
            out += struct.pack(">H", len(b)) + b
    return out


def emit_sampler_lut_header() -> str:
    rows = []
    for i in range(0, 256, 12):
        rows.append("  " + ", ".join(str(v) for v in LUT_EXP2[i:i + 12]) + ",")
    return (
        "/* AUTOGENERATED by trainer/make_m11_1_blob.py - do not edit.\n"
        " * exp2 LUT over [-16, 0) in Q10 (64-unit buckets), values Q15.\n"
        " * Twin of ngpt_trainer/sampler_lut.py - one source, no drift. */\n"
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        "static const uint16_t NGPT_LUT_EXP2[256] = {\n"
        + "\n".join(rows) + "\n};\n"
    )


def emit_selftest_header(pairs: list[tuple[str, str]]) -> str:
    def carr(name: str, items) -> str:
        vals = ", ".join(f'"{s}"' for s in items)
        return f"static const char *{name}[] = {{{vals}}};\n"

    return (
        "/* AUTOGENERATED by trainer/make_m11_1_blob.py - do not edit.\n"
        " * Prompts + SEEDED SAMPLED golden responses replayed by the ROM\n"
        " * boot self-test. Must stay in sync with tests/vectors/m11_1_goldens.bin. */\n"
        "#pragma once\n"
        "#include <stdint.h>\n\n"
        f"static const uint32_t SELFTEST_SAMPLE_SEED = {SAMPLE_SEED}u;\n"
        f"static const uint16_t SELFTEST_INV_T_Q8 = {INV_T_Q8};\n"
        f"static const uint16_t SELFTEST_TOP_K = {TOP_K};\n\n"
        f"static const uint32_t SELFTEST_COUNT = {len(pairs)};\n"
        + carr("SELFTEST_PROMPTS", [p for p, _ in pairs])
        + carr("SELFTEST_GOLDEN", [r for _, r in pairs])
    )


def main() -> None:
    (selena_pairs, guard_pairs, cast_pairs, shadewrath_pairs,
     korrath_pairs, princess_pairs) = build_all_pairs()
    train_pairs, val_pairs, held_combos = combo_split(selena_pairs)
    all_train = (train_pairs + guard_pairs + cast_pairs
                + shadewrath_pairs + korrath_pairs + princess_pairs)
    full_text = (sc.corpus_text(seed=SEED, per_combo=PER_COMBO)
                + "".join(p + r for p, r in guard_pairs)
                + cc.corpus_text(seed=SEED)
                + swc.corpus_text(seed=SEED, per_combo=SHADEWRATH_PER_COMBO, lore_bank_enabled=LORE_BANK_ENABLED)
                + kc.corpus_text(seed=SEED, per_combo=KORRATH_PER_COMBO, lore_bank_enabled=LORE_BANK_ENABLED)
                + pc.corpus_text(seed=SEED, per_combo=PRINCESS_PER_COMBO, lore_bank_enabled=LORE_BANK_ENABLED))
    vocab = Vocab.from_text(full_text)
    gossip_count = sum(1 for p, _ in cast_pairs
                       if p.split("EV:")[1].split("|")[0] in cc.GOSSIP_EVENTS)
    print(f"corpus: {len(selena_pairs)} selena + {len(guard_pairs)} guard + "
         f"{len(cast_pairs)} compositional-cast (9 chars, {gossip_count} gossip-tagged) + "
         f"{len(shadewrath_pairs)} shadewrath + {len(korrath_pairs)} korrath + "
         f"{len(princess_pairs)} elowen pairs ({len(full_text)} chars, "
         f"{len(full_text)/1e6:.2f} MB) -- lore_bank_enabled={LORE_BANK_ENABLED}")
    print(f"combo split: {len(train_pairs)} train-combo lines, {len(val_pairs)} "
         f"held-out-combo lines, {len(held_combos)} combos held out of Selena's 120")
    print(f"vocab: {len(vocab)} symbols (incl EOS)")

    model = get_model(all_train, val_pairs, vocab)
    q = quantize(model)
    print(f"trained: H={HIDDEN}, val loss {model.final_loss:.4f} "
         f"(M11 gossip-only was {M11_GOSSIP_VAL_LOSS:.4f}, M11.1's own isolated "
         f"experiment baseline was {M11_1_EXPERIMENT_VAL_LOSS:.4f})")

    probe = val_pairs[::max(1, len(val_pairs) // 150)][:150]
    agree = top1_agreement(model, q, vocab, probe)
    print(f"int-vs-float top-1 agreement (held-out combos, completion-only): {agree:.4f}")
    if agree < MIN_AGREEMENT:
        print(f"FATAL: agreement {agree:.4f} < {MIN_AGREEMENT}")
        sys.exit(1)

    golden_combos = curated_golden_combos()
    golden_pairs = []
    for combo in golden_combos:
        npc_id = combo[0]
        _, trust, mood, context = combo
        if npc_id == "selena":
            event = sc.EVENTS_FOR_CONTEXT[context][0]
            prompt = sc.prompt_for(trust, mood, context, event)
        elif npc_id == "shadewrath":
            event = swc.EVENTS_FOR_CONTEXT[context][0]
            prompt = swc.prompt_for(trust, mood, context, event)
        elif npc_id == "korrath":
            event = kc.EVENTS_FOR_CONTEXT[context][0]
            prompt = kc.prompt_for(trust, mood, context, event)
        elif npc_id == "elowen":
            event = pc.EVENTS_FOR_CONTEXT[context][0]
            prompt = pc.prompt_for(trust, mood, context, event)
        else:
            prompt = gc.prompt_for(npc_id, trust, mood, context)
        got = generate_sampled(q, vocab, prompt, seed=SAMPLE_SEED,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K)
        print(f"  {prompt}{got}")
        if not (1 <= len(got) <= MAX_GOLDEN_LEN):
            print(f"FATAL: degenerate golden for {prompt!r}: {got!r}")
            sys.exit(1)
        golden_pairs.append((prompt, got))

    for prompts_fn, args in (
        (cast_golden_prompts, (cast_pairs,)),
        (gossip_golden_prompts, (cast_pairs,)),
        (shadewrath_golden_prompts, ()),
        (korrath_golden_prompts, ()),
        (princess_golden_prompts, ()),
    ):
        for prompt in prompts_fn(*args):
            got = generate_sampled(q, vocab, prompt, seed=SAMPLE_SEED,
                                   inv_t_q8=INV_T_Q8, top_k=TOP_K)
            print(f"  {prompt}{got}")
            if not (1 <= len(got) <= MAX_GOLDEN_LEN):
                print(f"FATAL: degenerate golden for {prompt!r}: {got!r}")
                sys.exit(1)
            golden_pairs.append((prompt, got))

    div_table = conditioning_divergence_table(q, vocab)
    print("conditioning-ablation divergence (trigram-Jaccard):")
    for axis, d in div_table.items():
        print(f"  {axis}: {d:.4f}")
    if div_table["occupation"] < div_table["mood"] * 0.8:
        print(f"WARNING: occupation divergence ({div_table['occupation']:.4f}) is "
             f"well below mood divergence ({div_table['mood']:.4f}) -- possible "
             f"catastrophic interference.")

    gen_results = generalization_check(q, vocab)
    print(f"\ngeneralization check ({len(gen_results)} held-out "
         f"(occupation, descriptor) combos, never seen anywhere in training):")
    degenerate_count = sum(1 for r in gen_results if r["degenerate"])
    for r in gen_results:
        flag = "DEGENERATE" if r["degenerate"] else "ok"
        print(f"  [{flag}] {r['occupation']}/{r['descriptor']}: "
             f"{r['prompt']}\n    -> {r['output']!r}")
    print(f"generalization: {len(gen_results) - degenerate_count}/{len(gen_results)} "
         f"held-out combos produced non-degenerate output")
    if degenerate_count > len(gen_results) // 2:
        print(f"WARNING: more than half of held-out combos degenerated.")

    blob = build_blob(q, vocab)
    targets = {
        REPO / "game" / "rawfs" / "model.bin": blob,
        REPO / "tests" / "vectors" / "m11_1_gru.bin": blob,
        REPO / "tests" / "vectors" / "m11_1_goldens.bin":
            goldens_bytes(golden_pairs),
        REPO / "tests" / "vectors" / "m11_1_trace.bin":
            trace_bytes(trace_sampled(q, vocab, golden_pairs[0][0],
                                      SAMPLE_SEED, INV_T_Q8, TOP_K), q.H),
        REPO / "game" / "src" / "user" / "selftestGolden.h":
            emit_selftest_header(golden_pairs).encode("ascii"),
        REPO / "core" / "ngpt_sampler_lut.h":
            emit_sampler_lut_header().encode("ascii"),
    }
    for path, data in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"wrote {path.relative_to(REPO)} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
