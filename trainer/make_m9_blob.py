#!/usr/bin/env python3
"""Build the M9 blob, golden vectors, and ROM self-test header: Selena +
guard (M7/M8, unchanged corpora, for non-regression) plus M9's
compositional-conditioning corpus, trained fresh at H=320 -- the first
model this project ships with the new P:/AGE:/D:/OCC:/R:/M:/C:/EV:
prompt schema, replacing M7/M8's opaque N:<id> tag entirely (M8's own
N: prompts stay in the training MIX for non-regression coverage of the
old schema's characters, but the NEW schema's compositional strings are
what M9 is actually testing). See docs/milestones/m9.md.

**Catastrophic-interference check**: same discipline as make_m8_blob.py
-- Selena's val loss/divergence recomputed here and printed next to
M8's recorded numbers so the corpus mix change can't silently degrade
her voice without someone noticing.

**New gate -- generalization check (the actual M9 hypothesis test)**:
docs/milestones/m9.md's Data Science Review specifically flags that
"the capacity-dilution hypothesis needs its own falsification test, not
just a plausibility argument." m9_corpus.py's combo-level holdout
(~18% of (occupation, descriptor) pairs, never in training) is exactly
that test: this script generates output for a sample of held-out
combos and checks it isn't degenerate -- non-empty, not wildly long,
not a verbatim copy of a trained combo's lines. This does NOT replace
the LLM-judge pass (llm_judge.py, run separately) for actual voice
quality on these unseen combos; it's the fast, automated half of the
same question, same relationship divergence gates have to the LLM-
judge's own text-quality read (m9.md section 5).

Training is cached in trainer/.m9_model.pt (git-ignored); delete it to
retrain. Run: uv run python make_m9_blob.py   (from trainer/)
"""
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from ngpt_trainer import guard_corpus as gc
from ngpt_trainer import m9_corpus as m9c
from ngpt_trainer import selena_corpus as sc
from ngpt_trainer.divergence import cross_set_divergence
from ngpt_trainer.export import build_blob, trace_bytes
from ngpt_trainer.model import CharGRU, train_corpus_conditioned
from ngpt_trainer.npc_service import age_gender_token, personality_descriptor
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled, gru_step, trace_sampled
from ngpt_trainer.sampler_lut import LUT_EXP2
from ngpt_trainer.vocab import Vocab

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / ".m9_model.pt"

SEED = 0
HIDDEN = 320
PER_COMBO = 300             # Selena's corpus density, unchanged from M7/M8
GUARD_PER_COMBO = 24        # unchanged from M8
HOLDOUT_COMBOS = 20         # Selena's own combo-level holdout, unchanged
SAMPLE_SEED = 0xC0FFEE
INV_T_Q8 = 384
TOP_K = 5
MIN_AGREEMENT = 0.95
MAX_GOLDEN_LEN = 300
DIVERGENCE_SAMPLES = 5
DIVERGENCE_TEMPERATURE_INV_T_Q8 = 200
GENERALIZATION_SAMPLE_SIZE = 8   # of the held-out combos, how many to probe

# M8's recorded numbers (docs/milestones/m8.md), for the catastrophic-
# interference comparison.
M8_VAL_LOSS = 0.1026
M8_DIVERGENCE = {"identity": None, "mood": None, "trust": None, "context": None}


def build_all_pairs():
    selena_pairs = sc.generate_pairs(seed=SEED, per_combo=PER_COMBO)
    thin_pairs = sc.generate_thin_identity_pairs(seed=1000)
    guard_pairs = gc.generate_pairs(seed=SEED, per_combo=GUARD_PER_COMBO)
    m9c.assert_no_holdout_leak()
    m9_pairs = m9c.generate_pairs()
    return selena_pairs, thin_pairs, guard_pairs, m9_pairs


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
    # once (same reasoning make_m7_blob.py/make_m8_blob.py document).
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
    """M7's Selena grid + one representative guard combo (M8's own
    goldens, unchanged) + a spread of M9 compositional combos (all
    TRAINED, not held out) proving the new schema round-trips through
    the real quantized engine."""
    tm_cycle = [(0, "cheerful"), (2, "embarrassed"), (1, "sassy")]
    combos = [("selena", *tm_cycle[i % len(tm_cycle)], context)
              for i, context in enumerate(sc.CONTEXTS)]
    combos += [(gid, 1, "cheerful", "greeting") for gid in gc.GUARD_IDS]
    return combos


def m9_golden_prompts(raw: list[dict], n: int = 6, seed: int = SAMPLE_SEED) -> list[str]:
    """A spread of TRAINED (not held-out) M9 prompts for the self-test."""
    from ngpt_trainer.npc_service import prompt_fields
    rng = random.Random(seed)
    sample = rng.sample(raw, min(n, len(raw)))
    return [prompt_fields(e["persona"], m9c._relationship_state(e["tier"]),
                          e["mood"], e["context"]) for e in sample]


def conditioning_divergence_table(q, vocab, held_combos) -> dict[str, float]:
    def draws(prompt, base_seed):
        return [generate_sampled(q, vocab, prompt, seed=base_seed + i,
                                 inv_t_q8=DIVERGENCE_TEMPERATURE_INV_T_Q8, top_k=TOP_K)
               for i in range(DIVERGENCE_SAMPLES)]

    trust, mood, context = 2, "cheerful", "item-found"
    event = "found_gem"

    def p(trust_, mood_, context_, event_, npc_id="selena"):
        return sc.prompt_for(trust_, mood_, context_, event_, npc_id=npc_id)

    identity_a = draws(p(trust, mood, context, event, "selena"), 0x1000)
    identity_b = draws(p(trust, mood, context, event, "kip"), 0x2000)
    identity_div = cross_set_divergence(identity_a, identity_b)

    mood_a = draws(p(trust, "cheerful", context, event), 0x3000)
    mood_b = draws(p(trust, "worried", context, event), 0x4000)
    mood_div = cross_set_divergence(mood_a, mood_b)

    trust_a = draws(p(0, mood, context, event), 0x5000)
    trust_b = draws(p(2, mood, context, event), 0x6000)
    trust_div = cross_set_divergence(trust_a, trust_b)

    context_a = draws(p(trust, mood, "greeting", "none"), 0x7000)
    context_b = draws(p(trust, mood, "farewell", "none"), 0x8000)
    context_div = cross_set_divergence(context_a, context_b)

    return {"identity": identity_div, "mood": mood_div,
           "trust": trust_div, "context": context_div}


def generalization_check(q, vocab, seed: int = SAMPLE_SEED) -> list[dict]:
    """The real M9 hypothesis test: generate for held-out (occupation,
    descriptor) combos the model NEVER trained on, and report whether
    the output is at least non-degenerate. Full voice-quality judgment
    is llm_judge.py's job (run separately against this same combo list)
    -- this is the fast automated half."""
    from ngpt_trainer.npc_service import prompt_fields, random_relationship_state
    held = m9c.holdout_pairs()
    rng = random.Random(seed ^ 0xF00D)
    sample = rng.sample(held, min(GENERALIZATION_SAMPLE_SIZE, len(held)))

    results = []
    for occupation, descriptor in sample:
        # Reconstruct a plausible profile matching this held-out combo --
        # any age/gender/mood/context works, only occupation+descriptor
        # need to match the withheld pair exactly.
        traits = {"warmth": 50, "humor": 50, "impulsivity": 50, "bravery": 50, "focus": 50}
        # Nudge traits toward the target descriptor by trying a few seeds.
        profile = {"occupation": occupation, "age": 30, "gender": "female", "traits": traits}
        for trial_seed in range(200):
            trial_traits = {}
            trng = trial_seed + 1
            for name in traits:
                trng = (trng * 1103515245 + 12345) & 0xFFFFFFFF
                trial_traits[name] = trng % 101
            if personality_descriptor(trial_traits) == descriptor:
                profile["traits"] = trial_traits
                break

        # Python's hash() on strings is randomized per-process by default
        # (PYTHONHASHSEED) -- must not be used anywhere a reproducible
        # seed is required. A trivial deterministic checksum instead.
        combo_checksum = sum(ord(c) for c in occupation + descriptor)
        rel = random_relationship_state(seed + combo_checksum)
        prompt = prompt_fields(profile, rel, "cheerful", "greeting")
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
        "/* AUTOGENERATED by trainer/make_m9_blob.py - do not edit.\n"
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
        "/* AUTOGENERATED by trainer/make_m9_blob.py - do not edit.\n"
        " * Prompts + SEEDED SAMPLED golden responses replayed by the ROM\n"
        " * boot self-test. Must stay in sync with tests/vectors/m9_goldens.bin. */\n"
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
    selena_pairs, thin_pairs, guard_pairs, m9_pairs = build_all_pairs()
    train_pairs, val_pairs, held_combos = combo_split(selena_pairs)
    # guard + M9 corpora: 100% train, no holdout at the line level -- M9's
    # holdout is at the (occupation, descriptor) COMBO level, already
    # enforced by m9_corpus.py never generating those lines in the first
    # place (assert_no_holdout_leak() re-verifies this above).
    all_train = train_pairs + thin_pairs + guard_pairs + m9_pairs
    full_text = (sc.corpus_text(seed=SEED, per_combo=PER_COMBO)
                + "".join(p + r for p, r in thin_pairs)
                + "".join(p + r for p, r in guard_pairs)
                + m9c.combo_text())
    vocab = Vocab.from_text(full_text)
    print(f"corpus: {len(selena_pairs)} selena + {len(thin_pairs)} thin-identity + "
         f"{len(guard_pairs)} guard + {len(m9_pairs)} M9 compositional pairs "
         f"({len(full_text)} chars, {len(full_text)/1e6:.2f} MB)")
    print(f"combo split: {len(train_pairs)} train-combo lines, {len(val_pairs)} "
         f"held-out-combo lines, {len(held_combos)} combos held out of Selena's 120")
    print(f"M9 combo-level holdout: {len(m9c.holdout_pairs())} (occupation, "
         f"descriptor) pairs never in training")
    print(f"vocab: {len(vocab)} symbols (incl EOS)")

    model = get_model(all_train, val_pairs, vocab)
    q = quantize(model)
    print(f"trained: H={HIDDEN}, val loss {model.final_loss:.4f} "
         f"(M8 was {M8_VAL_LOSS:.4f} at H=256)")

    probe = val_pairs[::max(1, len(val_pairs) // 150)][:150]
    agree = top1_agreement(model, q, vocab, probe)
    print(f"int-vs-float top-1 agreement (held-out combos, completion-only): {agree:.4f}")
    if agree < MIN_AGREEMENT:
        print(f"FATAL: agreement {agree:.4f} < {MIN_AGREEMENT}")
        sys.exit(1)

    raw_m9 = m9c.load_raw()
    golden_combos = curated_golden_combos()
    golden_pairs = []
    for combo in golden_combos:
        npc_id = combo[0]
        if npc_id == "selena":
            _, trust, mood, context = combo
            event = sc.EVENTS_FOR_CONTEXT[context][0]
            prompt = sc.prompt_for(trust, mood, context, event)
        else:
            _, trust, mood, context = combo
            prompt = gc.prompt_for(npc_id, trust, mood, context)
        got = generate_sampled(q, vocab, prompt, seed=SAMPLE_SEED,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K)
        print(f"  {prompt}{got}")
        if not (1 <= len(got) <= MAX_GOLDEN_LEN):
            print(f"FATAL: degenerate golden for {prompt!r}: {got!r}")
            sys.exit(1)
        golden_pairs.append((prompt, got))

    for prompt in m9_golden_prompts(raw_m9):
        got = generate_sampled(q, vocab, prompt, seed=SAMPLE_SEED,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K)
        print(f"  {prompt}{got}")
        if not (1 <= len(got) <= MAX_GOLDEN_LEN):
            print(f"FATAL: degenerate golden for {prompt!r}: {got!r}")
            sys.exit(1)
        golden_pairs.append((prompt, got))

    div_table = conditioning_divergence_table(q, vocab, held_combos)
    print("Selena conditioning-ablation divergence (trigram-Jaccard), vs. M8:")
    for axis, d in div_table.items():
        m8_val = M8_DIVERGENCE.get(axis)
        m8_str = f"{m8_val:.4f}" if m8_val is not None else "not recorded"
        print(f"  {axis}: {d:.4f}  (M8: {m8_str})")
    if div_table["identity"] < div_table["mood"] * 0.8:
        print(f"WARNING: Selena's identity divergence ({div_table['identity']:.4f}) is "
             f"well below her mood divergence ({div_table['mood']:.4f}) — possible "
             f"catastrophic interference from the M9 corpus addition.")

    print(f"\ngeneralization check ({GENERALIZATION_SAMPLE_SIZE} held-out "
         f"(occupation, descriptor) combos, never seen in training):")
    gen_results = generalization_check(q, vocab)
    degenerate_count = sum(1 for r in gen_results if r["degenerate"])
    for r in gen_results:
        flag = "DEGENERATE" if r["degenerate"] else "ok"
        print(f"  [{flag}] {r['occupation']}/{r['descriptor']}: "
             f"{r['prompt']}\n    -> {r['output']!r}")
    print(f"generalization: {len(gen_results) - degenerate_count}/{len(gen_results)} "
         f"held-out combos produced non-degenerate output")
    if degenerate_count > len(gen_results) // 2:
        print(f"WARNING: more than half of held-out combos degenerated — the "
             f"compositional-generalization hypothesis is NOT supported by this "
             f"run. Run llm_judge.py against these specific outputs for the "
             f"actual voice-quality read before drawing conclusions.")

    blob = build_blob(q, vocab)
    targets = {
        REPO / "game" / "rawfs" / "model.bin": blob,
        REPO / "tests" / "vectors" / "m9_gru.bin": blob,
        REPO / "tests" / "vectors" / "m9_goldens.bin":
            goldens_bytes(golden_pairs),
        REPO / "tests" / "vectors" / "m9_trace.bin":
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
