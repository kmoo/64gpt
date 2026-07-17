#!/usr/bin/env python3
"""Build the M8 blob, golden vectors, and ROM self-test header — Selena
plus the `guard` archetype's 4 trained instances, one shared model.

Pipeline (docs/milestones/m8.md): Selena's M7 corpus + thin-identity
pairs + the new guard archetype corpus (trainer/ngpt_trainer/
guard_corpus.py, 4 concrete seeded instances) -> retrain H=256 with the
same prefix-loss masking and combo-level holdout M7 used -> quantize ->
ACCEPTANCE GATES (M7's gates, unchanged, PLUS the new within-archetype
divergence gate) -> emit everything in one run so nothing can drift.
Mirrors make_m7_blob.py almost exactly; only the corpus mix and the new
gate are M8-specific — see that script's docstring for the full output
list and gate rationale this one reuses unchanged.

**Catastrophic-interference check**: Selena's own val-loss and per-axis
divergence table are recomputed here and printed next to M7's recorded
numbers (val loss 0.0968; identity 0.9405 / mood 0.9168 / trust 0.9099 /
context 0.9409 — docs/milestones/m7.md) so adding the guard archetype's
corpus to the shared model's training data can't silently degrade
Selena's voice without someone noticing, per M8's Data Science Review.

**New gate — within-archetype divergence**: do guard#1001 and guard#1002
(same archetype, different trained instance) actually diverge when only
the N: id is swapped, holding trust/mood/context fixed? Same
trigram-Jaccard methodology as M7's identity-swap gate, with a
guard-specific mood-swap baseline computed the same session (there's no
existing "guard mood divergence" number to compare against, unlike
Selena's).

Training is cached in trainer/.m8_model.pt (git-ignored); delete it to
retrain. Run: uv run python make_m8_blob.py   (from trainer/)
"""
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from ngpt_trainer import selena_corpus as sc
from ngpt_trainer import guard_corpus as gc
from ngpt_trainer.divergence import cross_set_divergence
from ngpt_trainer.export import build_blob, trace_bytes
from ngpt_trainer.model import CharGRU, train_corpus_conditioned
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled, gru_step, trace_sampled
from ngpt_trainer.sampler_lut import LUT_EXP2
from ngpt_trainer.vocab import Vocab

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / ".m8_model.pt"

SEED = 0
HIDDEN = 256
PER_COMBO = 300            # Selena's corpus density, unchanged from M7
GUARD_PER_COMBO = 24       # bumped 3->12->24 (2026-07-17): 3 gave 1/4 clean
                           # goldens (whole-line collapse), 12 gave 2/4 clean
                           # (remaining failures shrank to tail-end garbling,
                           # not full collapse) -- real but incomplete progress,
                           # so pushing density once more. 24 brings guard's
                           # share of the combined corpus to ~12.6% (4320 of
                           # ~34K pairs); see docs/milestones/m8.md's
                           # Evaluation Protocol note
HOLDOUT_COMBOS = 20        # of Selena's 120, combo-level holdout (unchanged from M7)
SAMPLE_SEED = 0xC0FFEE
INV_T_Q8 = 384
TOP_K = 5
MIN_AGREEMENT = 0.95
MAX_GOLDEN_LEN = 300
DIVERGENCE_SAMPLES = 5
DIVERGENCE_TEMPERATURE_INV_T_Q8 = 200

# M7's recorded numbers (docs/milestones/m7.md), for the catastrophic-
# interference comparison — not recomputed from a saved checkpoint,
# since M7's own acceptance run already established them.
M7_VAL_LOSS = 0.0968
M7_DIVERGENCE = {"identity": 0.9405, "mood": 0.9168, "trust": 0.9099, "context": 0.9409}


def build_all_pairs():
    selena_pairs = sc.generate_pairs(seed=SEED, per_combo=PER_COMBO)
    thin_pairs = sc.generate_thin_identity_pairs(seed=1000)
    guard_pairs = gc.generate_pairs(seed=SEED, per_combo=GUARD_PER_COMBO)
    return selena_pairs, thin_pairs, guard_pairs


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
    # device="cpu" forced — see make_m7_blob.py's identical comment:
    # MPS training competes with qwen's MLX server for unified GPU
    # memory and OOM-crashes both if they run at once.
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
    """Same M7 Selena grid, plus one representative combo per guard
    instance so the ROM self-test replays evidence for all 5 trained
    identities, not just Selena."""
    tm_cycle = [(0, "cheerful"), (2, "embarrassed"), (1, "sassy")]
    combos = [("selena", *tm_cycle[i % len(tm_cycle)], context)
              for i, context in enumerate(sc.CONTEXTS)]
    combos += [
        ("selena", 1, "worried", "quiet-moment"), ("selena", 2, "tender", "quiet-moment"),
        ("selena", 0, "sassy", "combat-banter"), ("selena", 2, "cheerful", "joke"),
        ("selena", 0, "tender", "farewell"), ("selena", 1, "embarrassed", "joke"),
    ]
    combos += [(gid, 1, "cheerful", "greeting") for gid in gc.GUARD_IDS]
    return combos


def conditioning_divergence_table(q, vocab, held_combos) -> dict[str, float]:
    """Selena's per-axis divergence table, unchanged methodology from
    M7 — the catastrophic-interference check compares this against
    M7_DIVERGENCE above."""
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


def guard_divergence_table(q, vocab) -> dict[str, float]:
    """New M8 gate: within-archetype divergence. id-swap holds trust/
    mood/context fixed and swaps only which trained guard instance is
    speaking — same trigram-Jaccard methodology as Selena's identity
    axis. mood-swap (same guard, two moods) is computed here as this
    gate's own baseline, since there's no prior guard-specific number to
    compare against the way Selena's mood axis already exists."""
    def draws(prompt, base_seed):
        return [generate_sampled(q, vocab, prompt, seed=base_seed + i,
                                 inv_t_q8=DIVERGENCE_TEMPERATURE_INV_T_Q8, top_k=TOP_K)
               for i in range(DIVERGENCE_SAMPLES)]

    trust, mood, context = 1, "cheerful", "greeting"

    id_a = draws(gc.prompt_for("guard#1001", trust, mood, context), 0xA000)
    id_b = draws(gc.prompt_for("guard#1002", trust, mood, context), 0xB000)
    within_archetype_div = cross_set_divergence(id_a, id_b)

    mood_a = draws(gc.prompt_for("guard#1001", trust, "cheerful", context), 0xC000)
    mood_b = draws(gc.prompt_for("guard#1001", trust, "worried", context), 0xD000)
    guard_mood_div = cross_set_divergence(mood_a, mood_b)

    return {"within_archetype": within_archetype_div, "guard_mood_baseline": guard_mood_div}


def repetition_check(q, vocab, combos) -> dict[tuple, float]:
    out = {}
    for npc_id, trust, mood, context in combos:
        if npc_id == "selena":
            event = sc.EVENTS_FOR_CONTEXT[context][0]
            prompt = sc.prompt_for(trust, mood, context, event)
        else:
            prompt = gc.prompt_for(npc_id, trust, mood, context)
        draws = [generate_sampled(q, vocab, prompt, seed=0x9000 + i,
                                  inv_t_q8=INV_T_Q8, top_k=TOP_K)
                for i in range(DIVERGENCE_SAMPLES)]
        from ngpt_trainer.divergence import jaccard_distance
        pairs = [(a, b) for i, a in enumerate(draws) for b in draws[i + 1:]]
        mean_div = sum(jaccard_distance(a, b) for a, b in pairs) / len(pairs)
        out[(npc_id, trust, mood, context)] = 1.0 - mean_div
    return out


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
        "/* AUTOGENERATED by trainer/make_m8_blob.py - do not edit.\n"
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
        "/* AUTOGENERATED by trainer/make_m8_blob.py - do not edit.\n"
        " * Prompts + SEEDED SAMPLED golden responses replayed by the ROM\n"
        " * boot self-test. Must stay in sync with tests/vectors/m8_goldens.bin. */\n"
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
    selena_pairs, thin_pairs, guard_pairs = build_all_pairs()
    train_pairs, val_pairs, held_combos = combo_split(selena_pairs)
    # guard corpus: 100% train, no holdout — same treatment as the thin
    # identity, both are "prove the mechanism," not full-coverage content
    all_train = train_pairs + thin_pairs + guard_pairs
    full_text = (sc.corpus_text(seed=SEED, per_combo=PER_COMBO)
                + "".join(p + r for p, r in thin_pairs)
                + "".join(p + r for p, r in guard_pairs))
    vocab = Vocab.from_text(full_text)
    print(f"corpus: {len(selena_pairs)} selena pairs + {len(thin_pairs)} thin-identity "
         f"pairs + {len(guard_pairs)} guard pairs ({len(full_text)} chars, "
         f"{len(full_text)/1e6:.2f} MB)")
    print(f"combo split: {len(train_pairs)} train-combo lines, {len(val_pairs)} "
         f"held-out-combo lines, {len(held_combos)} combos held out of Selena's 120")
    print(f"vocab: {len(vocab)} symbols (incl EOS)")

    model = get_model(all_train, val_pairs, vocab)
    q = quantize(model)
    print(f"trained: H={HIDDEN}, val loss {model.final_loss:.4f} "
         f"(M7 was {M7_VAL_LOSS:.4f})")

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

    # gate: Selena's own conditioning-collapse guard, unchanged from M7 —
    # this IS the catastrophic-interference check: if adding the guard
    # corpus quietly hurt Selena's identity signal, it shows up here.
    div_table = conditioning_divergence_table(q, vocab, held_combos)
    print("Selena conditioning-ablation divergence (trigram-Jaccard), vs. M7:")
    for axis, d in div_table.items():
        print(f"  {axis}: {d:.4f}  (M7: {M7_DIVERGENCE[axis]:.4f})")
    if div_table["identity"] < div_table["mood"] * 0.8:
        print(f"WARNING: Selena's identity divergence ({div_table['identity']:.4f}) is "
             f"well below her mood divergence ({div_table['mood']:.4f}) — possible "
             f"catastrophic interference from the guard archetype's added corpus.")

    # new gate: within-archetype divergence
    guard_div = guard_divergence_table(q, vocab)
    print("guard within-archetype divergence (trigram-Jaccard):")
    for axis, d in guard_div.items():
        print(f"  {axis}: {d:.4f}")
    if guard_div["within_archetype"] < guard_div["guard_mood_baseline"] * 0.8:
        print(f"WARNING: within-archetype divergence ({guard_div['within_archetype']:.4f}) "
             f"is well below the guard mood-swap baseline "
             f"({guard_div['guard_mood_baseline']:.4f}) — instance personality may be "
             f"washing out, same failure mode M7's identity gate checks for.")

    rep_combos = golden_combos[:6] + [(gid, 1, "cheerful", "greeting") for gid in gc.GUARD_IDS]
    rep = repetition_check(q, vocab, rep_combos)
    print("repetition self-similarity (1.0 = identical every draw):")
    for combo, score in rep.items():
        print(f"  {combo}: {score:.4f}")

    blob = build_blob(q, vocab)
    targets = {
        REPO / "game" / "rawfs" / "model.bin": blob,
        REPO / "tests" / "vectors" / "m8_gru.bin": blob,
        REPO / "tests" / "vectors" / "m8_goldens.bin":
            goldens_bytes(golden_pairs),
        REPO / "tests" / "vectors" / "m8_trace.bin":
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
