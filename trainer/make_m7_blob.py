#!/usr/bin/env python3
"""Build the M7 Selena blob, golden vectors, and ROM self-test header.

Pipeline (docs/milestones/m7.md): Selena's schema-conditioned corpus +
a thin placeholder second identity -> train H=256 with prefix-loss
masking and a combo-level holdout split -> quantize -> ACCEPTANCE GATES
-> emit everything in one run so nothing can drift:
  game/rawfs/model.bin            blob packed into the ROM filesystem
  tests/vectors/m7_gru.bin        identical copy, used by host tests
  tests/vectors/m7_goldens.bin    sampler params + curated (prompt, seeded
                                  sampled response) pairs across the grid
  tests/vectors/m7_trace.bin      per-step sampled goldens for pair 0
  game/src/user/selftestGolden.h  prompts+goldens+params (replaces M4's)
M4's own vectors/tests are untouched — test_matvec_hook.cpp and
test_sampled_model.cpp keep proving the frozen streaming API against the
M4 golden set; this script only adds the M7 generation, never edits M4's.

Acceptance gates (m7.md): val-loss (recorded, no fixed threshold — first
training round establishes the baseline like M4 did) + int8-vs-float
top-1 agreement >= 95% on held-out COMBOS (not just held-out lines) +
conditioning-collapse guard (identity/mood/trust/context divergence,
trigram-Jaccard per the spike's calibrated metric) + every curated
golden non-degenerate.

Training is cached in trainer/.m7_model.pt (git-ignored); delete it to
retrain. Run: uv run python make_m7_blob.py   (from trainer/)
"""
import random
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from ngpt_trainer import selena_corpus as sc
from ngpt_trainer.divergence import cross_set_divergence
from ngpt_trainer.export import build_blob, trace_bytes
from ngpt_trainer.model import CharGRU, train_corpus_conditioned
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled, gru_step, prime, trace_sampled
from ngpt_trainer.sampler_lut import LUT_EXP2
from ngpt_trainer.vocab import Vocab

REPO = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).resolve().parent / ".m7_model.pt"

SEED = 0
HIDDEN = 256
PER_COMBO = 300           # round 1 of the m7.md iteration loop — see docs/milestones/m7.md
HOLDOUT_COMBOS = 20       # of 120, combo-level (not line-level) holdout
SAMPLE_SEED = 0xC0FFEE
INV_T_Q8 = 384            # T = 2/3, same choice M4 made after its own eyeball gate
TOP_K = 5
MIN_AGREEMENT = 0.95
MAX_GOLDEN_LEN = 300      # companion opener+body+closer runs longer than M4's single clause
DIVERGENCE_SAMPLES = 5
DIVERGENCE_TEMPERATURE_INV_T_Q8 = 200  # a bit hotter than shipped sampling, for metric sensitivity


def build_all_pairs():
    selena_pairs = sc.generate_pairs(seed=SEED, per_combo=PER_COMBO)
    thin_pairs = sc.generate_thin_identity_pairs(seed=1000)
    return selena_pairs, thin_pairs


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
    # device="cpu" forced: MPS training competes with qwen's MLX 14b
    # server for the same unified GPU memory on this Mac and OOM-crashed
    # mid-run (kIOGPUCommandBufferCallbackErrorOutOfMemory) when both ran
    # at once. CPU is slower but doesn't touch the GPU's memory pool, so
    # it's safe to run alongside qwen-worker dispatches.
    model = train_corpus_conditioned(train_pairs, val_pairs, vocab, hidden=HIDDEN,
                                     seed=SEED, max_epochs=120, patience=12,
                                     device="cpu")
    torch.save({"state": model.state_dict(), "val_loss": model.final_loss,
                "hidden": HIDDEN}, CACHE)
    return model


def top1_agreement(model, q, vocab, probe: list[tuple[str, str]]) -> float:
    """Teacher-forced float-vs-int argmax agreement, scored ONLY on the
    completion (positions >= prompt length) — matches what prefix-loss
    masking actually trained the model to get right; M4 didn't need this
    distinction (no schema to mask over)."""
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
    """~18 combos spanning every axis at least twice — full ROM boot
    coverage of all 120 would blow up boot time; this mirrors M4's own
    "one golden per representative condition" choice at Selena's scale."""
    tm_cycle = [(0, "cheerful"), (2, "embarrassed"), (1, "sassy")]
    combos = []
    for i, context in enumerate(sc.CONTEXTS):
        trust, mood = tm_cycle[i % len(tm_cycle)]
        combos.append((trust, mood, context))
    combos += [
        (1, "worried", "quiet-moment"), (2, "tender", "quiet-moment"),
        (0, "sassy", "combat-banter"), (2, "cheerful", "joke"),
        (0, "tender", "farewell"), (1, "embarrassed", "joke"),
    ]
    return combos


def conditioning_divergence_table(q, vocab, held_combos) -> dict[str, float]:
    """Per-axis conditioning-ablation divergence (m7.md Evaluation
    Protocol), via the quantized model (ref_impl.generate_sampled) —
    the actual shipped behavior, not the float training scaffold."""
    def draws(prompt, base_seed):
        return [generate_sampled(q, vocab, prompt, seed=base_seed + i,
                                 inv_t_q8=DIVERGENCE_TEMPERATURE_INV_T_Q8, top_k=TOP_K,
                                 max_len=MAX_GOLDEN_LEN)
               for i in range(DIVERGENCE_SAMPLES)]

    rng = random.Random(42)
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


def repetition_check(q, vocab, combos) -> dict[tuple, float]:
    """Self-similarity within a combo at the shipped sampler settings —
    high overlap reads as "vending machine," not alive (m7.md)."""
    out = {}
    for trust, mood, context in combos:
        event = sc.EVENTS_FOR_CONTEXT[context][0]
        prompt = sc.prompt_for(trust, mood, context, event)
        draws = [generate_sampled(q, vocab, prompt, seed=0x9000 + i,
                                  inv_t_q8=INV_T_Q8, top_k=TOP_K,
                                  max_len=MAX_GOLDEN_LEN)
                for i in range(DIVERGENCE_SAMPLES)]
        # self-similarity = 1 - mean pairwise divergence among the draws
        from ngpt_trainer.divergence import jaccard_distance
        pairs = [(a, b) for i, a in enumerate(draws) for b in draws[i + 1:]]
        mean_div = sum(jaccard_distance(a, b) for a, b in pairs) / len(pairs)
        out[(trust, mood, context)] = 1.0 - mean_div
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
        "/* AUTOGENERATED by trainer/make_m7_blob.py - do not edit.\n"
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
        "/* AUTOGENERATED by trainer/make_m7_blob.py - do not edit.\n"
        " * Prompts + SEEDED SAMPLED golden responses replayed by the ROM\n"
        " * boot self-test. Must stay in sync with tests/vectors/m7_goldens.bin. */\n"
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
    selena_pairs, thin_pairs = build_all_pairs()
    train_pairs, val_pairs, held_combos = combo_split(selena_pairs)
    all_train = train_pairs + thin_pairs  # thin identity: 100% train, no holdout
    full_text = sc.corpus_text(seed=SEED, per_combo=PER_COMBO) + "".join(
        p + r for p, r in thin_pairs)
    vocab = Vocab.from_text(full_text)
    print(f"corpus: {len(selena_pairs)} selena pairs ({len(full_text)} chars, "
         f"{len(full_text)/1e6:.2f} MB) + {len(thin_pairs)} thin-identity pairs")
    print(f"combo split: {len(train_pairs)} train-combo lines, {len(val_pairs)} "
         f"held-out-combo lines, {len(held_combos)} combos held out of 120")
    print(f"vocab: {len(vocab)} symbols (incl EOS)")

    model = get_model(all_train, val_pairs, vocab)
    q = quantize(model)
    print(f"trained: H={HIDDEN}, val loss {model.final_loss:.4f}")

    # gate 1: int8 faithfully reproduces the float model on held-out COMBOS
    probe = val_pairs[::max(1, len(val_pairs) // 150)][:150]
    agree = top1_agreement(model, q, vocab, probe)
    print(f"int-vs-float top-1 agreement (held-out combos, completion-only): {agree:.4f}")
    if agree < MIN_AGREEMENT:
        print(f"FATAL: agreement {agree:.4f} < {MIN_AGREEMENT}")
        sys.exit(1)

    # gate 2: curated seeded sampled goldens, degenerate-checked
    golden_combos = curated_golden_combos()
    golden_pairs = []
    for trust, mood, context in golden_combos:
        event = sc.EVENTS_FOR_CONTEXT[context][0]
        prompt = sc.prompt_for(trust, mood, context, event)
        got = generate_sampled(q, vocab, prompt, seed=SAMPLE_SEED,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K,
                               max_len=MAX_GOLDEN_LEN)
        print(f"  {prompt}{got}")
        if not (1 <= len(got) <= MAX_GOLDEN_LEN):
            print(f"FATAL: degenerate golden for {prompt!r}: {got!r}")
            sys.exit(1)
        golden_pairs.append((prompt, got))

    # gate 3: conditioning-collapse guard — per-axis divergence table
    div_table = conditioning_divergence_table(q, vocab, held_combos)
    print("conditioning-ablation divergence (trigram-Jaccard):")
    for axis, d in div_table.items():
        print(f"  {axis}: {d:.4f}")
    if div_table["identity"] < div_table["mood"] * 0.8:
        print(f"WARNING: identity divergence ({div_table['identity']:.4f}) is well below "
             f"mood divergence ({div_table['mood']:.4f}) — conditioning-collapse risk on "
             f"the identity axis at H=256; see docs/spikes/identity-conditioning.md before "
             f"treating this as shippable.")

    # gate 4: repetition / mode-collapse self-similarity per combo
    rep = repetition_check(q, vocab, golden_combos[:6])
    print("repetition self-similarity (1.0 = identical every draw):")
    for combo, score in rep.items():
        print(f"  {combo}: {score:.4f}")

    blob = build_blob(q, vocab)
    targets = {
        REPO / "game" / "rawfs" / "model.bin": blob,
        REPO / "tests" / "vectors" / "m7_gru.bin": blob,
        REPO / "tests" / "vectors" / "m7_goldens.bin":
            goldens_bytes(golden_pairs),
        REPO / "tests" / "vectors" / "m7_trace.bin":
            # max_len=MAX_GOLDEN_LEN: same latent bug class M12.1 found and
            # fixed (docs/plan.md Known Follow-ups) -- back-ported here.
            trace_bytes(trace_sampled(q, vocab, golden_pairs[0][0],
                                      SAMPLE_SEED, INV_T_Q8, TOP_K,
                                      max_len=MAX_GOLDEN_LEN), q.H),
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
