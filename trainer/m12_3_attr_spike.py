#!/usr/bin/env python3
"""M12.3 host-only spike -- does per-step D:/M: attribute conditioning
(model.CharGRU with a widened one-hot input, docs/ideas-m12.3-
conditioning-strategies.md option A) beat M12.2's honest negative on the
richer-voice corpus (branch m12.3-voices, docs/milestones/m12.2.md)?

MECHANISM: D:/M: are parsed back out of the existing prompt string
(npc_service.parse_prompt_fields()) and fed as CONSTANT extra one-hot
columns alongside the char one-hot at every timestep (model.
one_hot_attr) -- mathematically identical to concatenating a small
learned embedding, but reusing CharGRU/quantize()/qat_finetune's
existing machinery unchanged (see model.py's CharGRU docstring). The
text prefix itself is UNCHANGED -- this is additive, not a replacement.

DECISION GATE: the SAME raw-model coherence probe M12.1/M12.2 used
(m12_1_coherence_probe.run_probe's methodology: quantized, no trie, same
seeds/settings) must beat M12.2's 1.65 invented/line, target <=1.0
(M12.1's own gate). Only if this passes do we touch the NGPT blob format
+ N64 kernel (the design doc's own sequencing) -- m12_1_coherence_probe.py
itself is left untouched (the frozen tool every prior milestone is judged
against); this script mirrors its methodology for the attr-conditioned
decode path instead of modifying it.

Host-only: fp32 training + int8 quantize()/ref_impl, like every other
milestone's blob-build script -- but this script writes NO blob, NO
goldens, NO ROM artifacts. It only measures.

Training is cached in trainer/.m12_3_attr_model.pt (git-ignored); delete
to retrain. Run from trainer/, on branch m12.3-voices:
  uv run python m12_3_attr_spike.py
"""
import random
import sys
import zlib
from pathlib import Path

import torch

from ngpt_trainer import cast_corpus as cc
from ngpt_trainer import guard_corpus as gc
from ngpt_trainer import korrath_corpus as kc
from ngpt_trainer import princess_corpus as pc
from ngpt_trainer import selena_corpus as sc
from ngpt_trainer import shadewrath_corpus as swc
from ngpt_trainer.model import CharGRU, qat_finetune_attr, train_corpus_conditioned_attr
from ngpt_trainer.npc_service import parse_prompt_fields
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled_attr
from ngpt_trainer.vocab import Vocab

# Reuse M12.1's build config (SEED, PER_COMBO, HIDDEN, sampler settings,
# invented_word_count, build_corpus_vocab, combo_split) -- only the
# corpus PHRASE BANKS differ on this branch (git diff b942a07..
# m12.3-voices touches only the six *_corpus.py phrase banks, confirmed
# before writing this script), the build parameters are identical.
import make_m12_1_blob as m12

CACHE = Path(__file__).resolve().parent / ".m12_3_attr_model.pt"

M12_2_INVENTED_PER_LINE = 1.65   # this spike's baseline to beat (honest negative)
M12_1_GATE = 1.0                 # the real target (M12.1's shipped gate)
LINES_PER_GROUP = 8
PROBE_SEED = 0xC0FFEE


def build_attr_vocabs(pairs: list[tuple[str, str]]) -> tuple[dict, dict]:
    """(desc_id_by_value, mood_id_by_value), built from the ACTUAL D:/M:
    values appearing in the corpus (sorted, deterministic) -- same idiom
    Vocab.from_text() uses for the char charset, not a hand-maintained
    list that could drift from the corpus."""
    descs, moods = set(), set()
    for prompt, _ in pairs:
        fields = parse_prompt_fields(prompt)
        descs.add(fields["D"])
        moods.add(fields["M"])
    return ({v: i for i, v in enumerate(sorted(descs))},
            {v: i for i, v in enumerate(sorted(moods))})


def attrs_for(pairs: list[tuple[str, str]], desc_by_value: dict,
             mood_by_value: dict) -> list[tuple[int, int]]:
    out = []
    for prompt, _ in pairs:
        fields = parse_prompt_fields(prompt)
        out.append((desc_by_value[fields["D"]], mood_by_value[fields["M"]]))
    return out


def attr_cols_for(prompt: str, V: int, n_desc: int, desc_by_value: dict,
                  mood_by_value: dict) -> tuple[int, int]:
    fields = parse_prompt_fields(prompt)
    return (V + desc_by_value[fields["D"]],
            V + n_desc + mood_by_value[fields["M"]])


def run_attr_probe(q, vocab, groups, corpus_vocab, V, n_desc,
                   desc_by_value, mood_by_value) -> dict:
    """Mirrors m12_1_coherence_probe.run_probe's methodology (same
    per-group seeding via zlib.crc32, same LINES_PER_GROUP, same shipped
    sampler settings) but resolves attr_cols per prompt and calls
    generate_sampled_attr instead of generate_sampled."""
    results = {}
    tot_inv = tot_lines = 0
    for name, pairs in groups.items():
        rng = random.Random(PROBE_SEED ^ (zlib.crc32(name.encode()) & 0xFFFF))
        sample = rng.sample(pairs, min(LINES_PER_GROUP, len(pairs)))
        inv = 0
        for i, (prompt, _) in enumerate(sample):
            attr_cols = attr_cols_for(prompt, V, n_desc, desc_by_value, mood_by_value)
            got = generate_sampled_attr(q, vocab, prompt, attr_cols=attr_cols,
                                        seed=PROBE_SEED + i, inv_t_q8=m12.INV_T_Q8,
                                        top_k=m12.TOP_K, max_len=m12.MAX_GOLDEN_LEN)
            n = m12.invented_word_count(got, corpus_vocab)
            inv += n
            print(f"  [{name}] [{n} inv] {prompt}{got}")
        k = len(sample)
        results[name] = {"lines": k, "invented_per_line": inv / k}
        tot_inv += inv
        tot_lines += k
    results["ALL"] = {"lines": tot_lines, "invented_per_line": tot_inv / tot_lines}
    return results


def main() -> None:
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
    print(f"D: {n_desc} values {sorted(desc_by_value)}")
    print(f"M: {n_mood} values {sorted(mood_by_value)}")

    train_attrs = attrs_for(all_train, desc_by_value, mood_by_value)
    val_attrs = attrs_for(val_pairs, desc_by_value, mood_by_value)

    if CACHE.exists():
        d = torch.load(CACHE, weights_only=True)
        model = CharGRU(vocab_size=V, hidden=d["hidden"], input_size=V + n_desc + n_mood)
        model.load_state_dict(d["state"])
        model.eval()
        model.final_loss = d["val_loss"]
        print(f"loaded cached model (qat val loss {model.final_loss:.4f}, "
              f"float val loss {d.get('float_val_loss', float('nan')):.4f})")
    else:
        model = train_corpus_conditioned_attr(
            all_train, val_pairs, train_attrs, val_attrs, vocab,
            n_desc=n_desc, n_mood=n_mood, hidden=m12.HIDDEN, seed=m12.SEED,
            max_epochs=120, patience=12, device="mps")
        float_val = model.final_loss
        print(f"float phase done: val loss {float_val:.4f} -- starting QAT fine-tune")
        model = qat_finetune_attr(
            model, all_train, val_pairs, train_attrs, val_attrs, vocab,
            n_desc=n_desc, n_mood=n_mood, seed=m12.SEED, lr=3e-4,
            max_epochs=30, patience=6, device="mps")
        torch.save({"state": model.state_dict(), "val_loss": model.final_loss,
                    "float_val_loss": float_val, "hidden": m12.HIDDEN}, CACHE)

    q = quantize(model)
    print(f"trained: H={m12.HIDDEN}, qat val loss {model.final_loss:.4f} "
          f"(M12.2's raw-model probe baseline: {M12_2_INVENTED_PER_LINE} invented/line)")

    corpus_vocab = m12.build_corpus_vocab(full_text)
    groups = {"selena": selena_pairs, "guard": guard_pairs, "cast": cast_pairs,
              "shadewrath": shadewrath_pairs, "korrath": korrath_pairs,
              "elowen": princess_pairs}

    print(f"\ncoherence probe WITH attribute conditioning "
          f"({LINES_PER_GROUP} lines/group, shipped sampler settings "
          f"inv_t_q8={m12.INV_T_Q8} top_k={m12.TOP_K}, seed={PROBE_SEED:#x}):")
    results = run_attr_probe(q, vocab, groups, corpus_vocab, V, n_desc,
                             desc_by_value, mood_by_value)
    print(f"\n{'group':<12}{'lines':>7}{'inv/line':>10}")
    for name, r in results.items():
        print(f"{name:<12}{r['lines']:>7}{r['invented_per_line']:>10.2f}")

    all_rate = results["ALL"]["invented_per_line"]
    print(f"\nALL: {all_rate:.2f} invented/line "
          f"(M12.2 baseline {M12_2_INVENTED_PER_LINE}, M12.1 gate <= {M12_1_GATE})")
    if all_rate < M12_2_INVENTED_PER_LINE:
        print("Beats M12.2's honest-negative baseline.")
    else:
        print("Does NOT beat M12.2's baseline.")
    if all_rate <= M12_1_GATE:
        print("CLEARS the M12.1 coherence gate -- proceed to blob + kernel work.")
        sys.exit(0)
    else:
        print("Does not clear the M12.1 gate -- do not proceed to blob/kernel work "
              "without further iteration (e.g. FiLM, option B).")
        sys.exit(1)


if __name__ == "__main__":
    main()
