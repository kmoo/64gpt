#!/usr/bin/env python3
"""M12.4 host-only spike -- redundancy ablation on M12.3's honest negative.

M12.3 (docs/milestones/m12.3.md) fed D:/M: as constant per-step one-hot
columns WHILE LEAVING them in the text prefix too -- and scored WORSE
(2.33 invented/line) than M12.2's own failing prefix-only baseline
(1.65). The leading (unconfirmed) hypothesis: the columns are redundant
with the still-present prefix text, and that redundancy -- not the
per-step-conditioning idea itself -- caused the regression (a magnitude-
based quantization explanation was checked and ruled out, see m12.3.md).

This spike tests that hypothesis directly: SAME architecture, SAME
corpus, SAME training code as M12.3 (model.CharGRU with a widened
one-hot input, model.train_corpus_conditioned_attr, model.
qat_finetune_attr, ref_impl.generate_sampled_attr) -- the ONLY change is
that D:/M: are stripped out of the text prefix the model actually sees
(npc_service.strip_prompt_fields), so the constant attribute columns
become the ONLY source of that signal instead of a redundant echo of it.
attr ids themselves are still parsed from the ORIGINAL (unstripped)
prompt, same as M12.3.

DECISION GATE: same as M12.3 -- does the coherence probe (quantized, no
trie, same seeds/settings as M12.1/M12.2/M12.3) beat M12.2's 1.65 and
M12.3's 2.33, target <=1.0? Only if it clears <=1.0 do we touch the NGPT
blob format + N64 kernel. If this still fails, that's evidence AGAINST
the redundancy hypothesis and points at M12.5 (FiLM) instead.

Host-only: fp32 training + int8 quantize()/ref_impl. Writes NO blob, NO
goldens, NO ROM artifacts -- measurement only.

Training is cached in trainer/.m12_4_attr_ablation_model.pt (git-
ignored); delete to retrain. Run from trainer/, on branch m12.3-voices:
  uv run python m12_4_attr_ablation_spike.py
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
from ngpt_trainer.npc_service import parse_prompt_fields, strip_prompt_fields
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled_attr
from ngpt_trainer.vocab import Vocab

# Reuse M12.1's build config (SEED, PER_COMBO, HIDDEN, sampler settings,
# invented_word_count, build_corpus_vocab, combo_split) -- same as
# M12.3's spike, only corpus PHRASE BANKS differ on this branch.
import make_m12_1_blob as m12

STRIPPED_KEYS = {"D", "M"}

CACHE = Path(__file__).resolve().parent / ".m12_4_attr_ablation_model.pt"

M12_2_INVENTED_PER_LINE = 1.65   # prefix-only baseline
M12_3_INVENTED_PER_LINE = 2.33   # redundant-columns baseline (this ablates that)
M12_1_GATE = 1.0                 # the real target
LINES_PER_GROUP = 8
PROBE_SEED = 0xC0FFEE


def build_attr_vocabs(pairs: list[tuple[str, str]]) -> tuple[dict, dict]:
    """Same as M12.3's -- built from the ORIGINAL (unstripped) prompts,
    since D:/M: values still need to be recovered to build attr_cols even
    though the model never sees them in its input text."""
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


def strip_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The model-visible corpus: D:/M: removed from every prompt, response
    text untouched."""
    return [(strip_prompt_fields(p, STRIPPED_KEYS), r) for p, r in pairs]


def attr_cols_for(original_prompt: str, V: int, n_desc: int, desc_by_value: dict,
                  mood_by_value: dict) -> tuple[int, int]:
    fields = parse_prompt_fields(original_prompt)
    return (V + desc_by_value[fields["D"]],
            V + n_desc + mood_by_value[fields["M"]])


def run_attr_probe(q, vocab, groups, corpus_vocab, V, n_desc,
                   desc_by_value, mood_by_value) -> dict:
    """Mirrors M12.3's run_attr_probe, but generates from the STRIPPED
    prompt (what the model was actually trained on) while still printing
    the original for readability and resolving attr_cols from it."""
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
    # Vocab is built from the FULL (unstripped) text, same charset as
    # M12.3 -- stripping only removes which columns are ACTIVE per
    # example, it never introduces new characters, so this stays valid
    # for both the attr-vocab-building pass and encoding the (fewer)
    # characters the stripped prompts actually contain.
    vocab = Vocab.from_text(full_text)
    V = len(vocab)

    desc_by_value, mood_by_value = build_attr_vocabs(all_train + val_pairs)
    n_desc, n_mood = len(desc_by_value), len(mood_by_value)
    print(f"D: {n_desc} values {sorted(desc_by_value)}")
    print(f"M: {n_mood} values {sorted(mood_by_value)}")
    print(f"stripping {sorted(STRIPPED_KEYS)} from the text prefix -- attribute "
          f"columns are now the ONLY source of that signal")

    train_attrs = attrs_for(all_train, desc_by_value, mood_by_value)
    val_attrs = attrs_for(val_pairs, desc_by_value, mood_by_value)
    stripped_train = strip_pairs(all_train)
    stripped_val = strip_pairs(val_pairs)

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
            stripped_train, stripped_val, train_attrs, val_attrs, vocab,
            n_desc=n_desc, n_mood=n_mood, hidden=m12.HIDDEN, seed=m12.SEED,
            max_epochs=120, patience=12, device="mps")
        float_val = model.final_loss
        print(f"float phase done: val loss {float_val:.4f} -- starting QAT fine-tune")
        model = qat_finetune_attr(
            model, stripped_train, stripped_val, train_attrs, val_attrs, vocab,
            n_desc=n_desc, n_mood=n_mood, seed=m12.SEED, lr=3e-4,
            max_epochs=30, patience=6, device="mps")
        torch.save({"state": model.state_dict(), "val_loss": model.final_loss,
                    "float_val_loss": float_val, "hidden": m12.HIDDEN}, CACHE)

    q = quantize(model)
    print(f"trained: H={m12.HIDDEN}, qat val loss {model.final_loss:.4f} "
          f"(M12.2 baseline: {M12_2_INVENTED_PER_LINE} inv/line, "
          f"M12.3 redundant-columns: {M12_3_INVENTED_PER_LINE} inv/line)")

    corpus_vocab = m12.build_corpus_vocab(full_text)
    groups = {"selena": selena_pairs, "guard": guard_pairs, "cast": cast_pairs,
              "shadewrath": shadewrath_pairs, "korrath": korrath_pairs,
              "elowen": princess_pairs}

    print(f"\ncoherence probe WITH attribute conditioning, D:/M: STRIPPED FROM PREFIX "
          f"({LINES_PER_GROUP} lines/group, shipped sampler settings "
          f"inv_t_q8={m12.INV_T_Q8} top_k={m12.TOP_K}, seed={PROBE_SEED:#x}):")
    results = run_attr_probe(q, vocab, groups, corpus_vocab, V, n_desc,
                             desc_by_value, mood_by_value)
    print(f"\n{'group':<12}{'lines':>7}{'inv/line':>10}")
    for name, r in results.items():
        print(f"{name:<12}{r['lines']:>7}{r['invented_per_line']:>10.2f}")

    all_rate = results["ALL"]["invented_per_line"]
    print(f"\nALL: {all_rate:.2f} invented/line (M12.2 {M12_2_INVENTED_PER_LINE}, "
          f"M12.3 {M12_3_INVENTED_PER_LINE}, M12.1 gate <= {M12_1_GATE})")
    if all_rate < M12_3_INVENTED_PER_LINE:
        print("Beats M12.3's redundant-columns result -- supports the redundancy hypothesis.")
    else:
        print("Does NOT beat M12.3 -- redundancy was likely not the (sole) cause; "
              "points at M12.5 (FiLM) instead.")
    if all_rate < M12_2_INVENTED_PER_LINE:
        print("Beats M12.2's prefix-only baseline too.")
    if all_rate <= M12_1_GATE:
        print("CLEARS the M12.1 coherence gate -- proceed to blob + kernel work.")
        sys.exit(0)
    else:
        print("Does not clear the M12.1 gate -- do not proceed to blob/kernel work.")
        sys.exit(1)


if __name__ == "__main__":
    main()
