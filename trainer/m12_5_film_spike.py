#!/usr/bin/env python3
"""M12.5 host-only spike -- FiLM (option B, docs/ideas-m12.3-conditioning-
strategies.md) on top of M12.4's redundancy ablation.

M12.3 (docs/milestones/m12.3.md): per-step D:/M: attribute columns
concatenated onto the char input, REDUNDANT with the still-present text
prefix -- 2.33 inv/line, worse than M12.2's 1.65.
M12.4 (docs/milestones/m12.4.md): same mechanism, D:/M: stripped from the
prefix so the columns are the sole signal -- 1.44 inv/line, real progress,
still short of the <=1.0 gate. selena/cast already clear it; guard/korrath
lag.

Both M12.3/M12.4 are "option A": the attribute reaches the model once, as
extra input columns, and the model must carry that signal forward on its
own across the whole reply. This spike tests "option B" instead: FiLM
(model.CharGRUFiLM) applies a per-channel gamma/beta to the GRU's hidden
state EVERY timestep, re-injecting the attribute at every step so it can't
decay the way a one-shot input signal could. D:/M: stay STRIPPED from the
text prefix (M12.4's finding that redundancy hurts still applies here --
no reason to reintroduce it).

Mechanism correctness + int8 quantization survival already proven by
tests/test_m12_5_film.py's overfit case; this script is the real-corpus
measurement. gamma/beta are squashed through the existing lut_tanh with a
FILM_SCALE=0.2 amplitude bound (model.py/ref_impl.py) -- an earlier
unbounded version compounded gamma multiplicatively across the unroll and
saturated the int16 Q14 hidden state within 2 primed characters; see both
files' docstrings for the trace that caught it.

DECISION GATE: same as M12.3/M12.4 -- does the coherence probe (quantized,
no trie, same seeds/settings as M12.1-M12.4) beat M12.4's 1.44, target
<=1.0? Only if it clears <=1.0 do we touch the NGPT blob format + N64
kernel (FiLM is a real kernel-shape change, per-channel multiply-add, not
just a wider column lookup -- see docs/ideas-m12.3-conditioning-strategies.md
options-at-a-glance table).

Host-only: fp32 training + int8 quantize_film()/ref_impl. Writes NO blob,
NO goldens, NO ROM artifacts -- measurement only.

Training is cached in trainer/.m12_5_film_model.pt (git-ignored); delete to
retrain. Run from trainer/, on branch m12.3-voices:
  uv run python m12_5_film_spike.py
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
from ngpt_trainer.model import CharGRUFiLM, qat_finetune_film, train_corpus_conditioned_film
from ngpt_trainer.npc_service import parse_prompt_fields, strip_prompt_fields
from ngpt_trainer.quantize import quantize_film
from ngpt_trainer.ref_impl import generate_sampled_film
from ngpt_trainer.vocab import Vocab

# Reuse M12.1's build config (SEED, PER_COMBO, HIDDEN, sampler settings,
# invented_word_count, build_corpus_vocab, combo_split) -- same as
# M12.3/M12.4's spikes, only the conditioning MECHANISM differs here.
import make_m12_1_blob as m12

STRIPPED_KEYS = {"D", "M"}

CACHE = Path(__file__).resolve().parent / ".m12_5_film_model.pt"

M12_2_INVENTED_PER_LINE = 1.65   # prefix-only baseline
M12_3_INVENTED_PER_LINE = 2.33   # redundant-columns (option A, not stripped)
M12_4_INVENTED_PER_LINE = 1.44   # option A, ablated -- this spike's baseline to beat
M12_1_GATE = 1.0                 # the real target
LINES_PER_GROUP = 8
PROBE_SEED = 0xC0FFEE


def build_attr_vocabs(pairs: list[tuple[str, str]]) -> tuple[dict, dict]:
    """Same as M12.3/M12.4's -- built from the ORIGINAL (unstripped)
    prompts, since D:/M: values still need to be recovered even though the
    model never sees them in its input text."""
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
    text untouched (M12.4's finding -- keep the redundancy gone)."""
    return [(strip_prompt_fields(p, STRIPPED_KEYS), r) for p, r in pairs]


def attr_cols_for(original_prompt: str, n_desc: int, desc_by_value: dict,
                  mood_by_value: dict) -> tuple[int, int]:
    """FiLM's attr_cols are indices into film's n_attr axis (desc_id,
    n_desc+mood_id) -- NOT offset by V, unlike M12.3/M12.4's attr_cols
    into W_ih (FiLM's attribute vector has no char columns to skip)."""
    fields = parse_prompt_fields(original_prompt)
    return (desc_by_value[fields["D"]], n_desc + mood_by_value[fields["M"]])


def run_film_probe(q, vocab, groups, corpus_vocab, n_desc,
                   desc_by_value, mood_by_value) -> dict:
    """Mirrors M12.4's run_attr_probe: generates from the STRIPPED prompt
    (what the model was actually trained on) while printing the original
    for readability and resolving attr_cols from it."""
    results = {}
    tot_inv = tot_lines = 0
    for name, pairs in groups.items():
        rng = random.Random(PROBE_SEED ^ (zlib.crc32(name.encode()) & 0xFFFF))
        sample = rng.sample(pairs, min(LINES_PER_GROUP, len(pairs)))
        inv = 0
        for i, (prompt, _) in enumerate(sample):
            attr_cols = attr_cols_for(prompt, n_desc, desc_by_value, mood_by_value)
            stripped_prompt = strip_prompt_fields(prompt, STRIPPED_KEYS)
            got = generate_sampled_film(q, vocab, stripped_prompt, attr_cols=attr_cols,
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
    print("FiLM conditioning: gamma/beta modulate h every step; D:/M: "
          "stripped from the text prefix (M12.4's finding)")

    train_attrs = attrs_for(all_train, desc_by_value, mood_by_value)
    val_attrs = attrs_for(val_pairs, desc_by_value, mood_by_value)
    stripped_train = strip_pairs(all_train)
    stripped_val = strip_pairs(val_pairs)

    if CACHE.exists():
        d = torch.load(CACHE, weights_only=True)
        model = CharGRUFiLM(vocab_size=V, hidden=d["hidden"], n_attr=n_desc + n_mood)
        model.load_state_dict(d["state"])
        model.eval()
        model.final_loss = d["val_loss"]
        print(f"loaded cached model (qat val loss {model.final_loss:.4f}, "
              f"float val loss {d.get('float_val_loss', float('nan')):.4f})")
    else:
        model = train_corpus_conditioned_film(
            stripped_train, stripped_val, train_attrs, val_attrs, vocab,
            n_desc=n_desc, n_mood=n_mood, hidden=m12.HIDDEN, seed=m12.SEED,
            max_epochs=120, patience=12, device="mps")
        float_val = model.final_loss
        print(f"float phase done: val loss {float_val:.4f} -- starting QAT fine-tune")
        model = qat_finetune_film(
            model, stripped_train, stripped_val, train_attrs, val_attrs, vocab,
            n_desc=n_desc, n_mood=n_mood, seed=m12.SEED, lr=3e-4,
            max_epochs=30, patience=6, device="mps")
        torch.save({"state": model.state_dict(), "val_loss": model.final_loss,
                    "float_val_loss": float_val, "hidden": m12.HIDDEN}, CACHE)

    q = quantize_film(model)
    print(f"trained: H={m12.HIDDEN}, qat val loss {model.final_loss:.4f} "
          f"(M12.2 {M12_2_INVENTED_PER_LINE}, M12.3 {M12_3_INVENTED_PER_LINE}, "
          f"M12.4 {M12_4_INVENTED_PER_LINE} inv/line)")

    corpus_vocab = m12.build_corpus_vocab(full_text)
    groups = {"selena": selena_pairs, "guard": guard_pairs, "cast": cast_pairs,
              "shadewrath": shadewrath_pairs, "korrath": korrath_pairs,
              "elowen": princess_pairs}

    print(f"\ncoherence probe WITH FiLM conditioning, D:/M: STRIPPED FROM PREFIX "
          f"({LINES_PER_GROUP} lines/group, shipped sampler settings "
          f"inv_t_q8={m12.INV_T_Q8} top_k={m12.TOP_K}, seed={PROBE_SEED:#x}):")
    results = run_film_probe(q, vocab, groups, corpus_vocab, n_desc,
                             desc_by_value, mood_by_value)
    print(f"\n{'group':<12}{'lines':>7}{'inv/line':>10}")
    for name, r in results.items():
        print(f"{name:<12}{r['lines']:>7}{r['invented_per_line']:>10.2f}")

    all_rate = results["ALL"]["invented_per_line"]
    print(f"\nALL: {all_rate:.2f} invented/line (M12.2 {M12_2_INVENTED_PER_LINE}, "
          f"M12.3 {M12_3_INVENTED_PER_LINE}, M12.4 {M12_4_INVENTED_PER_LINE}, "
          f"M12.1 gate <= {M12_1_GATE})")
    if all_rate < M12_4_INVENTED_PER_LINE:
        print("Beats M12.4 -- FiLM's per-step re-injection helps over one-shot columns.")
    else:
        print("Does NOT beat M12.4 -- FiLM did not improve on the input-concat ablation.")
    if all_rate <= M12_1_GATE:
        print("CLEARS the M12.1 coherence gate -- proceed to blob + kernel work.")
        sys.exit(0)
    else:
        print("Does not clear the M12.1 gate -- do not proceed to blob/kernel work.")
        sys.exit(1)


if __name__ == "__main__":
    main()
