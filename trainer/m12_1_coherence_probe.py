#!/usr/bin/env python3
"""M12.1 phase 1 -- the shipped-config coherence probe.

Generates ~8 sampled lines per character group through the QUANTIZED
model at the SHIPPED sampler settings (the exact configuration a player
sees), and scores what val loss cannot see:

  invented/line   -- invented words per line (make_m12_blob.py's own
                     invented_word_count against the full corpus vocab)
  EOS-before-cap  -- fraction of lines that reached EOS before max_len
                     (derailed generations lose the EOS signal and
                     run to the cap; see docs/ideas-coherence-rescue-plan.md)
  verbatim        -- fraction of lines that are a verbatim corpus
                     response (memorization-vs-recombination indicator)

Reported PER CHARACTER, because the corpus's 75:1 pair skew is exactly
the signal corpus-wide averages erased for three milestones.

Usage:  uv run python m12_1_coherence_probe.py [model_cache.pt]
        (default: .m12_model.pt -- the M12 H=1024 baseline)

The probe is import-safe: run_probe() is the reusable entry point that
make_m12_1_blob.py's acceptance gates will call.
"""
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

from ngpt_trainer import cast_corpus as cc
from ngpt_trainer import guard_corpus as gc
from ngpt_trainer import korrath_corpus as kc
from ngpt_trainer import princess_corpus as pc
from ngpt_trainer import selena_corpus as sc
from ngpt_trainer import shadewrath_corpus as swc
from ngpt_trainer.model import CharGRU
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled
from ngpt_trainer.vocab import Vocab

import make_m12_1_blob as m12

PROBE_SEED = 0xC0FFEE
LINES_PER_GROUP = 8
MAX_LEN = m12.MAX_GOLDEN_LEN          # 300, same as the goldens
INV_T_Q8 = m12.INV_T_Q8               # shipped sampler settings
TOP_K = m12.TOP_K


def build_corpus():
    """Corpus groups keyed by character, plus vocab artifacts -- the
    exact M12.1 shipped, rebalanced mix (lore bank off). NOTE: this
    function imports make_m12_1_blob (fixed 2026-07-24 -- it originally
    imported make_m12_blob, the pre-rebalance M12 script, a phase-1
    leftover from before make_m12_1_blob.py existed; the real M12.1
    build gate never used this helper, since make_m12_1_blob.main()
    builds its own groups/vocab directly, so nothing already shipped
    was affected -- but any standalone use of this function, like the
    phase-3 min-p sweep, silently ran against the WRONG 75:1-skewed
    corpus until this fix)."""
    (selena_pairs, guard_pairs, cast_pairs, shadewrath_pairs,
     korrath_pairs, princess_pairs) = m12.build_all_pairs()
    full_text = (sc.corpus_text(seed=m12.SEED, per_combo=m12.PER_COMBO)
                 + "".join(p + r for p, r in guard_pairs)
                 + cc.corpus_text(seed=m12.SEED, per_combo=m12.CAST_PER_COMBO)
                 + swc.corpus_text(seed=m12.SEED, per_combo=m12.SHADEWRATH_PER_COMBO,
                                   lore_bank_enabled=m12.LORE_BANK_ENABLED)
                 + kc.corpus_text(seed=m12.SEED, per_combo=m12.KORRATH_PER_COMBO,
                                  lore_bank_enabled=m12.LORE_BANK_ENABLED)
                 + pc.corpus_text(seed=m12.SEED, per_combo=m12.PRINCESS_PER_COMBO,
                                  lore_bank_enabled=m12.LORE_BANK_ENABLED))
    groups = {"selena": selena_pairs, "guard": guard_pairs,
              "cast": cast_pairs, "shadewrath": shadewrath_pairs,
              "korrath": korrath_pairs, "elowen": princess_pairs}
    vocab = Vocab.from_text(full_text)
    corpus_vocab = m12.build_corpus_vocab(full_text)
    corpus_lines = {r for pairs in groups.values() for _, r in pairs}
    return groups, vocab, corpus_vocab, corpus_lines


def load_quantized(cache_path: Path, vocab):
    d = torch.load(cache_path, weights_only=True)
    model = CharGRU(vocab_size=len(vocab), hidden=d["hidden"])
    model.load_state_dict(d["state"])
    model.eval()
    return quantize(model), d


def run_probe(q, vocab, groups, corpus_vocab, corpus_lines,
              lines_per_group: int = LINES_PER_GROUP,
              seed: int = PROBE_SEED, verbose: bool = True,
              minp_shift: int = 0) -> dict:
    """Returns {group: {"pairs", "lines", "invented_per_line",
    "eos_before_cap", "verbatim"}} plus an "ALL" rollup.

    minp_shift (M12.1 phase 3): 0 = disabled (phase-1/2 behavior,
    unchanged); > 0 applies the integer min-p gate to every draw --
    passed straight through to generate_sampled/sample_from_logits.

    Per-group salt uses zlib.crc32, NOT Python's builtin hash(): str
    hashing is randomized per-process (PYTHONHASHSEED) unless disabled,
    so hash(name) silently drew a DIFFERENT sample of probe lines every
    fresh process -- found 2026-07-24 while sweeping phase 3's min-p
    shift, when two supposedly-identical re-runs disagreed. crc32 is a
    fixed, stable function of the string, restoring this probe's
    seed-in/result-out determinism (the same principle every sampler
    seed in this project already depends on)."""
    results = {}
    tot_inv = tot_eos = tot_verb = tot_lines = 0
    for name, pairs in groups.items():
        rng = random.Random(seed ^ (zlib.crc32(name.encode()) & 0xFFFF))
        prompts = [p for p, _ in rng.sample(pairs, min(lines_per_group, len(pairs)))]
        inv = eos = verb = 0
        for i, prompt in enumerate(prompts):
            got = generate_sampled(q, vocab, prompt, seed=seed + i,
                                   inv_t_q8=INV_T_Q8, top_k=TOP_K, max_len=MAX_LEN,
                                   minp_shift=minp_shift)
            n = m12.invented_word_count(got, corpus_vocab)
            inv += n
            eos += int(len(got) < MAX_LEN)
            verb += int(got in corpus_lines)
            if verbose:
                print(f"  [{name}] [{n} inv{' CAP' if len(got) >= MAX_LEN else ''}] {got}")
        k = len(prompts)
        results[name] = {"pairs": len(pairs), "lines": k,
                         "invented_per_line": inv / k,
                         "eos_before_cap": eos / k, "verbatim": verb / k}
        tot_inv += inv; tot_eos += eos; tot_verb += verb; tot_lines += k
    results["ALL"] = {"pairs": sum(len(p) for p in groups.values()),
                      "lines": tot_lines,
                      "invented_per_line": tot_inv / tot_lines,
                      "eos_before_cap": tot_eos / tot_lines,
                      "verbatim": tot_verb / tot_lines}
    return results


def print_table(results: dict) -> None:
    print(f"\n{'group':<12}{'pairs':>8}{'lines':>7}{'inv/line':>10}"
          f"{'eos<cap':>9}{'verbatim':>10}")
    for name, r in results.items():
        print(f"{name:<12}{r['pairs']:>8}{r['lines']:>7}"
              f"{r['invented_per_line']:>10.2f}{r['eos_before_cap']:>9.0%}"
              f"{r['verbatim']:>10.0%}")


def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent / ".m12_model.pt"
    groups, vocab, corpus_vocab, corpus_lines = build_corpus()
    q, d = load_quantized(cache, vocab)
    print(f"probe: {cache.name} H={q.H} V={q.V} "
          f"cached val loss {d['val_loss']:.4f} -- sampler inv_t_q8={INV_T_Q8} "
          f"top_k={TOP_K} max_len={MAX_LEN} seed={PROBE_SEED:#x}\n")
    results = run_probe(q, vocab, groups, corpus_vocab, corpus_lines)
    print_table(results)


if __name__ == "__main__":
    main()
