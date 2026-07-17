#!/usr/bin/env python3
"""M7 gating spike: does text-priming through the shared H=128 model
actually carry an identity signal, or does a small conditioned GRU learn
to ignore an identity tag the way M4's WIZARD/CALM/THEFT miss suggests it
might?

Two throwaway identities (ID_A, ID_B), no character bible, a few hundred
generated lines each, trained on the exact existing pipeline
(ngpt_trainer.model.train_corpus, H=128) — deliberately decoupled from
H=256 and from Selena's real corpus so this tests exactly one risk.

Metric: conditioning-ablation divergence via character-trigram Jaccard
distance between greedy-decoded completions. Two comparisons:
  - identity-swap: fix mood+context, swap only the identity tag
  - mood-swap:     fix identity+context, swap only the mood tag (the
                    in-mechanism baseline — mood is known to already work,
                    M3/M4 both condition on it successfully)
Gate: mean identity-swap divergence >= mean mood-swap divergence. See
docs/milestones/m7.md "Spike" section and docs/spikes/identity-conditioning.md
for the write-up this script produces.

Throwaway: no blob export, no ROM, no quantization — this only has to
answer the yes/no research question. Run: uv run python spike_identity.py
(from trainer/, after `uv sync`).
"""
import itertools
import random
from pathlib import Path

import torch

from ngpt_trainer.divergence import cross_set_divergence, jaccard_distance
from ngpt_trainer.model import CharGRU, one_hot, train_corpus
from ngpt_trainer.vocab import Vocab

SEED = 0
HIDDEN = 128
IDENTITIES = ("ID_A", "ID_B")
MOODS = ("CHEERFUL", "WORRIED", "SASSY")
CONTEXTS = ("GREETING", "ITEM_FOUND", "FAREWELL")

# Distinct vocabulary per identity so the signal the model must learn is
# genuinely "which identity is this," not just "which combo is this" —
# same discipline as M4's per-NPC openers/bodies (ngpt_trainer/corpus_gen.py).
_OPENERS = {
    ("ID_A", "CHEERFUL"): ("HEY THERE!", "OH GOOD, YOU'RE HERE!", "HI HI HI!",
                           "GREAT TO SEE YOU!"),
    ("ID_A", "WORRIED"): ("UM, HELLO...", "OH NO, IS EVERYTHING OKAY?",
                          "WAIT, WHAT HAPPENED?", "I'M A LITTLE NERVOUS."),
    ("ID_A", "SASSY"): ("WELL WELL WELL.", "LOOK WHO SHOWED UP.",
                        "TOOK YOU LONG ENOUGH.", "OH, IT'S YOU."),
    ("ID_B", "CHEERFUL"): ("SALUTATIONS, FRIEND!", "WHAT A GLORIOUS DAY!",
                           "MY SPIRITS ARE HIGH!", "SPLENDID TIMING!"),
    ("ID_B", "WORRIED"): ("HOLD ON A MOMENT.", "SOMETHING FEELS AMISS.",
                          "I MUST CONFESS UNEASE.", "THIS GIVES ME PAUSE."),
    ("ID_B", "SASSY"): ("HOW UNEXPECTED.", "DO GO ON.", "CHARMING, TRULY.",
                        "IF YOU INSIST."),
}

_BODIES = {
    ("ID_A", "GREETING"): ("LET'S GO ON AN ADVENTURE!",
                           "WHAT ARE WE DOING TODAY?",
                           "I'VE BEEN WAITING FOR YOU ALL DAY!"),
    ("ID_A", "ITEM_FOUND"): ("OOH, SHINY! CAN I HOLD IT?",
                             "YOU FOUND SOMETHING GOOD!",
                             "GRAB IT QUICK BEFORE SOMEONE ELSE DOES!"),
    ("ID_A", "FAREWELL"): ("SEE YOU SOON, OKAY?", "DON'T BE A STRANGER!",
                          "COME BACK SAFE!"),
    ("ID_B", "GREETING"): ("SHALL WE PROCEED WITH THE DAY'S BUSINESS?",
                          "STATE YOUR INTENTIONS, TRAVELER.",
                          "I TRUST YOUR JOURNEY WAS UNEVENTFUL."),
    ("ID_B", "ITEM_FOUND"): ("A NOTABLE DISCOVERY, INDEED.",
                             "EXAMINE IT CAREFULLY BEFORE PROCEEDING.",
                             "ONE WONDERS WHO LOST SUCH A THING."),
    ("ID_B", "FAREWELL"): ("UNTIL OUR PATHS CROSS AGAIN.",
                          "FAREWELL, AND TAKE CARE.",
                          "I BID YOU A SAFE DEPARTURE."),
}


def generate_sampled_prompted(model: CharGRU, vocab: Vocab, prompt: str, seed: int,
                              temperature: float = 0.7, max_len: int = 256) -> str:
    """Prime on EOS+prompt, then ancestrally sample the completion (float
    domain, throwaway spike — no quantization needed). Greedy decode is the
    wrong tool for this corpus: multiple valid responses exist per combo
    (rng.choice at corpus-gen time, same as M4's corpus_gen.py), so greedy
    always collapses onto a single global mode regardless of conditioning
    and can look like "identity ignored" even when the learned distribution
    correctly differs by identity. M4 hit the same issue and built a real
    sampler (ngpt_trainer/ref_impl.py generate_sampled) for its acceptance
    goldens for exactly this reason."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        logits, h = model(one_hot([vocab.eos_id] + vocab.encode(prompt), len(vocab)))
        probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
        current = int(torch.multinomial(probs, 1, generator=gen).item())
        for _ in range(max_len):
            if current == vocab.eos_id:
                break
            out.append(vocab.decode([current]))
            logits, h = model(one_hot([current], len(vocab)), h)
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
            current = int(torch.multinomial(probs, 1, generator=gen).item())
    return "".join(out)


def prompt_for(identity: str, mood: str, context: str) -> str:
    return f"N={identity} M={mood} C={context}|"


def _response(rng: random.Random, identity: str, mood: str, context: str) -> str:
    opener = rng.choice(_OPENERS[(identity, mood)])
    body = rng.choice(_BODIES[(identity, context)])
    return f"{opener} {body}"


def generate_pairs(seed: int = 0, per_combo: int = 30) -> list[tuple[str, str]]:
    """per_combo pairs for each identity x mood x context combo,
    interleaved so any prefix already covers every condition."""
    rng = random.Random(seed)
    combos = [(i, m, c) for i in IDENTITIES for m in MOODS for c in CONTEXTS]
    pairs = []
    for _ in range(per_combo):
        for identity, mood, context in combos:
            pairs.append((prompt_for(identity, mood, context),
                          _response(rng, identity, mood, context)))
    return pairs


def corpus_text(seed: int = 0, per_combo: int = 30) -> str:
    return "".join(p + r for p, r in generate_pairs(seed, per_combo))


SAMPLES_PER_CONDITION = 5
TEMPERATURE = 0.7


def draw_and_diverge(model, vocab, prompt_a: str, prompt_b: str, base_seed: int):
    """K samples per side (robust to one string landing on either side's
    mode by chance), divergence via the shared trigram-Jaccard metric."""
    samples_a = [generate_sampled_prompted(model, vocab, prompt_a, base_seed + i, TEMPERATURE)
                 for i in range(SAMPLES_PER_CONDITION)]
    samples_b = [generate_sampled_prompted(model, vocab, prompt_b, base_seed + 1000 + i, TEMPERATURE)
                 for i in range(SAMPLES_PER_CONDITION)]
    return cross_set_divergence(samples_a, samples_b), samples_a, samples_b


def main() -> None:
    pairs = generate_pairs(seed=SEED)
    vocab = Vocab.from_text(corpus_text(seed=SEED))
    print(f"spike corpus: {len(pairs)} pairs, {len(corpus_text(seed=SEED))} chars, "
          f"{len(vocab)} vocab (incl. EOS)")

    model = train_corpus(pairs, vocab, hidden=HIDDEN, seed=SEED,
                         max_epochs=200, patience=20)
    print(f"trained: val loss {model.final_loss:.4f}")
    print(f"sampling: {SAMPLES_PER_CONDITION} draws/condition, T={TEMPERATURE}")

    # identity-swap: fix mood+context, swap identity
    identity_divs = []
    identity_samples = []
    for mood, context in itertools.product(MOODS, CONTEXTS):
        d, sa, sb = draw_and_diverge(model, vocab, prompt_for("ID_A", mood, context),
                                     prompt_for("ID_B", mood, context), base_seed=0xC0FFEE)
        identity_divs.append(d)
        identity_samples.append((mood, context, sa, sb, d))
        print(f"  [identity swap] M={mood} C={context}: div={d:.3f}")
        print(f"    ID_A samples: {sa}")
        print(f"    ID_B samples: {sb}")

    # mood-swap: fix identity+context, swap mood (pairwise across the 3 moods)
    mood_divs = []
    mood_samples = []
    for identity, context in itertools.product(IDENTITIES, CONTEXTS):
        for m1, m2 in itertools.combinations(MOODS, 2):
            d, sa, sb = draw_and_diverge(model, vocab, prompt_for(identity, m1, context),
                                         prompt_for(identity, m2, context), base_seed=0xBEEF)
            mood_divs.append(d)
            mood_samples.append((identity, context, m1, m2, sa, sb, d))
            print(f"  [mood swap] N={identity} C={context}: {m1} vs {m2}: div={d:.3f}")

    mean_identity = sum(identity_divs) / len(identity_divs)
    mean_mood = sum(mood_divs) / len(mood_divs)
    gate = mean_identity >= mean_mood
    print()
    print(f"mean identity-swap divergence: {mean_identity:.4f}")
    print(f"mean mood-swap divergence:     {mean_mood:.4f}")
    print(f"GATE ({'CLEARS' if gate else 'DOES NOT CLEAR'}): "
          f"identity-swap {'>=' if gate else '<'} mood-swap")

    report = Path(__file__).resolve().parent.parent / "docs" / "spikes" / "identity-conditioning.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spike: identity-conditioning divergence (M7 gate)",
        "",
        f"Corpus: {len(pairs)} pairs / {len(corpus_text(seed=SEED))} chars, "
        f"2 identities x {len(MOODS)} moods x {len(CONTEXTS)} contexts, "
        f"H={HIDDEN}, seed={SEED}. Each combo's response is drawn from a "
        f"small opener x body mixture at corpus-gen time (like M4), so "
        f"evaluation samples ancestrally (T={TEMPERATURE}, "
        f"{SAMPLES_PER_CONDITION} draws/condition) rather than greedy-"
        f"decoding — greedy collapses onto one global mode regardless of "
        f"conditioning on a multi-modal corpus like this one; M4 hit the "
        f"same issue and built a real sampler for its acceptance goldens.",
        "",
        f"Val loss: {model.final_loss:.4f}",
        "",
        f"Mean identity-swap divergence (n={len(identity_divs)}): {mean_identity:.4f}",
        f"Mean mood-swap divergence (n={len(mood_divs)}): {mean_mood:.4f}",
        "",
        f"**GATE: {'CLEARS' if gate else 'DOES NOT CLEAR'}** — "
        f"identity-swap divergence {'is >= ' if gate else 'is BELOW '} mood-swap divergence.",
        "",
        "## Identity-swap samples (mood+context fixed, identity swapped)",
        "",
    ]
    for mood, context, sa, sb, d in identity_samples:
        lines.append(f"- M={mood} C={context} (div={d:.3f})")
        lines.append(f"  - `ID_A` draws: {sa}")
        lines.append(f"  - `ID_B` draws: {sb}")
    lines += ["", "## Mood-swap samples (identity+context fixed, mood swapped)", ""]
    for identity, context, m1, m2, sa, sb, d in mood_samples:
        lines.append(f"- N={identity} C={context}: `{m1}` vs `{m2}` (div={d:.3f})")
        lines.append(f"  - `{m1}` draws: {sa}")
        lines.append(f"  - `{m2}` draws: {sb}")
    report.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
