#!/usr/bin/env python3
"""M10 Data Science Review: long-horizon consistency eval for Shadewrath
(docs/milestones/m10.md, "New eval category: long-horizon consistency").

Nothing tested before M10 covers a character across MANY separate
encounters spread over a long session -- Selena's M7 eval is continuous
within one scene, archetype instances don't have memory to be
inconsistent about. Shadewrath is the first character where this
actually matters: contradicting an earlier taunt, forgetting an
established score, or resetting emotional tone between encounters would
all read as "not alive" even if any single line sounds fine.

This is a SCRIPTED synthetic-session eval, not an automated pass/fail
gate -- N simulated encounters with known intervening events, checked by
HUMAN EYEBALL for continuity (m10.md's Data Science Review is explicit
that this is a new axis, not a bigger version of the per-condition
eyeball check M4/M7 already do). The trust-tier progression IS the
"known intervening events": tier 0 -> 1 -> 2 simulates the player
returning across separate encounters, same mapping DialogueDemo.cpp's
D-pad already uses (relationshipForTrustTier()).

Requires the model already trained by make_m10_blob.py (loads its cache,
trainer/.m10_model.pt -- run that first if it doesn't exist).
Run: uv run python eval_shadewrath_long_horizon.py   (from trainer/)
"""
import random
from pathlib import Path

import torch

from ngpt_trainer import shadewrath_corpus as swc
from ngpt_trainer.cast_corpus import CHARACTERS  # noqa: F401  (import-time sanity)
from ngpt_trainer.model import CharGRU
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled
from ngpt_trainer.vocab import Vocab

CACHE = Path(__file__).resolve().parent / ".m10_model.pt"
SAMPLE_SEED = 0xC0FFEE
INV_T_Q8 = 384
TOP_K = 5

# A scripted "session" -- one player, several separate encounters, with
# a narrative reason for the trust tier and mood at each step (this is
# the "known intervening events" the DSR asks for). Not exhaustive; this
# is a spot-check session, not a combinatorial sweep.
SESSION = [
    ("Encounter 1: first meeting, player is cautious", 0, "worried", "greeting", "none"),
    ("Encounter 1: player fights past him, takes damage", 0, "sassy", "damage-taken", "took_damage"),
    ("Encounter 1: player retreats", 0, "cheerful", "farewell", "none"),
    ("Encounter 2 (later): player returns, more confident now", 1, "sassy", "greeting", "returned_from_trip"),
    ("Encounter 2: a quiet moment mid-fight, he studies the player", 1, "tender", "quiet-moment", "none"),
    ("Encounter 2: player is encouraged after a near-loss", 1, "worried", "encouragement", "player_failed"),
    ("Encounter 3 (much later): player has proven themselves", 2, "tender", "greeting", "returned_from_trip"),
    ("Encounter 3: the real offer surfaces", 2, "cheerful", "farewell", "none"),
]


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(
            f"{CACHE} not found -- run make_m10_blob.py first to train the model "
            "this eval loads."
        )
    d = torch.load(CACHE, weights_only=True)

    # Rebuild the same vocab make_m10_blob.py used -- must match exactly
    # or ids won't line up. Cheapest correct way: reconstruct from the
    # same corpus assembly (small/fast, no training).
    from make_m10_blob import (KORRATH_PER_COMBO, PER_COMBO,
                                GUARD_PER_COMBO, SEED, SHADEWRATH_PER_COMBO)
    from ngpt_trainer import cast_corpus as cc
    from ngpt_trainer import guard_corpus as gc
    from ngpt_trainer import korrath_corpus as kc
    from ngpt_trainer import selena_corpus as sc

    full_text = (sc.corpus_text(seed=SEED, per_combo=PER_COMBO)
                + "".join(p + r for p, r in sc.generate_thin_identity_pairs(seed=1000))
                + "".join(p + r for p, r in gc.generate_pairs(seed=SEED, per_combo=GUARD_PER_COMBO))
                + cc.corpus_text(seed=SEED)
                + swc.corpus_text(seed=SEED, per_combo=SHADEWRATH_PER_COMBO)
                + kc.corpus_text(seed=SEED, per_combo=KORRATH_PER_COMBO))
    vocab = Vocab.from_text(full_text)

    model = CharGRU(vocab_size=len(vocab), hidden=d["hidden"])
    model.load_state_dict(d["state"])
    model.eval()
    q = quantize(model)

    print(f"Loaded cached M10 model (val loss {d['val_loss']:.4f}, H={d['hidden']})\n")
    print("=" * 78)
    print("SHADEWRATH -- scripted long-horizon session")
    print("Human-eyeball review target: does he stay recognizably the same")
    print("character across encounters, does trust-tier progression read as")
    print("a real arc (menace -> testing -> the offer), not random noise?")
    print("=" * 78)

    rng = random.Random(0xF00DFACE)
    for label, tier, mood, context, event in SESSION:
        prompt = swc.prompt_for(tier, mood, context, event)
        seed = SAMPLE_SEED ^ rng.randrange(0, 0xFFFFFF)
        got = generate_sampled(q, vocab, prompt, seed=seed,
                               inv_t_q8=INV_T_Q8, top_k=TOP_K)
        print(f"\n[{label}]")
        print(f"  {prompt}")
        print(f"  -> {got}")

    print("\n" + "=" * 78)
    print("Review checklist (fill in by hand after reading the above):")
    print("  [ ] Tone escalates with trust tier (menace -> grudging respect ->")
    print("      the alliance offer), not flat or randomly ordered")
    print("  [ ] No line contradicts an earlier one in ways that break")
    print("      character (e.g. claiming ignorance of something tier 1")
    print("      already revealed)")
    print("  [ ] Tier 2 output actually gestures at the offer (bible's")
    print("      desire field), not generic menace repeated at higher tier")
    print("  [ ] Coherence (word-level fluency) holds across the session, not")
    print("      just in the first encounter")


if __name__ == "__main__":
    main()
