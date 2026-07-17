"""M9 corpus assembly: turns generate_m9_corpus_llm.py's raw generated
lines into (prompt, response) training pairs via npc_service.prompt_fields()
-- the same "generator produces raw lines, a corpus module turns them into
the frozen prompt schema" separation M7/M8 used (selena_corpus.py/
guard_corpus.py wrapping hand-authored templates; here wrapping LLM output
instead, per docs/milestones/m8.1.md's flagged methodology shift).

Combo-level holdout (docs/milestones/m9.md's Data Science Review,
"the capacity-dilution hypothesis needs its own falsification test"):
generate_m9_corpus_llm.py already never dispatches the withheld
(occupation, descriptor) pairs, but this module re-asserts that at load
time rather than trusting the generator silently got it right --
holdout_pairs() is the single source of truth the capacity-check step
(make_m9_blob.py) probes against.
"""
import json
from pathlib import Path

from ngpt_trainer.npc_service import personality_descriptor, prompt_fields

REPO = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO / "m9_corpus_llm.json"
HOLDOUT_PATH = REPO / "m9_corpus_holdout.json"


def load_raw() -> list[dict]:
    return json.loads(CORPUS_PATH.read_text())


def holdout_pairs() -> list[tuple[str, str]]:
    return [tuple(p) for p in json.loads(HOLDOUT_PATH.read_text())]


def combo_key(entry: dict) -> tuple:
    """(occupation, descriptor) -- the axis M9's generalization claim is
    actually about; mood/context/tier stay fully covered in training."""
    return (entry["persona"]["occupation"],
            personality_descriptor(entry["persona"]["traits"]))


def generate_pairs(raw: list[dict] | None = None) -> list[tuple[str, str]]:
    """(prompt, response) pairs via prompt_fields() -- every entry in the
    raw corpus. Combo-level holdout is enforced at generation time
    (generate_m9_corpus_llm.py never dispatches those personas), so this
    is a straight conversion, not a filter -- see assert_no_holdout_leak()
    for the explicit safety-net check."""
    raw = raw if raw is not None else load_raw()
    pairs = []
    for entry in raw:
        prompt = prompt_fields(entry["persona"], _relationship_state(entry["tier"]),
                                entry["mood"], entry["context"])
        pairs.append((prompt, entry["line"]))
    return pairs


def assert_no_holdout_leak(raw: list[dict] | None = None) -> None:
    """Safety net: fails loudly if any generated line's (occupation,
    descriptor) combo matches a held-out pair -- the capacity-check gate
    is meaningless if held-out combos leaked into training."""
    raw = raw if raw is not None else load_raw()
    held = set(holdout_pairs())
    leaked = [e for e in raw if combo_key(e) in held]
    if leaked:
        combos = sorted({combo_key(e) for e in leaked})
        raise AssertionError(
            f"{len(leaked)} corpus lines leaked from {len(combos)} held-out "
            f"combos: {combos} -- capacity-check gate would be invalid")


# Relationship state axes only matter here insofar as they map to the
# tier prompt_fields() emits (R:<tier>) -- reconstruct any state whose
# closeness lands in the requested tier's bucket rather than threading
# the original random_relationship_state() draw through the JSON (it
# wasn't stored per-line, only the resolved tier was).
_TIER_MIDPOINT = {
    "stranger": 0.10, "acquaintance": 0.30, "neutral": 0.50,
    "friend": 0.70, "close_friend": 0.90, "best_friend": 0.975,
}


def _relationship_state(tier: str) -> dict:
    v = _TIER_MIDPOINT[tier]
    return {"familiarity": v, "affection": v, "trust": v, "respect": v, "fear": 0.0}


def combo_text(raw: list[dict] | None = None) -> str:
    """Every character the vocab must cover -- prompts + responses."""
    return "".join(p + r for p, r in generate_pairs(raw))
