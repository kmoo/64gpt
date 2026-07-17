#!/usr/bin/env python3
"""M9 compositional-conditioning corpus generator: dispatches one local-LLM
call per PERSONA (not per line, not batched across personas -- M8's own
finding was that an all-characters-in-one-prompt dispatch causes voice
collapse; docs/milestones/m9.md section 4 applies the same discipline one
level more granular, per feature combo rather than per character id).

**Combo-level holdout, same discipline as M8's Selena holdout.** The
whole premise of M9 (docs/milestones/m9.md's Data Science Review, "the
capacity-dilution hypothesis needs its own falsification test") is that
compositional features let the model generalize to combinations it never
saw in training -- unlike M8's opaque per-id tags, which by construction
could NEVER generalize (an untrained id is just unknown vocabulary). That
claim is untestable unless some (occupation, descriptor) pairs are
withheld from corpus generation entirely. ~15-20% of the achievable
(occupation, descriptor) pairs are reserved here (HELD_OUT_PAIRS,
resolved by real sampling below, not hand-picked) and never dispatched --
recorded in m9_corpus_holdout.json for the later capacity/LLM-judge
evaluation to probe directly: does "gruff merchant" (never trained) come
out coherent, or garbled the way an unknown opaque id always would?

Personas are drawn from REAL random_npc_profile() sampling across many
seeds (not fixed hand-picked trait dicts) specifically so the corpus
exercises the actual personality_descriptor() vocabulary breadth, not
just one or two blend labels repeated everywhere.

Each dispatch asks for a numbered list of short N64-style dialogue lines,
one per (mood, context, relationship-tier) scenario, for ONE persona.
Output is parsed, uppercased (the debug font has no lowercase glyphs --
selena_corpus.py/guard_corpus.py's same constraint), and turned into
(prompt, response) pairs via npc_service.prompt_fields().

Local-only, open-weight model (Mistral-7B via ~/bin/opencoder), never
Claude/proprietary APIs for corpus text -- per m9.md section 4.

Run: uv run python generate_m9_corpus_llm.py   (from trainer/)
Writes trainer/m9_corpus_llm.json (raw generated lines) and
trainer/m9_corpus_holdout.json (the withheld combo list) -- both
git-ignored, regenerable, consumed by m9_corpus.py.
"""
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from ngpt_trainer.npc_service import (
    OCCUPATIONS,
    age_gender_token,
    personality_descriptor,
    random_npc_profile,
)

REPO = Path(__file__).resolve().parent
OUT_PATH = REPO / "m9_corpus_llm.json"
HOLDOUT_PATH = REPO / "m9_corpus_holdout.json"

# 3 representative contexts (same discipline as guard_corpus.py's
# GUARD_CONTEXTS -- a subset, not the full 8, keeps dispatch size sane)
# and 3 representative moods spanning distinct emotional registers.
CONTEXTS = ("greeting", "combat-banter", "quiet-moment")
MOODS = ("cheerful", "worried", "sassy")
ALL_TIERS = ("stranger", "acquaintance", "neutral", "friend", "close_friend", "best_friend")

SAMPLE_SEED = 0xB16B00B5
N_CANDIDATES = 400          # real sampling pool -- big enough to hit most
                             # (occupation, descriptor) pairs at least once
HOLDOUT_FRACTION = 0.18     # ~1 in 5-6 achieved pairs withheld entirely


def _build_persona_pool() -> dict[tuple[str, str], dict]:
    """Sample N_CANDIDATES real profiles via random_npc_profile and keep
    one representative persona per (occupation, descriptor) pair reached
    -- real sampled coverage, not hand-picked trait dicts."""
    pool: dict[tuple[str, str], dict] = {}
    rng = random.Random(SAMPLE_SEED)
    seed = rng.randrange(1, 2**32)
    for _ in range(N_CANDIDATES):
        seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
        profile = random_npc_profile(seed)
        descriptor = personality_descriptor(profile["traits"])
        key = (profile["occupation"], descriptor)
        if key not in pool:
            pool[key] = profile
    return pool


def _choose_holdout(pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
    rng = random.Random(SAMPLE_SEED ^ 0xC0FFEE)
    k = max(1, round(len(pairs) * HOLDOUT_FRACTION))
    return set(rng.sample(pairs, k))


def _persona(profile: dict, tier_offset: int) -> dict:
    tiers = tuple(ALL_TIERS[(tier_offset + i) % len(ALL_TIERS)] for i in range(3))
    return dict(profile, tiers=tiers)


def build_personas():
    pool = _build_persona_pool()
    all_pairs = sorted(pool.keys())
    holdout_pairs = _choose_holdout(all_pairs)
    train_pairs = [p for p in all_pairs if p not in holdout_pairs]

    print(f"sampled {len(pool)} distinct (occupation, descriptor) pairs from "
          f"{N_CANDIDATES} candidate profiles")
    print(f"  occupations covered: {len({occ for occ, _ in all_pairs})}/{len(OCCUPATIONS)}")
    print(f"  descriptors covered: {sorted({d for _, d in all_pairs})}")
    print(f"  holding out {len(holdout_pairs)} pairs entirely: {sorted(holdout_pairs)}")

    personas = [_persona(pool[key], i) for i, key in enumerate(train_pairs)]
    return personas, sorted(holdout_pairs)


def persona_label(p: dict) -> str:
    person = age_gender_token(p["age"], p["gender"])
    descriptor = personality_descriptor(p["traits"])
    return f"a {descriptor} {p['age']}-year-old {person}, working as a {p['occupation']}"


def build_prompt(p: dict) -> tuple[str, list[tuple[str, str, str]]]:
    """Returns (llm_prompt, scenario_list) where scenario_list[i] =
    (mood, context, tier) matching numbered item i+1 in the dispatch."""
    scenarios = [(mood, ctx, tier)
                 for tier in p["tiers"] for ctx in CONTEXTS for mood in MOODS]
    lines = [
        f"You write terse dialogue lines for an N64 fantasy game NPC. "
        f"Character: {persona_label(p)}. Rules: ALL CAPS text only (no "
        f"lowercase letters exist in the game font), each line under 90 "
        f"characters, in-character voice matching the personality and "
        f"relationship closeness given, no narration or stage directions, "
        f"just the spoken line. Output ONLY a numbered list, one line per "
        f"number, nothing else -- no preamble, no explanation.",
        "",
    ]
    for i, (mood, ctx, tier) in enumerate(scenarios, 1):
        lines.append(f"{i}. mood={mood} scene={ctx} relationship={tier}")
    return "\n".join(lines), scenarios


_LINE_RE = re.compile(r"^\s*(\d+)[.)]\s*(.+?)\s*$")


def parse_response(text: str, n_expected: int) -> dict[int, str]:
    out = {}
    for raw in text.splitlines():
        m = _LINE_RE.match(raw)
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n_expected:
            line = m.group(2).strip().strip('"').upper()
            line = "".join(c for c in line if c.isascii())
            if 1 <= len(line) <= 90:
                out[idx] = line
    return out


OPENCODER = str(Path.home() / "bin" / "opencoder")


def dispatch(prompt: str) -> str:
    result = subprocess.run(
        [OPENCODER, prompt],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  DISPATCH FAILED: {result.stderr[-500:]}", file=sys.stderr)
        return ""
    return result.stdout


def main():
    personas, holdout_pairs = build_personas()
    HOLDOUT_PATH.write_text(json.dumps(holdout_pairs, indent=2))
    print(f"wrote {len(holdout_pairs)} holdout pairs to {HOLDOUT_PATH}\n")

    all_results = []
    for i, persona in enumerate(personas):
        label = persona_label(persona)
        print(f"[{i+1}/{len(personas)}] dispatching: {label} (tiers={persona['tiers']})")
        llm_prompt, scenarios = build_prompt(persona)
        raw = dispatch(llm_prompt)
        parsed = parse_response(raw, len(scenarios))
        got = len(parsed)
        print(f"  got {got}/{len(scenarios)} valid lines")
        for idx, line in parsed.items():
            mood, ctx, tier = scenarios[idx - 1]
            all_results.append({
                "persona": persona,
                "mood": mood,
                "context": ctx,
                "tier": tier,
                "line": line,
            })
        # Write incrementally so a crash/interrupt doesn't lose prior work.
        OUT_PATH.write_text(json.dumps(all_results, indent=2))

    print(f"\nwrote {len(all_results)} total lines to {OUT_PATH}")


if __name__ == "__main__":
    main()
