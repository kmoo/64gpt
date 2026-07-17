#!/usr/bin/env python3
"""M9 LLM-judge evaluation pipeline (docs/milestones/m9.md section 5):
replaces the *manual* part of M7/M8's "gates pass numerically != output
is coherent" lesson, not the underlying principle -- a local model scores
generated corpus lines on a small rubric (grammatical coherence, matches
the stated personality/voice cues) at the trainer/host level, no ROM
boot required. The N64 boot test (SELFTEST PASS, XCHK) stays reserved
for hardware bit-exactness only, per that section's own note.

Two measurements, deliberately kept separate:
  1. Coherence + voice-match: judged by a local LLM (batched dispatches,
     not per-line -- there's no "voice collapse" risk for a scoring task
     the way there is for generation, so batching is safe and much
     cheaper than M8's isolated-dispatch discipline for corpus text).
  2. Near-duplicate detection *within* each persona's line set: done
     programmatically via character-trigram Jaccard distance
     (ngpt_trainer.divergence, the same metric M7/M8's own acceptance
     gates use) -- more reliable than asking an LLM to eyeball
     duplication across a whole batch in one prompt.

Explicitly NOT a trusted gate yet: this doc's own Data Science Review
requires a human spot-check against the LLM's scores before trusting it
at scale ("same skepticism M7/M8 applied to numeric gates"). This script
prints a spot-check sample (spanning the full score range, not just the
top) for that read -- by a human, or here, as an explicitly-labeled
first pass by Claude, since Luke asked for multiple model perspectives
("use you, mistral, or haiku... does this make sense") on this corpus.
Not a substitute for Luke's own read before this pipeline is trusted for
real acceptance decisions.

Run: uv run python llm_judge.py   (from trainer/, after
generate_m9_corpus_llm.py has produced trainer/m9_corpus_llm.json)
"""
import json
import random
import re
import subprocess
from pathlib import Path

from ngpt_trainer.divergence import jaccard_distance
from ngpt_trainer.npc_service import age_gender_token, personality_descriptor

REPO = Path(__file__).resolve().parent
CORPUS_PATH = REPO / "m9_corpus_llm.json"
REPORT_PATH = REPO / "m9_llm_judge_report.json"
OPENCODER = str(Path.home() / "bin" / "opencoder")

BATCH_SIZE = 10
JUDGE_SAMPLE_SEED = 0x5EED
JUDGE_SAMPLE_SIZE = 150   # cap dispatch cost; corpus is 1000+ lines
DUP_THRESHOLD = 0.3       # jaccard_distance below this = near-duplicate
LOW_SCORE_THRESHOLD = 3   # out of 5


def persona_key(persona: dict) -> tuple:
    return (persona["occupation"], persona["age"], persona["gender"],
            tuple(sorted(persona["traits"].items())))


def persona_label(persona: dict) -> str:
    person = age_gender_token(persona["age"], persona["gender"])
    descriptor = personality_descriptor(persona["traits"])
    return f"a {descriptor} {persona['age']}-year-old {person}, working as a {persona['occupation']}"


def find_near_duplicates(corpus: list[dict]) -> list[dict]:
    """Groups lines by persona, flags within-persona pairs below
    DUP_THRESHOLD -- the actual test of M8's "isolated dispatch avoids
    voice collapse into repeated phrasing" discipline, applied to M9's
    persona-batched generation."""
    by_persona: dict[tuple, list[dict]] = {}
    for entry in corpus:
        by_persona.setdefault(persona_key(entry["persona"]), []).append(entry)

    flagged = []
    for key, entries in by_persona.items():
        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                d = jaccard_distance(a["line"], b["line"])
                if d < DUP_THRESHOLD:
                    flagged.append({
                        "persona": persona_label(a["persona"]),
                        "distance": round(d, 3),
                        "line_a": a["line"], "line_b": b["line"],
                    })
    return flagged


def build_judge_prompt(batch: list[dict]) -> str:
    lines = [
        "You are scoring video-game NPC dialogue lines for quality. For "
        "each numbered item below, you're given the character's intended "
        "voice and the line they supposedly said. Rate two things 1-5 "
        "(5=best): COHERENCE (is it grammatical, sensible English?) and "
        "VOICE (does it actually match the stated personality/occupation, "
        "not generic filler?). Output ONLY a JSON array, one object per "
        "item, like: [{\"i\":1,\"coherence\":4,\"voice\":3}, ...] -- no "
        "other text before or after.",
        "",
    ]
    for i, entry in enumerate(batch, 1):
        label = persona_label(entry["persona"])
        lines.append(f'{i}. Character: {label}. Mood: {entry["mood"]}. '
                      f'Relationship: {entry["tier"]}. Line: "{entry["line"]}"')
    return "\n".join(lines)


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def dispatch_judge(batch: list[dict]) -> list[dict] | None:
    prompt = build_judge_prompt(batch)
    result = subprocess.run([OPENCODER, prompt], capture_output=True,
                             text=True, timeout=600)
    if result.returncode != 0:
        return None
    m = _JSON_ARRAY_RE.search(result.stdout)
    if not m:
        return None
    try:
        scores = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if len(scores) != len(batch):
        return None
    return scores


def main():
    corpus = json.loads(CORPUS_PATH.read_text())
    print(f"loaded {len(corpus)} corpus lines")

    dups = find_near_duplicates(corpus)
    print(f"\nnear-duplicate check (within-persona, jaccard < {DUP_THRESHOLD}): "
          f"{len(dups)} flagged pairs")
    for d in dups[:10]:
        print(f"  [{d['distance']}] {d['persona']}")
        print(f"    a: {d['line_a']}")
        print(f"    b: {d['line_b']}")

    rng = random.Random(JUDGE_SAMPLE_SEED)
    sample = corpus if len(corpus) <= JUDGE_SAMPLE_SIZE else rng.sample(corpus, JUDGE_SAMPLE_SIZE)
    print(f"\ndispatching LLM-judge over {len(sample)} sampled lines "
          f"({(len(sample) + BATCH_SIZE - 1) // BATCH_SIZE} batched dispatches)...")

    judged = []
    for start in range(0, len(sample), BATCH_SIZE):
        batch = sample[start:start + BATCH_SIZE]
        scores = dispatch_judge(batch)
        if scores is None:
            print(f"  batch {start // BATCH_SIZE + 1}: judge dispatch failed, skipping")
            continue
        for entry, score in zip(batch, scores):
            judged.append({**entry, "coherence": score.get("coherence"),
                           "voice": score.get("voice")})
        print(f"  batch {start // BATCH_SIZE + 1}: {len(scores)} scored")

    valid = [j for j in judged if isinstance(j.get("coherence"), (int, float))
             and isinstance(j.get("voice"), (int, float))]
    if valid:
        mean_coh = sum(j["coherence"] for j in valid) / len(valid)
        mean_voice = sum(j["voice"] for j in valid) / len(valid)
        low = [j for j in valid
               if j["coherence"] < LOW_SCORE_THRESHOLD or j["voice"] < LOW_SCORE_THRESHOLD]
        print(f"\njudged {len(valid)}/{len(sample)} lines successfully")
        print(f"  mean coherence: {mean_coh:.2f}/5")
        print(f"  mean voice match: {mean_voice:.2f}/5")
        print(f"  low scorers (<{LOW_SCORE_THRESHOLD} on either axis): "
              f"{len(low)} ({100 * len(low) / len(valid):.1f}%)")
    else:
        print("\nno valid judge scores collected -- pipeline needs debugging "
              "before it can be trusted at all, let alone spot-checked")

    report = {
        "corpus_size": len(corpus),
        "near_duplicates": dups,
        "judged": judged,
        "summary": {
            "mean_coherence": mean_coh if valid else None,
            "mean_voice": mean_voice if valid else None,
            "low_scorer_count": len(low) if valid else None,
            "judged_count": len(valid),
            "sample_size": len(sample),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT_PATH}")

    if valid:
        by_score = sorted(valid, key=lambda j: j["coherence"] + j["voice"])
        spot_check = by_score[:5] + by_score[len(by_score) // 2 - 2:len(by_score) // 2 + 3] + by_score[-5:]
        print(f"\n--- spot-check sample ({len(spot_check)} lines spanning the "
              f"score range, for a human/independent read before trusting "
              f"this pipeline -- NOT yet validated against Luke's own "
              f"judgment, per m9.md's Data Science Review) ---")
        for j in spot_check:
            label = persona_label(j["persona"])
            print(f"  [coh={j['coherence']} voice={j['voice']}] {label}")
            print(f"    mood={j['mood']} tier={j['tier']}: {j['line']}")


if __name__ == "__main__":
    main()
