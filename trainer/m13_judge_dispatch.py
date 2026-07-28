"""M13 mechanism 4: judges m13_judge_validation_candidates.json's fragments
on tone/voice-plausibility, one slot (character+mood) per dispatch --
tonight's own finding was that batching multiple slots collapses scores
to a uniform per-group value, so this deliberately stays fine-grained.
Uses the SAME opencoder subprocess pattern as llm_judge.py's
dispatch_judge(), not a new client. Run: uv run python3 m13_judge_dispatch.py
"""
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent
CANDIDATES_PATH = REPO / "m13_judge_validation_candidates.json"
SCORES_PATH = REPO / "m13_judge_scores.json"
OPENCODER = str(Path.home() / "bin" / "opencoder")

VOICE = {
    "BRAM": ("BRAM (guard): by-the-book, terse, essentially never jokes, "
             "blunt but not cruel."),
    "EDRIC": ("EDRIC-A (guard): careful, methodical, double-checks "
              "everything, dry understatement."),
    "KORRATH": ("KORRATH (bound knight guarding a captive princess): "
                "formal, weary, tragic rather than cruel, exhausted by "
                "eternal duty."),
}
MOOD_NOTE = {
    "WORRIED": "alert/cautious, appropriate to their base voice.",
    "CHEERFUL": ("still guarded, none of these characters is ever "
                 "genuinely carefree -- CHEERFUL just means less bleak "
                 "than their WORRIED register."),
}

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def build_prompt(character: str, mood: str, lines: list[str]) -> str:
    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))
    return (
        f"You are rating short spoken-line candidates for a video-game "
        f"character on how well they match the character's voice/mood. "
        f"{VOICE[character]} {mood} mood = {MOOD_NOTE[mood]}\n\n"
        f"Rate each 1-5 on TONE_FIT (5=perfectly matches, 1=wrong "
        f"register/generic filler). Do NOT give every line the same "
        f"score -- spread across the range where real quality "
        f"differences exist. Output ONLY JSON: "
        f'[{{"i":1,"tone_fit":N}}, ...]\n\n{numbered}'
    )


def dispatch(prompt: str, max_tokens: int = 256) -> list[dict] | None:
    env_prefix = ["env", f"OPENCODER_MAX_TOKENS={max_tokens}"]
    result = subprocess.run(env_prefix + [OPENCODER, prompt],
                             capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return None
    m = _JSON_ARRAY_RE.search(result.stdout)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    candidates = json.loads(CANDIDATES_PATH.read_text())
    scores = {}
    for slot, lines in candidates.items():
        character, mood = slot.rsplit("_", 1)
        prompt = build_prompt(character, mood, lines)
        result = dispatch(prompt)
        if result is None or len(result) != len(lines):
            got = 0 if result is None else len(result)
            print(f"  [{slot}] dispatch failed or incomplete "
                  f"({got}/{len(lines)}), retrying once")
            result = dispatch(prompt)
        if result is None:
            print(f"  [{slot}] FAILED after retry, skipping")
            continue
        slot_scores = [None] * len(lines)
        for entry in result:
            i = entry.get("i")
            if isinstance(i, int) and 1 <= i <= len(lines):
                slot_scores[i - 1] = entry.get("tone_fit")
        scores[slot] = slot_scores
        n_scored = sum(1 for s in slot_scores if s is not None)
        print(f"  [{slot}] {n_scored}/{len(lines)} scored: {slot_scores}")

    SCORES_PATH.write_text(json.dumps(scores, indent=2))
    print(f"\nwrote {SCORES_PATH}")


if __name__ == "__main__":
    main()
