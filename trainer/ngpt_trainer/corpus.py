"""The M3 hand-written corpus: 12 prompt→response pairs.

Single source of truth — the demo's on-screen string tables, the
training data, and the committed goldens all derive from these. Prompt
format is frozen in docs/milestones/m3.md: `NPC=<n> MOOD=<m> EV=<e>|`.
Uppercase only (the debug font has no lowercase glyphs).
"""

NPCS = ("GUARD", "MERCHANT", "WIZARD")
MOODS = ("ANGRY", "CALM")
EVENTS = ("THEFT", "FESTIVAL")

_RESPONSES = {
    ("GUARD", "ANGRY", "THEFT"): "HALT! THE THIEF HANGS AT DAWN!",
    ("GUARD", "ANGRY", "FESTIVAL"): "KEEP YOUR REVELS OFF MY GATE!",
    ("GUARD", "CALM", "THEFT"): "REPORT THE THEFT AT THE KEEP.",
    ("GUARD", "CALM", "FESTIVAL"): "ENJOY THE FEAST. NO TROUBLE.",
    ("MERCHANT", "ANGRY", "THEFT"): "A THIEF TOOK MY BEST WARES!",
    ("MERCHANT", "ANGRY", "FESTIVAL"): "CROWDS EVERYWHERE, NOBODY BUYS!",
    ("MERCHANT", "CALM", "THEFT"): "LOCK YOUR COIN AWAY, FRIEND.",
    ("MERCHANT", "CALM", "FESTIVAL"): "FINE GOODS, FESTIVAL PRICES!",
    ("WIZARD", "ANGRY", "THEFT"): "WHO DARES STEAL FROM A WIZARD?",
    ("WIZARD", "ANGRY", "FESTIVAL"): "THIS NOISE RUINS MY STUDIES!",
    ("WIZARD", "CALM", "THEFT"): "THE STARS WILL NAME YOUR THIEF.",
    ("WIZARD", "CALM", "FESTIVAL"): "EVEN MAGIC RESTS ON FEAST DAYS.",
}


def prompt_for(npc: str, mood: str, event: str) -> str:
    return f"NPC={npc} MOOD={mood} EV={event}|"


def pairs() -> list[tuple[str, str]]:
    """All 12 (prompt, response) pairs, in deterministic order."""
    return [
        (prompt_for(n, m, e), _RESPONSES[(n, m, e)])
        for n in NPCS for m in MOODS for e in EVENTS
    ]


def corpus_text() -> str:
    """Every character the vocab must cover (prompts + responses)."""
    return "".join(p + r for p, r in pairs())
