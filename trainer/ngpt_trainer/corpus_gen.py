"""M4 corpus generator — deterministic template grammar.

Where M3's corpus.py hand-writes 12 lines, this module *generates*
thousands: per-NPC voice templates with filler slots, modulated by MOOD
(angry: short, exclamatory, no closer; calm: measured, with a closing
phrase) and EVENT (theft/festival vocabulary). The point is variety with
a rigid format, so the ~100K-param model generalizes across fillers
instead of memorizing whole lines.

Deterministic: one random.Random(seed) drives every choice in a fixed
draw order, so the same seed is byte-identical everywhere. stdlib only.
Uppercase printable ASCII throughout (the N64 debug font has no
lowercase). Prompts come from corpus.prompt_for — the single source of
the prompt protocol since M3.
"""
import random

from ngpt_trainer.corpus import NPCS, MOODS, EVENTS, prompt_for

# ---- grammar tables ----------------------------------------------------
# response = OPENER[npc,mood] + " " + BODY[npc,event]{a,b} (+ " " +
# CLOSER[npc] when CALM). ANGRY turns the body's final "." into "!".

_OPENERS = {
    ("GUARD", "ANGRY"): ("HALT!", "OI, YOU THERE!", "STOP RIGHT THERE!",
                         "NOT ON MY WATCH!", "YOU AGAIN!", "ENOUGH!"),
    ("GUARD", "CALM"): ("STEADY NOW.", "EASY, FRIEND.", "ALL IN ORDER.",
                        "AT EASE.", "MIND YOURSELF.", "CARRY ON."),
    ("MERCHANT", "ANGRY"): ("OUTRAGEOUS!", "MY COIN!", "THIEVES EVERYWHERE!",
                            "HANDS OFF!", "THE NERVE!", "RUINED, I TELL YOU!"),
    ("MERCHANT", "CALM"): ("AH, A CUSTOMER.", "WELCOME, FRIEND.",
                           "GOOD DAY TO YOU.", "STEP CLOSER.",
                           "A FINE HOUR.", "BROWSE FREELY."),
    ("WIZARD", "ANGRY"): ("FOOL!", "BY THE VOID!", "THE STARS RAGE!",
                          "INSOLENCE!", "BEWARE, MORTAL!", "DARK OMENS!"),
    ("WIZARD", "CALM"): ("HMM, CURIOUS.", "THE STARS ALIGN.",
                         "PEACE, TRAVELER.", "AS FORETOLD.", "PATIENCE.",
                         "THE AETHER HUMS."),
}

_BODIES = {
    ("GUARD", "THEFT"): (
        "A {a} WAS STOLEN AND I WILL {b}.",
        "THE {a} IS GONE, SO WE {b} AT ONCE.",
        "WHOEVER TOOK THE {a}, I SHALL {b}.",
        "REPORT ANY {a} MISSING AND I WILL {b}.",
        "GUARD THE {a} WELL OR I MUST {b}.",
        "THE THIEF WITH THE {a} CANNOT HIDE, I {b}.",
    ),
    ("GUARD", "FESTIVAL"): (
        "THE {a} DRAWS A CROWD, SO I {b}.",
        "KEEP THE {a} ORDERLY WHILE I {b}.",
        "ENJOY THE {a}, BUT I STILL {b}.",
        "NO TROUBLE AT THE {a} WHILE I {b}.",
        "THE {a} IS SAFE BECAUSE I {b}.",
        "STAY CLEAR OF THE {a} GATES AS I {b}.",
    ),
    ("MERCHANT", "THEFT"): (
        "MY {a} WAS TAKEN, YET I {b}.",
        "SOMEONE LIFTED A {a}, SO NOW I {b}.",
        "THE {a} STALL WAS ROBBED AND I {b}.",
        "NO {a} FOR SALE UNTIL I {b}.",
        "A THIEF PRICED MY {a} AT NOTHING, SO I {b}.",
        "GUARD MY {a} WHILE I {b}.",
    ),
    ("MERCHANT", "FESTIVAL"): (
        "FESTIVAL CROWDS LOVE {a}, SO I OFFER {b}.",
        "BUY {a} TODAY AND GET {b}.",
        "THE FAIR BRINGS BUYERS FOR {a} AND {b}.",
        "MY {a} SELL FAST WHEN {b} IS PROMISED.",
        "COME FOR THE {a}, STAY FOR {b}.",
        "EVERY {a} COMES WITH {b} THIS WEEK.",
    ),
    ("WIZARD", "THEFT"): (
        "MY {a} WAS STOLEN AND {b} FOLLOWS.",
        "WHO DARES TAKE THE {a}? {b} AWAITS.",
        "THE {a} IS MISSING, THUS {b} STIRS.",
        "RETURN THE {a} OR FACE {b}.",
        "WITHOUT THE {a}, {b} GATHERS.",
        "THE THIEF OF THE {a} SHALL KNOW {b}.",
    ),
    ("WIZARD", "FESTIVAL"): (
        "FOR THE FESTIVAL I CONJURE {a} BENEATH {b}.",
        "TONIGHT {a} SHALL DANCE ACROSS {b}.",
        "THE CROWD GASPS AS {a} LIGHTS {b}.",
        "WATCH {a} BLOOM UNDER {b}.",
        "MY GIFT TO THE FAIR IS {a} IN {b}.",
        "{a} SHALL CROWN {b} AT DUSK.",
    ),
}

_SLOT_A = {
    ("GUARD", "THEFT"): ("PURSE", "LOCKET", "DAGGER", "SIGNET RING",
                         "STRONGBOX", "SILVER CUP"),
    ("GUARD", "FESTIVAL"): ("PARADE", "BONFIRE", "LANTERN MARCH",
                            "GRAND FEAST", "MASQUERADE", "TOURNAMENT"),
    ("MERCHANT", "THEFT"): ("SILKS", "SPICES", "AMULETS", "FINE RUGS",
                            "PEARLS", "COPPER POTS"),
    ("MERCHANT", "FESTIVAL"): ("SILKS", "SPICES", "AMULETS", "FINE RUGS",
                               "PEARLS", "COPPER POTS"),
    ("WIZARD", "THEFT"): ("GRIMOIRE", "CRYSTAL ORB", "RUNESTONE",
                          "PHOENIX QUILL", "STAR CHART", "SILVER WAND"),
    ("WIZARD", "FESTIVAL"): ("SILVER FIREWORKS", "DANCING LIGHTS",
                             "A RAIN OF STARS", "GLOWING RUNES",
                             "EMERALD FLAMES", "SINGING SPARKS"),
}

_SLOT_B = {
    ("GUARD", "THEFT"): ("WATCH EVERY GATE", "SEARCH EVERY CART",
                         "QUESTION EVERYONE", "DOUBLE THE PATROL",
                         "LOCK THE SQUARE", "SOUND THE ALARM"),
    ("GUARD", "FESTIVAL"): ("HOLD THE LINE", "WATCH THE WALLS",
                            "GUARD THE GATE", "KEEP THE PEACE",
                            "PATROL TILL DAWN", "STAND MY POST"),
    ("MERCHANT", "THEFT"): ("COUNT MY COIN", "BAR THE SHUTTERS",
                            "HIRE A GUARD", "RAISE MY PRICES",
                            "CHECK THE LEDGER", "WATCH THE TILL"),
    ("MERCHANT", "FESTIVAL"): ("HALF PRICE", "A FREE RIBBON", "TWO FOR ONE",
                               "A LUCKY CHARM", "AN HONEST DEAL",
                               "A BRIGHT DISCOUNT"),
    ("WIZARD", "THEFT"): ("A DARK OMEN", "A COLD CURSE", "THE VOID'S GAZE",
                          "A STORM OF SPARKS", "AN ILL FATE",
                          "A WANING MOON"),
    ("WIZARD", "FESTIVAL"): ("THE NIGHT SKY", "A HARVEST MOON",
                             "THE OLD TOWER", "THE LANTERN GLOW",
                             "A VELVET DUSK", "THE FESTIVAL SQUARE"),
}

_CLOSERS = {  # CALM only — measured voices trail off politely
    "GUARD": ("STAY SAFE.", "WALK ON.", "GOOD DAY.", "KEEP TO THE LIGHT.",
              "REST EASY.", "MOVE ALONG NOW."),
    "MERCHANT": ("FAIR COIN, FAIR TRADE.", "COME AGAIN.", "A PLEASURE.",
                 "TAKE YOUR TIME.", "HONEST WORK.", "GOOD FORTUNE."),
    "WIZARD": ("SO THE STARS SAY.", "THE OMENS AGREE.", "FATE IS KIND.",
               "THE AETHER SETTLES.", "ALL IS FORETOLD.", "SEEK WISDOM."),
}


def _response(rng: random.Random, npc: str, mood: str, event: str) -> str:
    """One response; the draw order (opener, body, a, b, closer) is fixed
    — it IS the determinism contract."""
    opener = rng.choice(_OPENERS[(npc, mood)])
    body = rng.choice(_BODIES[(npc, event)]).format(
        a=rng.choice(_SLOT_A[(npc, event)]),
        b=rng.choice(_SLOT_B[(npc, event)]),
    )
    if mood == "ANGRY":
        if body.endswith("."):
            body = body[:-1] + "!"
        return f"{opener} {body}"
    return f"{opener} {body} {rng.choice(_CLOSERS[npc])}"


def generate_pairs(seed: int = 0, per_combo: int = 1200) -> list[tuple[str, str]]:
    """per_combo pairs for each of the 12 NPC x MOOD x EVENT combos,
    interleaved (the combo order cycles) so any prefix of the corpus
    already covers every condition — matters for train/val splits."""
    rng = random.Random(seed)
    combos = [(n, m, e) for n in NPCS for m in MOODS for e in EVENTS]
    pairs = []
    for _ in range(per_combo):
        for npc, mood, event in combos:
            pairs.append((prompt_for(npc, mood, event),
                          _response(rng, npc, mood, event)))
    return pairs


def corpus_text(seed: int = 0, per_combo: int = 1200) -> str:
    """Concatenated prompt+response stream — the vocab/charset source.
    No separator: the prompt's trailing '|' already delimits, exactly as
    the model sees it."""
    return "".join(p + r for p, r in generate_pairs(seed, per_combo))
