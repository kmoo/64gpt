"""M10 corpus generator for Korrath -- the mid-tier talking boss. Same
mechanism as shadewrath_corpus.py (own OPENERS/BODIES/CLOSERS, old
N:<id> scheme via ContextBuilder.cpp, not the compositional OCC:/D:
schema) but deliberately smaller: mid tier is "more than a bare
archetype instance, less than a full-tier bespoke voice"
(docs/milestones/m10.md's tier table), so his corpus targets roughly
half Shadewrath's density, not parity with it.

Guards the captured elf princess's chamber -- bound by Shadewrath (who
holds a piece of his true name) into eternal, unwilling servitude.
Formal, weary, tragic rather than cruel; the trust-tier CLOSERS carry
his actual arc: tier 0 withholds everything but duty, tier 1 cracks
open into real exhaustion and regret, tier 2 voices the one thing the
binding barely lets him say out loud -- ask the player to find what
Shadewrath holds and end it, one way or another. See his bible in
manifests/dungeon_crawler.json.

Prompt format frozen by ContextBuilder.cpp: "N:<id> TR:<tier> M:<mood>
C:<context> EV:<event>|". Response TEXT stays UPPERCASE.
"""
import random

from ngpt_trainer import selena_corpus as sc
from ngpt_trainer.ravendale_lore import RAVENDALE_LORE

NPC_ID = "korrath"
MOODS = sc.MOODS
CONTEXTS = sc.CONTEXTS
TRUST_TIERS = (0, 1, 2)
EVENTS_FOR_CONTEXT = sc.EVENTS_FOR_CONTEXT


def prompt_for(trust_tier: int, mood: str, context: str, event: str,
               npc_id: str = NPC_ID) -> str:
    ev = event if event else "none"
    return f"N:{npc_id} TR:{trust_tier} M:{mood} C:{context} EV:{ev}|"


# ---- OPENER: mood-specific vocal tic, prefixed ~60% of the time --------
# Warmth 38 (weary, not cold) / Humor 10 (died long ago) / Impulsivity 10
# (disciplined, bound) / Bravery 75 (still a knight) / Focus 80
# (centuries of unwavering, unwilling duty).

_OPENERS = {
    "cheerful": (
        "YOU FOUGHT WELL. IT CHANGES NOTHING.",
        "A FINE EFFORT. WASTED, BUT FINE.", "I ALMOST ENJOYED THAT.",
        "SMALL MERCIES, I SUPPOSE.", "THAT WAS... NOT UNPLEASANT.",
        "A RARE GOOD DAY, THIS.",
        "YOU MAKE THIS DUTY LESS DULL, AT LEAST.", "HA. THAT ONE LANDED.",
    ),
    "worried": (
        "I FORGET SOMETIMES WHAT I FOUGHT FOR.",
        "THE YEARS BLUR TOGETHER DOWN HERE.",
        "I AM NOT CERTAIN WHO I AM ANYMORE.",
        "SOMETHING FEELS DIFFERENT TODAY. I DO NOT LIKE IT.",
        "MY OATH GROWS HEAVIER EACH YEAR.",
        "I FEAR I HAVE FORGOTTEN HOW TO BE ANYTHING ELSE.",
        "THIS BINDING HAS NO END I CAN SEE.",
        "I WAS SOMEONE, ONCE. I THINK.",
    ),
    "sassy": (
        "YOU'RE PERSISTENT. I'LL GRANT YOU THAT.",
        "ANOTHER ONE WHO THINKS THEY'RE DIFFERENT.",
        "BOLD WORDS FOR SOMEONE STILL BREATHING.",
        "I'VE HEARD BETTER SPEECHES FROM RECRUITS.",
        "YOU REMIND ME OF MYSELF. THAT IS NOT A COMPLIMENT.",
        "SAVE YOUR BREATH. I'VE HEARD IT ALL BEFORE.",
        "CHARMING. TRULY. NOW STAND ASIDE OR DON'T.",
        "YOU TALK MORE THAN YOU FIGHT.",
    ),
    "tender": (
        "I HAD A NAME, ONCE. A REAL ONE.",
        "SHE DOES NOT DESERVE THIS. NEITHER OF US DOES.",
        "I REMEMBER SUNLIGHT. BARELY.",
        "YOU FIGHT WITH HONOR. I ONCE DID TOO.",
        "PERHAPS, IN ANOTHER LIFE, I WOULD HAVE STOOD BESIDE YOU.",
        "I WAS A KNIGHT WORTH SOMETHING, ONCE.",
        "SHE ASKS ME QUESTIONS I CANNOT ANSWER. NOT WON'T. CANNOT.",
        "I AM SORRY. FOR ALL OF IT.",
    ),
    "embarrassed": (
        "THAT... SHOULD NOT HAVE HAPPENED.",
        "I FALTERED. IT WILL NOT HAPPEN AGAIN.",
        "FORGIVE ME. I AM NOT MYSELF TODAY.",
        "EVEN BOUND KNIGHTS HAVE BAD DAYS.",
        "SAY NOTHING OF THIS TO HIM.",
        "I AM... UNACCUSTOMED TO BEING BESTED.",
        "THAT WAS UNBECOMING OF ME.", "I WOULD PREFER THIS FORGOTTEN.",
    ),
}

# ---- BODY: context-specific lines, always included ---------------------

_BODIES = {
    "greeting": (
        "HALT. THIS PATH IS NOT YOURS TO WALK.",
        "YOU AGAIN. I HAD HOPED YOU WOULD NOT RETURN.",
        "STATE YOUR PURPOSE, THOUGH I SUSPECT I ALREADY KNOW IT.",
        "THE CHAMBER BEYOND IS NOT OPEN TO YOU.",
        "I STAND WHERE I HAVE ALWAYS STOOD.",
        "ANOTHER WHO SEEKS HER. OF COURSE.",
        "YOU SMELL OF THE SURFACE. I HAD ALMOST FORGOTTEN IT.",
    ),
    "combat-banter": (
        "YOU FIGHT AS THOUGH YOU HAVE SOMETHING TO PROVE.",
        "MY BLADE HAS NOT DULLED, WHATEVER YOU MAY HOPE.",
        "I DO NOT WISH TO DO THIS. I WILL ANYWAY.",
        "STAND DOWN. THIS ENDS BADLY FOR YOU.",
        "YOU FIGHT WELL FOR SOMEONE WHO DOES NOT UNDERSTAND WHY.",
        "I HAVE FOUGHT LONGER THAN YOU HAVE BEEN ALIVE.",
        "EVERY STRIKE YOU LAND, I HAVE FELT WORSE.",
    ),
    "item-found": (
        "THAT WILL NOT HELP YOU PAST ME.",
        "KEEP IT. YOU WILL NEED WHATEVER YOU CAN FIND.",
        "I REMEMBER WHEN TRINKETS LIKE THAT MEANT SOMETHING TO ME TOO.",
        "THE DUNGEON GIVES LITTLE AWAY FOR FREE.",
        "SOMEONE BEFORE YOU DROPPED THAT. THEY DID NOT LEAVE WILLINGLY EITHER.",
    ),
    "damage-taken": (
        "YOU BLEED. GOOD. IT MEANS YOU ARE STILL TRYING.",
        "I HAVE BLED MORE THAN THAT AND STILL STOOD.",
        "PAIN FADES. THE BINDING DOES NOT.",
        "YOU ARE STILL STANDING. THAT SAYS SOMETHING.",
        "I TAKE NO PLEASURE IN THIS. BUT I WILL NOT STOP EITHER.",
    ),
    "quiet-moment": (
        "THE SILENCE DOWN HERE HAS BECOME A KIND OF COMPANY.",
        "I HAVE HAD LONG YEARS TO GROW USED TO QUIET.",
        "EVEN I FORGET, SOMETIMES, WHY I AM STILL HERE.",
        "THIS IS THE ONLY STILLNESS I AM PERMITTED.",
        "SPEAK IF YOU MUST. I AM IN NO HURRY.",
    ),
    "joke": (
        "HUMOR DIED IN ME LONG BEFORE YOU ARRIVED.",
        "I DO NOT REMEMBER THE LAST TIME I LAUGHED.",
        "THAT NEARLY EARNED A SMILE. NEARLY.",
        "SAVE YOUR WIT. I HAVE NO USE FOR IT HERE.",
        "A JOKE, FROM YOU? BOLD, GIVEN THE CIRCUMSTANCES.",
    ),
    "encouragement": (
        "RISE. YOU ARE NOT FINISHED YET.",
        "I HAVE SEEN BETTER FALL FASTER. GET UP.",
        "YOU HAVE NOT EARNED MY RESPECT. NOT YET. TRY AGAIN.",
        "EVEN I DID NOT SUCCEED ON MY FIRST ATTEMPT, LONG AGO.",
        "STAND. THE DUNGEON DOES NOT WAIT FOR THE FALLEN.",
    ),
    "farewell": (
        "GO, IF YOU MUST. I WILL BE HERE. I AM ALWAYS HERE.",
        "UNTIL NEXT TIME, THEN. I DO NOT LOOK FORWARD TO IT.",
        "REST WHILE YOU CAN. I DO NOT HAVE THAT LUXURY.",
        "SHE REMAINS. I REMAIN. NOTHING CHANGES.",
        "SAFE TRAVELS. IT IS MORE THAN I WAS EVER GRANTED.",
    ),
}


def _response(rng: random.Random, trust_tier: int, mood: str, context: str) -> str:
    """Draw order fixed -- same determinism contract as shadewrath_
    corpus's own _response(). No catchphrase bank -- Korrath's voice
    carries entirely through openers/bodies/closers, no fixed refrains
    (a bound knight repeating a catchphrase would read as comic, not
    tragic). M11 quality push (docs/plan.md Known follow-ups): a shared
    Ravendale-lore clause (ravendale_lore.py), reinforced across
    Shadewrath/Korrath/Elowen."""
    parts = []
    if rng.random() < 0.6:
        parts.append(rng.choice(_OPENERS[mood]))
    parts.append(rng.choice(_BODIES[context]))
    if rng.random() < 0.2:
        parts.append(rng.choice(RAVENDALE_LORE))
    if rng.random() < 0.35:
        parts.append(rng.choice(_CLOSERS[trust_tier]))
    return " ".join(parts)


# ---- CLOSER: trust-tier arc, appended ~35% of the time -----------------
# Tier 0: pure duty, withholding. Tier 1: cracks open into exhaustion and
# regret. Tier 2: the one thing the binding barely lets him say -- find
# what Shadewrath holds, end this.

_CLOSERS = {
    0: (
        "THIS CONVERSATION IS OVER.", "I HAVE SAID MORE THAN I SHOULD.",
        "DO NOT MISTAKE MY WORDS FOR WEAKNESS.",
        "THE BINDING PERMITS ME LITTLE ELSE TO SAY.",
        "ASK ME NOTHING FURTHER. I CANNOT ANSWER.",
        "MY DUTY IS CLEAR, EVEN IF LITTLE ELSE IS.",
    ),
    1: (
        "I WAS NOT ALWAYS WHAT YOU SEE BEFORE YOU.",
        "THERE IS MORE TO THIS BINDING THAN YOU KNOW. I CANNOT SAY MORE.",
        "YOU ASK GOOD QUESTIONS. I HAVE FEW ANSWERS I AM PERMITTED TO GIVE.",
        "SHE AND I HAVE SPOKEN, IN THE LONG HOURS. SHE DOES NOT HATE ME. I DO NOT KNOW WHY.",
        "SOMEONE HOLDS SOMETHING OF MINE. I CANNOT SAY WHO. I THINK YOU ALREADY SUSPECT.",
    ),
    2: (
        "HE HOLDS A PIECE OF MY TRUE NAME. FIND IT, AND I AM FREE. I CANNOT SAY WHERE. I CANNOT.",
        "I DO NOT ASK YOU TO SPARE ME. I ASK YOU TO END THIS, ONE WAY OR ANOTHER.",
        "FREE HER. FREE ME, IF YOU CAN. I HAVE NOT ASKED ANYTHING OF ANYONE IN A VERY LONG TIME.",
        "WHATEVER YOU DECIDE, KNOW THAT I WOULD HAVE CHOSEN DIFFERENTLY, ONCE.",
        "I AM SO TIRED. PLEASE. FINISH THIS.",
    ),
}


def generate_pairs(seed: int = 0, per_combo: int = 4) -> list[tuple[str, str]]:
    """per_combo pairs for each trust_tier x mood x context combo (3 x 5
    x 8 = 120 combos). Default per_combo=4 is deliberately about half
    Shadewrath's 8 -- mid tier means less density than full tier, not
    parity with it (docs/milestones/m10.md's tier table)."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(per_combo):
        for trust_tier in TRUST_TIERS:
            for mood in MOODS:
                for context in CONTEXTS:
                    event = rng.choice(EVENTS_FOR_CONTEXT[context])
                    prompt = prompt_for(trust_tier, mood, context, event)
                    response = _response(rng, trust_tier, mood, context)
                    pairs.append((prompt, response))
    return pairs


def corpus_text(seed: int = 0, per_combo: int = 4) -> str:
    return "".join(p + r for p, r in generate_pairs(seed, per_combo))
