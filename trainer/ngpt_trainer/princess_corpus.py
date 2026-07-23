"""M11 corpus generator for Elowen -- the rescued elf princess of
Ravendale. Same mechanism as shadewrath_corpus.py/korrath_corpus.py (own
OPENERS/BODIES/CLOSERS) but mid tier like Korrath: deliberately about
half Shadewrath's density, not parity with it.

Held captive by Shadewrath -- abducted from Ravendale, kept in the
chamber Korrath guards. Her bible in manifests/dungeon_crawler.json is
the source of truth; this corpus is her voice. Hopeful, scared-but-
brave, grateful -- a clear tonal contrast to both her captor (cold,
patient, controlling) and her guard (weary, dutiful, tragic), so a
model that can hold all three coherently is real evidence it's learning
distinct voices, not copying content across characters.

Trust tier here tracks how far she's opened up to the player, same
interactive dial as Shadewrath/Korrath's own TR: field (D-pad up/down
in the shipped demo) -- tier 0: guarded, testing whether this is
another trick; tier 1: opens up about her captivity and Shadewrath's
plans; tier 2: the actual rescue moment, asking to be freed. Reaching
tier 2 for the first time is the real in-game "rescue" event
(game/src/user/SaveData.h's princessHighestTier, DialogueDemo.cpp's
gossip trigger) -- same mechanism Shadewrath/Korrath's own trust tiers
already use, not a new one invented just for her.

M11.1 (docs/milestones/m11.1.md Part 1): genericized onto NpcService's
compositional scheme -- occupation "noble" (already existed, fits
"captive royalty" cleanly, no new entry needed), species "elf" (her
bible's own "elf princess of Ravendale"). ContextBuilder/N:<id> is gone;
prompt_for() now wraps npc_service.prompt_fields(). Response TEXT stays
UPPERCASE.
"""
import random

from ngpt_trainer import selena_corpus as sc
from ngpt_trainer.npc_service import prompt_fields
from ngpt_trainer.ravendale_lore import RAVENDALE_LORE

MOODS = sc.MOODS
CONTEXTS = sc.CONTEXTS
TRUST_TIERS = (0, 1, 2)
EVENTS_FOR_CONTEXT = sc.EVENTS_FOR_CONTEXT

# Matches game/src/user/NPCDatabase.cpp's princess NPC exactly.
ELOWEN_PROFILE = {
    "occupation": "noble", "age": 24, "gender": "female",
    "species": "elf", "bond": "captive",
    "traits": {"warmth": 78, "humor": 45, "impulsivity": 55,
              "bravery": 60, "focus": 50},
}

# Matches DialogueDemo.cpp's relationshipForTrustTier() exactly, same
# rationale as shadewrath_corpus.py/korrath_corpus.py's own copies.
_TRUST_TIER_MIDPOINT = {0: 0.100, 1: 0.500, 2: 0.975}


def _relationship_state(trust_tier: int) -> dict:
    v = _TRUST_TIER_MIDPOINT[trust_tier]
    return {"familiarity": v, "affection": v, "trust": v, "respect": v, "fear": 0.0}


# AUD: -- a labeling pass over existing content (docs/milestones/m11.1.md
# Part 3): "tender"/"embarrassed" are where she drops her guarded public
# performance ("I DON'T EVEN KNOW YOUR NAME AND I TRUST YOU ALREADY,"
# "FORGIVE ME, I'M NOT USED TO BEING ASKED HOW I FEEL") -- her bible's
# PRIVATE register, same pattern as Shadewrath's own corpus.
_ALONE_MOODS = ("tender", "embarrassed")


def prompt_for(trust_tier: int, mood: str, context: str, event: str) -> str:
    audience = "alone" if mood in _ALONE_MOODS else "witnessed"
    return prompt_fields(ELOWEN_PROFILE, _relationship_state(trust_tier),
                         mood, context, audience, event)


# ---- OPENER: mood-specific vocal tic, prefixed ~60% of the time --------
# Warmth 78 (warm once she trusts you) / Humor 45 (dry, wry -- captivity
# didn't kill her wit) / Impulsivity 55 (quick to hope, quick to fear) /
# Bravery 60 (brave in small, real ways -- not a fighter, but not
# broken) / Focus 50.

_OPENERS = {
    "cheerful": (
        "YOU CAME BACK. I WASN'T SURE YOU WOULD.",
        "EVERY VISIT FEELS LIKE A SMALL VICTORY.",
        "I ALMOST FORGOT WHAT GOOD NEWS SOUNDED LIKE.",
        "SOMETHING IN ME LIGHTS UP WHEN THAT DOOR OPENS AND IT'S YOU.",
        "TODAY FEELS DIFFERENT. IN A GOOD WAY, I THINK.",
        "I'VE BEEN COUNTING THE DAYS. THIS ONE'S A GOOD ONE.",
        "YOU HAVE NO IDEA WHAT IT MEANS THAT YOU KEEP COMING BACK.",
        "I LET MYSELF HOPE TODAY. IT FELT DANGEROUS AND WONDERFUL.",
    ),
    "worried": (
        "HE WAS HERE EARLIER. I COULD FEEL IT.",
        "PLEASE BE CAREFUL. I'VE SEEN WHAT HE DOES TO THOSE WHO AREN'T.",
        "I DON'T LIKE HOW QUIET IT'S BEEN.",
        "WHAT IF THIS IS ANOTHER ONE OF HIS TRICKS?",
        "I'M SORRY. IT'S HARD NOT TO EXPECT THE WORST HERE.",
        "SOMETHING FEELS WRONG TODAY. I CAN'T EXPLAIN IT.",
        "I KEEP WAITING FOR THIS TO GO BADLY. IT USUALLY DOES.",
        "DON'T LINGER TOO LONG. HE NOTICES EVERYTHING EVENTUALLY.",
    ),
    "sassy": (
        "OH, SO NOW YOU SHOW UP.",
        "A PRINCESS LOCKED IN A CHAMBER STILL HAS STANDARDS, YOU KNOW.",
        "I'VE HAD WORSE COMPANY. BARELY.",
        "DON'T LOOK SO SURPRISED I CAN STILL BE CHARMING DOWN HERE.",
        "CAPTIVITY HASN'T TAKEN MY SENSE OF HUMOR. YET.",
        "I'D CURTSY, BUT THE CHAINS MAKE IT AWKWARD.",
        "YOU'RE LATE. I KEPT MYSELF ENTERTAINED IMAGINING WHY.",
        "IMPRESSIVE. YOU FOUND THE ONE ROOM HE DIDN'T WANT YOU IN.",
    ),
    "tender": (
        "I DON'T EVEN KNOW YOUR NAME AND I TRUST YOU ALREADY. IS THAT FOOLISH?",
        "NOBODY'S ASKED HOW I'M DOING IN SO LONG.",
        "I THINK ABOUT RAVENDALE MORE THAN I LET ON.",
        "YOU'RE THE FIRST KIND FACE I'VE SEEN IN WHAT FEELS LIKE YEARS.",
        "I USED TO THINK NO ONE WAS COMING. I'M GLAD I WAS WRONG.",
        "THANK YOU. FOR COMING BACK, EVEN WHEN YOU DIDN'T HAVE TO.",
        "I REMEMBER SUNLIGHT. I HOLD ONTO THAT MOST DAYS.",
        "YOU DIDN'T HAVE TO CARE. I NOTICED THAT YOU DO ANYWAY.",
    ),
    "embarrassed": (
        "I DIDN'T MEAN TO SAY THAT OUT LOUD.",
        "FORGIVE ME, I'M NOT USED TO BEING ASKED HOW I FEEL.",
        "THAT WAS MORE THAN I MEANT TO SHARE. IGNORE THAT LAST PART.",
        "I SOUND FOOLISH, HOPING THE WAY I DO.",
        "I'M NOT ALWAYS THIS -- WHATEVER THIS IS. IT'S BEEN A LONG WHILE.",
        "PLEASE DON'T THINK LESS OF ME FOR CRYING JUST NOW.",
        "I PROMISED MYSELF I WOULDN'T GET MY HOPES UP AGAIN.",
        "SORRY. IT'S JUST -- NO ONE ASKS ME THAT HERE.",
    ),
}

# ---- BODY: context-specific lines, always included ---------------------
# Not a fighter -- her "combat-banter" is encouragement from behind the
# chamber door, not her own fighting; "damage-taken" is HER reaction to
# the PLAYER being hurt, same convention shadewrath_corpus/korrath_corpus
# use for a non-combatant/observer character.

_BODIES = {
    "greeting": (
        "YOU FOUND YOUR WAY BACK DOWN HERE AGAIN.",
        "I HEAR FOOTSTEPS AND HOPE, EVERY TIME, THAT IT'S YOU.",
        "THE GUARD LET YOU THROUGH. THAT'S SOMETHING, AT LEAST.",
        "I'VE BEEN ALONE WITH MY THOUGHTS SINCE YOU LEFT.",
        "WELCOME BACK TO MY VERY SMALL WORLD.",
        "YOU LOOK TIRED. SIT, IF THIS PLACE WILL LET YOU.",
    ),
    "combat-banter": (
        "BE CAREFUL OUT THERE. I CAN HEAR MORE THAN YOU'D THINK FROM HERE.",
        "I WISH I COULD HELP. I'M NOT ALLOWED EVEN THAT.",
        "YOU FIGHT LIKE SOMEONE WHO HAS SOMETHING TO COME BACK FOR.",
        "I LISTEN FOR THE SOUNDS OF A FIGHT AND PRAY IT'S NOT YOU LOSING.",
        "WHATEVER'S OUT THERE, YOU'VE SURVIVED WORSE TO REACH ME.",
    ),
    "item-found": (
        "THAT LOOKS LIKE SOMETHING FROM RAVENDALE, ACTUALLY.",
        "KEEP IT SAFE. NOTHING DOWN HERE IS GIVEN FREELY.",
        "I RECOGNIZE THAT CRAFTSMANSHIP. MY PEOPLE MADE THINGS LIKE THAT.",
        "SOMEONE BEFORE YOU MUST HAVE DROPPED THAT. I HOPE THEY GOT OUT.",
        "IT'S STRANGE, SEEING SOMETHING BEAUTIFUL IN A PLACE LIKE THIS.",
    ),
    "damage-taken": (
        "YOU'RE HURT. PLEASE TELL ME YOU'RE ALRIGHT.",
        "I HATE THAT I CAN'T DO ANYTHING BUT WORRY WHEN THAT HAPPENS.",
        "SIT FOR A MOMENT. YOU DON'T HAVE TO PROVE ANYTHING TO ME.",
        "EVERY TIME YOU'RE HURT I WONDER IF THIS IS TOO MUCH TO ASK OF YOU.",
        "PLEASE BE MORE CAREFUL. I DON'T KNOW WHAT I'D DO IF YOU STOPPED COMING.",
    ),
    "quiet-moment": (
        "THE QUIET DOWN HERE USED TO TERRIFY ME. NOW IT'S JUST FAMILIAR.",
        "I COUNT THE CRACKS IN THE CEILING WHEN I CAN'T SLEEP.",
        "I TRY TO REMEMBER MY MOTHER'S VOICE. IT'S HARDER THAN IT USED TO BE.",
        "SOME DAYS THE SILENCE IS ALMOST PEACEFUL. ALMOST.",
        "I'VE HAD A LOT OF TIME TO THINK DOWN HERE. TOO MUCH, PROBABLY.",
    ),
    "joke": (
        "I'VE HAD TO GET CREATIVE FOR ENTERTAINMENT DOWN HERE.",
        "THE GUARD DOESN'T LAUGH AT MY JOKES EITHER. IT'S NOT JUST YOU.",
        "THAT ACTUALLY MADE ME LAUGH. IT'S BEEN A WHILE.",
        "I'VE NAMED THE SPIDERS IN THE CORNER. DON'T JUDGE ME.",
        "CAPTIVITY HUMOR IS AN ACQUIRED TASTE. I'M SORRY IN ADVANCE.",
    ),
    "encouragement": (
        "YOU'VE COME THIS FAR. DON'T STOP NOW.",
        "I BELIEVE YOU CAN DO THIS. I DON'T SAY THAT LIGHTLY.",
        "GET UP. I'M NOT READY TO STOP HOPING YET, AND NEITHER SHOULD YOU BE.",
        "WHATEVER HAPPENED OUT THERE, YOU'RE STILL HERE. THAT COUNTS.",
        "I'VE SEEN WHAT YOU'RE CAPABLE OF. TRUST THAT, EVEN WHEN IT'S HARD.",
    ),
    "farewell": (
        "GO CAREFULLY. I'LL BE HERE. I'M ALWAYS HERE.",
        "COME BACK TO ME. PLEASE.",
        "I'LL COUNT THE HOURS UNTIL YOU RETURN, LIKE ALWAYS.",
        "BE SAFE OUT THERE. YOU'RE ALL THE HOPE I HAVE LEFT.",
        "UNTIL NEXT TIME. I MEAN THAT MORE THAN YOU KNOW.",
    ),
}


def _response(rng: random.Random, trust_tier: int, mood: str, context: str,
              lore_bank_enabled: bool = True) -> str:
    """Draw order fixed -- same determinism contract as korrath_corpus's
    own _response(). No catchphrase bank, same reasoning as Korrath's own
    (a captive repeating a fixed refrain reads as comic, not sincere).
    A shared Ravendale-lore clause (ravendale_lore.py) is appended
    probabilistically, same mechanism/discipline as Shadewrath's and
    Korrath's own splice -- the M11 quality-push lever (docs/plan.md
    Known follow-ups): reinforced content across the three narratively-
    linked characters, not a wholesale voice-bank reuse.

    M11.1 Part 2: lore_bank_enabled gates the draw's RESULT, not the
    draw itself -- see shadewrath_corpus._response()'s docstring for why."""
    parts = []
    if rng.random() < 0.6:
        parts.append(rng.choice(_OPENERS[mood]))
    parts.append(rng.choice(_BODIES[context]))
    lore_drawn = rng.random() < 0.2
    lore_line = rng.choice(RAVENDALE_LORE) if lore_drawn else None
    if lore_bank_enabled and lore_line:
        parts.append(lore_line)
    if rng.random() < 0.35:
        parts.append(rng.choice(_CLOSERS[trust_tier]))
    return " ".join(parts)


# ---- CLOSER: trust-tier arc, appended ~35% of the time -----------------
# Tier 0: guarded, testing whether this is real help or another trick.
# Tier 1: opens up about her captivity and what she's learned watching
# Shadewrath. Tier 2: the actual rescue moment -- asking, plainly, to be
# freed. Reaching tier 2 for the first time is the real "rescued" event
# (SaveData::recordPrincessTier(), the third gossip trigger).

_CLOSERS = {
    0: (
        "I DON'T KNOW YOU YET. FORGIVE ME FOR BEING CAREFUL.",
        "HE'S SENT OTHERS BEFORE, PRETENDING TO HELP. I WON'T SAY MORE.",
        "PROVE YOU'RE DIFFERENT. I'VE BEEN WRONG BEFORE.",
        "I WANT TO TRUST YOU. WANTING ISN'T THE SAME AS DOING.",
        "ASK ME AGAIN WHEN I KNOW YOU BETTER.",
        "I'VE LEARNED NOT TO HOPE TOO QUICKLY HERE.",
    ),
    1: (
        "HE TALKS TO HIMSELF WHEN HE THINKS NO ONE'S LISTENING. ABOUT A DOOR.",
        "THE KNIGHT WHO GUARDS ME ISN'T CRUEL. I THINK HE'S TRAPPED TOO, SOMEHOW.",
        "I'VE BEEN HERE LONG ENOUGH TO LEARN HIS PATTERNS. THAT MIGHT MATTER.",
        "I'M STARTING TO BELIEVE YOU MIGHT ACTUALLY MEAN IT.",
        "THERE'S SOMETHING BENEATH RAVENDALE HE WON'T STOP TALKING ABOUT.",
        "I HAVEN'T TOLD ANYONE ELSE THAT. I DON'T KNOW WHY I'M TELLING YOU.",
    ),
    2: (
        "PLEASE. GET ME OUT OF HERE. I CAN'T DO IT MYSELF.",
        "I TRUST YOU. THAT'S NOT A WORD I USE LIGHTLY ANYMORE.",
        "WHATEVER IT COSTS, I WANT TO GO HOME. TAKE ME WITH YOU.",
        "I'M READY. I'VE BEEN READY FOR LONGER THAN YOU KNOW.",
        "FREE ME, AND I'LL TELL YOU EVERYTHING I'VE LEARNED DOWN HERE.",
        "THANK YOU. WHATEVER HAPPENS NEXT, THANK YOU FOR NOT GIVING UP ON ME.",
    ),
}


def generate_pairs(seed: int = 0, per_combo: int = 4,
                   lore_bank_enabled: bool = True) -> list[tuple[str, str]]:
    """per_combo pairs for each trust_tier x mood x context combo (3 x 5
    x 8 = 120 combos). Default per_combo=4, matching Korrath's own mid-
    tier density exactly -- both are mid tier, same discipline.
    lore_bank_enabled: see _response()'s docstring."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(per_combo):
        for trust_tier in TRUST_TIERS:
            for mood in MOODS:
                for context in CONTEXTS:
                    event = rng.choice(EVENTS_FOR_CONTEXT[context])
                    prompt = prompt_for(trust_tier, mood, context, event)
                    response = _response(rng, trust_tier, mood, context, lore_bank_enabled)
                    pairs.append((prompt, response))
    return pairs


def corpus_text(seed: int = 0, per_combo: int = 4,
                lore_bank_enabled: bool = True) -> str:
    return "".join(p + r for p, r in generate_pairs(seed, per_combo, lore_bank_enabled))
