"""M14 portability proof (docs/milestones/m14.md section 2): "port one
archetype to a deliberately different genre and confirm it works
through the unmodified toolkit... a sci-fi 'engineer' instead of a
fantasy 'guard'." This is that archetype -- deliberately mirrors
guard_corpus.py's shape (OPENER/BODY/CLOSER template grammar, same
compositional schema via npc_service.prompt_fields, same thin-corpus
precedent: a couple of mood/context values, not a finished cast) with
ONE fixed instance rather than 4, since the proof only needs one
archetype, not a full roster (m14.md section 2's own "lightweight
version" framing).

Zero core/, npc_service.py, or training-pipeline changes -- only new
corpus content and new field VALUES (OCC:engineer instead of OCC:guard,
new CONTEXTS). MOODS/TRUST_TIERS are reused as-is from selena_corpus.py:
mood/closeness are genre-agnostic axes, and reusing them rather than
inventing sci-fi-flavored equivalents is itself part of the portability
claim -- the schema's shared axes should transfer, only the archetype-
specific vocabulary (occupation, context, voice content) needs to be
genre-local.
"""
import random

from ngpt_trainer.selena_corpus import MOODS, TRUST_TIERS
from ngpt_trainer.npc_service import prompt_fields

ENGINEER_ID = "engineer#7"

# One hand-picked profile, not seed-jittered (no spawn_*_instance()
# equivalent exists for this genre -- that machinery is guard-specific
# and out of scope for a one-instance proof). Values chosen for a
# distinct, plausible voice: competent and focused, moderate warmth,
# not reckless.
ENGINEER_PROFILE = {
    "occupation": "engineer", "species": "human", "bond": "stranger",
    "age": 35, "gender": "female",
    "traits": {"warmth": 55, "humor": 40, "impulsivity": 30,
               "bravery": 65, "focus": 85},
}

# Thin, matching guard_corpus.py's own "3 representative contexts, not
# all 8" precedent -- this corpus exists to prove the mechanism
# transfers, not to ship a finished sci-fi cast.
ENGINEER_CONTEXTS = ("reactor-check", "hull-breach", "quiet-shift")

_TRUST_TIER_MIDPOINT = {0: 0.100, 1: 0.500, 2: 0.975}


def _relationship_state(trust_tier: int) -> dict:
    v = _TRUST_TIER_MIDPOINT[trust_tier]
    return {"familiarity": v, "affection": v, "trust": v, "respect": v, "fear": 0.0}


_ALONE_MOODS = ("tender", "embarrassed")


def prompt_for(trust_tier: int, mood: str, context: str, event: str = "") -> str:
    audience = "alone" if mood in _ALONE_MOODS else "witnessed"
    return prompt_fields(ENGINEER_PROFILE, _relationship_state(trust_tier),
                         mood, context, audience, event)


def combo_key(prompt: str) -> tuple[int, str, str]:
    fields = {}
    for tok in prompt.rstrip("|").split(" "):
        k, _, v = tok.partition(":")
        fields[k] = v
    tier_by_r = {"stranger": 0, "neutral": 1, "best_friend": 2}
    return tier_by_r[fields["R"]], fields["M"], fields["C"]


# ---- OPENER: mood-specific vocal tic, prefixed ~60% of draws ----------
_OPENERS = {
    "cheerful": ("REACTOR'S PURRING LIKE A KITTEN.", "CLEAN READING, TOP TO BOTTOM.",
                 "THAT'S THE KIND OF SHIFT I LIKE.", "SYSTEMS GREEN ACROSS THE BOARD.",
                 "COULDN'T HAVE ASKED FOR A SMOOTHER RUN."),
    "worried": ("I'M SEEING A FLUCTUATION I DON'T LIKE.", "SOMETHING'S DRAWING TOO MUCH POWER.",
                "THAT COOLANT LINE HAS ME NERVOUS.", "PRESSURE'S CLIMBING FASTER THAN IT SHOULD.",
                "I WANT EYES ON THIS BEFORE IT GETS WORSE."),
    "sassy": ("DID YOU EVEN READ THE MANUAL?", "THAT'S NOT HOW ANY OF THIS WORKS.",
              "BOLD OF YOU TO TOUCH THAT WITHOUT ASKING.",
              "I'VE SEEN CADETS DO BETTER.", "SPARE ME THE EXCUSES, JUST FIX IT."),
    "tender": ("YOU KEPT THIS SHIP TOGETHER TODAY.", "I DON'T SAY THIS OFTEN -- GOOD WORK.",
               "THAT WAS STEADY HANDS UNDER PRESSURE.",
               "I NOTICED THE EXTRA EFFORT ON THE CONDUIT.",
               "YOU'VE EARNED A REAL BREAK AFTER THAT."),
    "embarrassed": ("I MISCALIBRATED THAT. MY ERROR.", "THAT'S ON ME, NOT THE HARDWARE.",
                    "I SHOULD HAVE CAUGHT THAT SOONER.",
                    "NOT MY FINEST DIAGNOSTIC, ADMITTEDLY.",
                    "LET'S NOT PUT THAT IN THE LOG."),
}

# ---- BODY: context skeletons ------------------------------------------
_BODIES = {
    "reactor-check": (
        "CORE TEMPERATURE'S HOLDING STEADY, WATCH THE {a} GAUGE ANYWAY.",
        "RUNNING DIAGNOSTICS ON THE {a} NOW, GIVE ME A MINUTE.",
        "THE {a} READING LOOKS NOMINAL, BUT I'M DOUBLE-CHECKING.",
        "EVERY DIAL ON THIS PANEL DEPENDS ON THE {a} STAYING IN RANGE.",
        "I'VE FLAGGED THE {a} FOR A CLOSER LOOK NEXT SHIFT.",
        "THAT {a} HAS BEEN DRIFTING ALL WEEK, WORTH TRACKING.",
    ),
    "hull-breach": (
        "SEAL THE BULKHEAD NEAR THE {a}, WE'RE LOSING PRESSURE.",
        "THE {a} TOOK THE WORST OF IT, PATCHING NOW.",
        "GET CLEAR OF THE {a}, IT'S STILL VENTING.",
        "STRUCTURAL INTEGRITY NEAR THE {a} IS MY MAIN CONCERN.",
        "I CAN HOLD THE {a} TOGETHER, BUT NOT FOREVER.",
        "THAT {a} BREACH IS SEALED, FOR NOW.",
    ),
    "quiet-shift": (
        "NOTHING ON THE BOARD TONIGHT. SUITS ME FINE.",
        "SLOW SHIFT. GOOD TIME TO CATCH UP ON MAINTENANCE LOGS.",
        "THE SHIP HUMS DIFFERENTLY WHEN EVERYTHING'S RUNNING RIGHT.",
        "I DON'T MIND THE QUIET ONES. EASIER ON THE INSTRUMENTS.",
        "NOTHING TO REPORT. I'LL TAKE IT.",
        "STILL, LIKE THIS. I ALMOST FORGET WE'RE MOVING AT ALL.",
    ),
}

_SLOT_A = {
    "reactor-check": ("COOLANT", "PLASMA CONDUIT", "CORE SHIELDING", "INTAKE VALVE"),
    "hull-breach": ("AIRLOCK", "CARGO BAY", "OUTER PLATING", "PORT CONDUIT"),
    "quiet-shift": (),
}


def _fill(rng: random.Random, context: str, body: str) -> str:
    slots = _SLOT_A.get(context) or ()
    if "{a}" in body and slots:
        body = body.replace("{a}", rng.choice(slots))
    return body


# ---- CLOSER: trust-tier relationship depth, appended ~35% -------------
_CLOSERS = {
    0: ("I DON'T KNOW YOU WELL ENOUGH TO SAY MORE.", "STANDARD PROTOCOL APPLIES TO YOU TOO.",
        "WE'LL SEE HOW YOU HANDLE THE NEXT ONE.",
        "I RESERVE JUDGMENT UNTIL I'VE SEEN MORE.",
        "NEW CREW GET A SECOND LOOK FROM ME."),
    1: ("YOU'VE BEEN RELIABLE SO FAR. I NOTICE THAT.",
        "I'VE STOPPED DOUBLE-CHECKING YOUR WORK.",
        "YOU'RE EARNING YOUR PLACE ON THIS CREW.",
        "MY READS ON YOU HAVE BEEN TRENDING POSITIVE.",
        "CONSISTENCY COUNTS FOR A LOT WITH ME."),
    2: ("YOU'RE ONE OF THE FEW I TRUST NEAR THE CORE UNSUPERVISED.",
        "I'D PUT MY NAME BEHIND YOURS, IF ASKED.",
        "THAT'S THE HIGHEST TRUST I EXTEND ON THIS SHIP.",
        "WHATEVER BREAKS NEXT, I WANT YOU ON IT WITH ME.",
        "YOU'VE EARNED A CONFIDENCE I DON'T HAND OUT LIGHTLY."),
}


def _response(rng: random.Random, trust_tier: int, mood: str, context: str) -> str:
    parts = []
    if rng.random() < 0.6:
        parts.append(rng.choice(_OPENERS[mood]))
    body = _fill(rng, context, rng.choice(_BODIES[context]))
    parts.append(body)
    if rng.random() < 0.35:
        parts.append(rng.choice(_CLOSERS[trust_tier]))
    return " ".join(parts)


def generate_pairs(seed: int = 0, per_combo: int = 3) -> list[tuple[str, str]]:
    """per_combo pairs for each of TRUST_TIERS x MOODS x ENGINEER_CONTEXTS
    combos, interleaved same as guard_corpus.generate_pairs -- any
    prefix covers every combo at least once before repeating."""
    rng = random.Random(seed)
    combos = [
        (tier, mood, ctx)
        for tier in TRUST_TIERS
        for mood in MOODS
        for ctx in ENGINEER_CONTEXTS
    ]
    pairs = []
    for _ in range(per_combo):
        for tier, mood, ctx in combos:
            prompt = prompt_for(tier, mood, ctx)
            response = _response(rng, tier, mood, ctx)
            pairs.append((prompt, response))
    return pairs


def corpus_text(seed: int = 0, per_combo: int = 3) -> str:
    return "".join(p + r for p, r in generate_pairs(seed=seed, per_combo=per_combo))
