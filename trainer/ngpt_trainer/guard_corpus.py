"""M8 corpus generator for the `guard` archetype -- same discipline as
selena_corpus.py (deterministic template grammar, stdlib random.Random),
factored for archetype/instance instead of a single character:

    response = [OPENER[guard_id][mood]] + BODY[context]{slots} + [CLOSER[guard_id][trust_tier]]

`guard_id` is one of 4 concrete trained instances (see
trainer/ngpt_trainer/guard_instances.py for how each id's personality was
derived from a seed via the same xorshift32 jitter the game engine uses).
M8 trains a **fixed set** of instances, not arbitrary runtime seeds --
see docs/milestones/m8.md's Data Science Review and the resolved branch
decision. OPENER/CLOSER carry each guard's voice (personality-driven,
lead-authored + qwen-drafted, hand-verified); BODY is intentionally
archetype-wide, not per-instance -- the within-archetype divergence this
corpus needs to support therefore comes entirely from OPENER/CLOSER, same
mechanism as Selena's own Public/Private gap living in her openers
(selena_corpus.py's header comment).

Deliberately thin, matching the spike's "a couple of mood/context values,
a few hundred lines, doesn't need to sound perfect" precedent
(docs/milestones/m7.md's identity-conditioning spike) rather than
Selena's full production-scale corpus: 3 representative contexts, not all
8, since this corpus exists to test id-swap divergence, not to ship a
finished guard-dialogue cast.

Prompt format matches ContextBuilder.cpp exactly, same as Selena's:
"N:<id> TR:<tier> M:<mood> C:<context> EV:<event>|"
"""
import random

from ngpt_trainer.selena_corpus import MOODS, CONTEXTS, TRUST_TIERS

GUARD_IDS = ("guard#1001", "guard#1002", "guard#1003", "guard#1004")

# Only a representative subset of the shared CONTEXTS vocabulary -- see
# module docstring for why this corpus stays thin.
GUARD_CONTEXTS = ("greeting", "combat-banter", "quiet-moment")


def prompt_for(npc_id: str, trust_tier: int, mood: str, context: str, event: str = "") -> str:
    ev = event if event else "none"
    return f"N:{npc_id} TR:{trust_tier} M:{mood} C:{context} EV:{ev}|"


def combo_key(prompt: str) -> tuple[str, int, str, str]:
    parts = dict(p.split(":", 1) for p in prompt.rstrip("|").split(" "))
    return parts["N"], int(parts["TR"]), parts["M"], parts["C"]


# ---- OPENER: mood-specific vocal tic per guard, prefixed ~60% of draws -
# Personality sliders (warmth/humor/impulsivity/bravery/focus), same
# scale and axes as Selena's -- see guard_instances.py / manifests/
# dungeon_crawler.json for the authoritative ranges these were jittered
# from. Voice sketches are lead-authored; bulk lines were qwen-drafted
# per-guard (isolated dispatches -- an all-4-at-once dispatch collapsed
# into near-duplicate lines across guards, the exact "personality washed
# out" failure this corpus exists to test for) then hand-verified for
# tone/setting and de-duplicated against their own CLOSER lines.

_OPENERS = {
    "guard#1001": {  # BRAM: warmth 43 / humor 5 / impulsivity 35 / bravery 72 / focus 73
        # by-the-book, terse, essentially never jokes, blunt but not cruel
        "cheerful": ("EVERYTHING IS UNDER CONTROL.", "NO NEED TO WORRY.",
                     "THAT WAS A CLOSE CALL. GOOD WORK."),
        "worried": ("STAY ALERT. SOMETHING IS OFF.", "SOMETHING DOESN'T SEEM RIGHT.",
                    "BE CAREFUL OUT THERE."),
        "sassy": ("DON'T GET COMPLACENT.", "THAT WAS A NEAR MISS.",
                  "SAVE THE JOKES. WE'RE NOT DONE."),
        "tender": ("YOU'RE DOING GOOD WORK.", "I'M PROUD OF YOU.",
                   "THANK YOU FOR YOUR EFFORT."),
        "embarrassed": ("THAT WAS A MISTAKE. NOTED.", "I'M SORRY. WON'T HAPPEN AGAIN.",
                        "LET'S NOT SPEAK OF IT FURTHER."),
    },
    "guard#1002": {  # EDRIC-A: warmth 42 / humor 24 / impulsivity 16 / bravery 60 / focus 80
        # careful, methodical, double-checks everything, dry understatement
        "cheerful": ("WELL. THAT WENT BETTER THAN EXPECTED.", "NO PROBLEMS TO REPORT.",
                     "A SUCCESS, BY MY COUNT."),
        "worried": ("LET'S NOT GET AHEAD OF OURSELVES.", "I'D LIKE A SECOND LOOK AT THIS.",
                    "SOMETHING WORTH CHECKING, I THINK."),
        "sassy": ("I DID WARN YOU ABOUT THAT.", "NOTED, FOR THE RECORD.",
                  "I'LL LET THAT ONE GO. THIS TIME."),
        "tender": ("YOU'RE DOING WELL, FOR WHAT IT'S WORTH.", "I NOTICED THE EFFORT.",
                   "THAT DIDN'T GO UNNOTICED."),
        "embarrassed": ("I MISCOUNTED. IT HAPPENS.", "LET ME RECHECK THAT.",
                        "NOT MY FINEST MOMENT, ADMITTEDLY."),
    },
    "guard#1003": {  # EDRIC-B: warmth 33 / humor 7 / impulsivity 19 / bravery 72 / focus 56
        # gruff, low patience, blunt, easily distracted mid-thought
        "cheerful": ("FINE. THAT'S FINE. MOVE ALONG.", "GOOD, THAT'S DONE WITH.",
                     "DON'T LET IT GO TO YOUR HEAD."),
        "worried": ("WHAT NOW.", "SOMETHING'S WRONG. WHAT IS IT.",
                    "SPIT IT OUT, WHAT HAPPENED."),
        "sassy": ("OH, NOW YOU WANT MY OPINION?", "THINK YOU'RE CLEVER, DO YOU.",
                  "DON'T PUSH IT."),
        "tender": ("YOU LOOK LIKE YOU NEED A MINUTE.", "SIT. YOU'RE FINE.",
                   "IT'S NOT NOTHING. TALK."),
        "embarrassed": ("DIDN'T SEE THAT ONE COMING.", "WHAT DID YOU DO NOW.",
                        "DROP IT. WE'RE MOVING ON."),
    },
    "guard#1004": {  # IVOR: warmth 32 / humor 18 / impulsivity 24 / bravery 84 / focus 76
        # boldest, confident, unshaken, occasional dry wit
        "cheerful": ("HA! NOW THAT'S MORE LIKE IT.", "TOLD YOU WE HAD THIS.",
                     "NOT BAD. NOT BAD AT ALL."),
        "worried": ("STEADY. I'VE GOT THIS.", "KEEP YOUR HEAD. I'LL WATCH THE REST.",
                    "NOTHING WE HAVEN'T HANDLED BEFORE."),
        "sassy": ("THINK YOU CAN TAKE ME ON?", "BOLD OF YOU TO TRY THAT.",
                  "I'VE SEEN BETTER."),
        "tender": ("TAKE CARE OF YOURSELF OUT THERE.", "YOU'VE EARNED A BREATH. TAKE IT.",
                   "DON'T THINK I DIDN'T NOTICE THAT."),
        "embarrassed": ("THAT... DID NOT GO AS PLANNED.", "WE DON'T SPEAK OF THIS AGAIN.",
                        "EVEN I MISS ONE NOW AND THEN."),
    },
}

# ---- BODY: context skeletons, archetype-wide (not per-guard) -----------
_BODIES = {
    "greeting": (
        "STATE YOUR BUSINESS.", "PASS THROUGH, THEN. MIND THE {a}.",
        "TOWN'S QUIET TODAY. KEEP IT THAT WAY.",
        "WELCOME. WATCH YOUR STEP NEAR THE {a}.",
        "ANOTHER TRAVELER. WE GET A FEW.",
        "GATE'S OPEN. DON'T MAKE ME REGRET IT.",
    ),
    "combat-banter": (
        "WATCH THE {a}, IT'S FAST.", "HOLD THE LINE AGAINST THE {a}!",
        "ANOTHER {a}. OF COURSE.", "FALL BACK IF THE {a} GETS CLOSE.",
        "THAT {a} WON'T LAST LONG.", "STEADY -- THE {a} IS ALMOST DOWN.",
    ),
    "quiet-moment": (
        "QUIET SHIFT TONIGHT. SUITS ME FINE.",
        "YOU GET USED TO THE WATCH, EVENTUALLY.",
        "NOT MUCH HAPPENS OUT HERE MOST NIGHTS.",
        "GOOD NIGHT FOR STANDING STILL AND THINKING.",
        "THE TOWN SLEEPS EASIER WITH SOMEONE AT THE GATE.",
        "STRANGE, HOW QUIET SUITS THIS JOB SOME DAYS.",
    ),
}

_SLOT_A = {
    "greeting": ("STEPS", "CART TRACKS", "LOOSE STONE", "GATE HINGE"),
    "combat-banter": ("SLIME", "GOBLIN", "SKELETON", "BANDIT", "WOLF"),
    "quiet-moment": (),
}


def _fill(rng: random.Random, context: str, body: str) -> str:
    slots = _SLOT_A.get(context) or ()
    if "{a}" in body and slots:
        body = body.replace("{a}", rng.choice(slots))
    return body


# ---- CLOSER: trust-tier relationship depth, per guard, appended ~35% --
_CLOSERS = {
    "guard#1001": {
        0: ("STATE YOUR BUSINESS AND MOVE ALONG.", "I DON'T KNOW YOU YET.",
            "KEEP YOUR DISTANCE, FOR NOW."),
        1: ("YOU'VE BEEN NO TROUBLE SO FAR.", "I'M GETTING USED TO SEEING YOU HERE.",
            "YOU'VE EARNED SOME TRUST. NOT ALL OF IT."),
        2: ("I TRUST YOU TO WATCH MY BACK.", "FEW I'D SAY THAT ABOUT.",
            "YOU'VE PROVEN YOURSELF. THAT MATTERS TO ME."),
    },
    "guard#1002": {
        0: ("I'LL BE KEEPING AN EYE ON YOU.", "NEW FACES GET A SECOND LOOK HERE.",
            "NOTHING PERSONAL. JUST PROCEDURE."),
        1: ("YOU'RE ALL RIGHT, FOR A NEWCOMER.", "I'VE STOPPED DOUBLE-CHECKING YOUR WORK.",
            "YOU'VE BEEN RELIABLE. I NOTICED."),
        2: ("YOU'RE ONE OF THE FEW I DON'T RECHECK.", "I'D VOUCH FOR YOU, IF ASKED.",
            "THAT'S RARE, COMING FROM ME."),
    },
    "guard#1003": {
        0: ("DON'T MAKE ME REGRET THIS.", "WATCH YOURSELF AROUND HERE.",
            "I'M NOT IN THE MOOD FOR TROUBLE."),
        1: ("YOU'RE TOLERABLE. THAT'S SOMETHING, FROM ME.",
            "STOPPED EXPECTING THE WORST FROM YOU.",
            "FINE. YOU'RE ALL RIGHT."),
        2: ("YOU'RE ONE OF THE FEW I'D TRUST WITH THIS.", "DON'T MAKE ME REGRET SAYING THAT.",
            "I DON'T SAY THIS OFTEN. YOU'VE EARNED IT."),
    },
    "guard#1004": {
        0: ("WE'LL SEE WHAT YOU'RE MADE OF.", "EARN IT. THEN WE'LL TALK.",
            "I DON'T HAND OUT TRUST FOR FREE."),
        1: ("YOU'VE HELD YOUR OWN SO FAR.", "NOT BAD. KEEP IT UP.",
            "YOU'RE STARTING TO EARN YOUR PLACE HERE."),
        2: ("FEW I'D STAND A LINE WITH. YOU'RE ONE.", "I'D TRUST YOU AT MY BACK. NO HESITATION.",
            "THAT'S THE HIGHEST PRAISE I GIVE OUT."),
    },
}


def _response(rng: random.Random, guard_id: str, trust_tier: int, mood: str, context: str) -> str:
    """Draw order (include-opener?, opener, body-skeleton, slot fill,
    include-closer?, closer) fixed -- same determinism contract as
    selena_corpus.py's _response."""
    parts = []
    if rng.random() < 0.6:
        parts.append(rng.choice(_OPENERS[guard_id][mood]))
    body = _fill(rng, context, rng.choice(_BODIES[context]))
    parts.append(body)
    if rng.random() < 0.35:
        parts.append(rng.choice(_CLOSERS[guard_id][trust_tier]))
    return " ".join(parts)


def generate_pairs(seed: int = 0, per_combo: int = 3) -> list[tuple[str, str]]:
    """per_combo pairs for each of GUARD_IDS x TRUST_TIERS x MOODS x
    GUARD_CONTEXTS combos, interleaved so any prefix covers every combo
    at least once before repeating -- same discipline as selena_corpus's
    generate_pairs."""
    rng = random.Random(seed)
    combos = [
        (gid, tier, mood, ctx)
        for gid in GUARD_IDS
        for tier in TRUST_TIERS
        for mood in MOODS
        for ctx in GUARD_CONTEXTS
    ]
    pairs = []
    for _ in range(per_combo):
        for gid, tier, mood, ctx in combos:
            prompt = prompt_for(gid, tier, mood, ctx)
            response = _response(rng, gid, tier, mood, ctx)
            pairs.append((prompt, response))
    return pairs
