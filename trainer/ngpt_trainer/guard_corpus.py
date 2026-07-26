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

M11.1 (docs/milestones/m11.1.md Part 1): genericized onto NpcService's
compositional scheme, the last old-scheme holdout closed -- migrating
this corpus (not just guard's own voice content) means guard#1001-1004's
OCC:guard tokens now sit in the SAME occupation bank cast_corpus.py's
Bram already trained (a real, deliberate voice-merge tradeoff weighed and
accepted, not an oversight -- see that milestone doc's Part 1 discussion).
GUARD_PROFILES below is built from guard_instances.spawn_guard_instance()
(the same seed-jitter the live game's spawnInstance() uses for these
exact 4 fixed seeds), not hand-invented values, so this corpus and
game/src/user/NPCDatabase.cpp's guardInstances[] describe the identical
4 individuals.
"""
import random

from ngpt_trainer.selena_corpus import MOODS, CONTEXTS, TRUST_TIERS
from ngpt_trainer.npc_service import prompt_fields
from ngpt_trainer.guard_instances import spawn_guard_instance

GUARD_IDS = ("guard#1001", "guard#1002", "guard#1003", "guard#1004")
GUARD_SEEDS = {"guard#1001": 0x1001, "guard#1002": 0x1002,
               "guard#1003": 0x1003, "guard#1004": 0x1004}

# One real Profile per fixed instance -- occupation/species/bond are
# uniform (every guard is the same archetype), age/gender/traits come
# from the actual seed jitter (spawn_guard_instance(), cross-checked
# against the compiled engine by test_guard_instances.py).
GUARD_PROFILES = {
    gid: {
        "occupation": "guard", "species": "human", "bond": "stranger",
        "age": spawn_guard_instance(seed)["age"],
        "gender": spawn_guard_instance(seed)["gender"],
        "traits": spawn_guard_instance(seed)["personality"],
    }
    for gid, seed in GUARD_SEEDS.items()
}

# Only a representative subset of the shared CONTEXTS vocabulary -- see
# module docstring for why this corpus stays thin.
GUARD_CONTEXTS = ("greeting", "combat-banter", "quiet-moment")

# Matches DialogueDemo.cpp's relationshipForTrustTier() exactly -- same
# rationale as selena_corpus.py's _TRUST_TIER_MIDPOINT.
_TRUST_TIER_MIDPOINT = {0: 0.100, 1: 0.500, 2: 0.975}


def _relationship_state(trust_tier: int) -> dict:
    v = _TRUST_TIER_MIDPOINT[trust_tier]
    return {"familiarity": v, "affection": v, "trust": v, "respect": v, "fear": 0.0}


# AUD: -- same labeling-pass approach as selena_corpus.py: "tender" is
# every guard's warmest, most personal opener bank (e.g. guard#1001's
# "YOU'RE DOING GOOD WORK. I'M PROUD OF YOU." vs. his terse professional
# default), "embarrassed" the one place a guard admits a mistake plainly
# rather than deflecting -- the two moods that read as off-duty/private
# rather than on-duty/public.
_ALONE_MOODS = ("tender", "embarrassed")


def prompt_for(npc_id: str, trust_tier: int, mood: str, context: str, event: str = "") -> str:
    audience = "alone" if mood in _ALONE_MOODS else "witnessed"
    return prompt_fields(GUARD_PROFILES[npc_id], _relationship_state(trust_tier),
                         mood, context, audience, event)


def combo_key(prompt: str) -> tuple[int, str, str]:
    """(trust_tier, mood, context) -- guard_id isn't recoverable from the
    prompt string anymore (P:/OCC: carry no per-instance identity, same
    as every other genericized character; the age/gender/traits in P:/D:
    already distinguish the 4 instances without a name tag). Same shape
    as selena_corpus.combo_key()."""
    fields = {}
    for tok in prompt.rstrip("|").split(" "):
        k, _, v = tok.partition(":")
        fields[k] = v
    tier_by_r = {"stranger": 0, "neutral": 1, "best_friend": 2}
    return tier_by_r[fields["R"]], fields["M"], fields["C"]


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
                     "THAT WAS A CLOSE CALL. GOOD WORK.",
                     "NOTED. WELL HANDLED.", "THAT WENT AS EXPECTED.",
                     "ACCEPTABLE OUTCOME.", "NO COMPLAINTS FROM ME.",
                     "SOLID WORK, FOR ONCE."),
        "worried": ("STAY ALERT. SOMETHING IS OFF.", "SOMETHING DOESN'T SEEM RIGHT.",
                    "BE CAREFUL OUT THERE.",
                    "KEEP YOUR GUARD UP.", "SOMETHING'S NOT ADDING UP.",
                    "I DON'T LIKE THE LOOK OF THIS.", "STAY SHARP. I MEAN IT.",
                    "THIS WARRANTS CAUTION."),
        "sassy": ("DON'T GET COMPLACENT.", "THAT WAS A NEAR MISS.",
                  "SAVE THE JOKES. WE'RE NOT DONE.",
                  "FOCUS. THIS ISN'T OVER.", "SPARE ME THE COMMENTARY.",
                  "THAT WAS SLOPPY. NOTED.", "SAVE IT FOR AFTER THE SHIFT.",
                  "I'VE HEARD BETTER EXCUSES."),
        "tender": ("YOU'RE DOING GOOD WORK.", "I'M PROUD OF YOU.",
                   "THANK YOU FOR YOUR EFFORT.",
                   "YOU'VE EARNED SOME REST.", "I NOTICED THE EXTRA EFFORT.",
                   "THAT MEANT SOMETHING TO ME.",
                   "YOU'VE GROWN INTO THIS ROLE WELL.",
                   "I DON'T SAY THIS OFTEN. WELL DONE."),
        "embarrassed": ("THAT WAS A MISTAKE. NOTED.", "I'M SORRY. WON'T HAPPEN AGAIN.",
                        "LET'S NOT SPEAK OF IT FURTHER.",
                        "THAT SHOULDN'T HAVE HAPPENED.",
                        "I MISJUDGED THAT. IT'S NOTED.", "MY ERROR. MOVING ON.",
                        "THAT WAS BENEATH MY STANDARDS.",
                        "CONSIDER IT FORGOTTEN. BY BOTH OF US."),
    },
    "guard#1002": {  # EDRIC-A: warmth 42 / humor 24 / impulsivity 16 / bravery 60 / focus 80
        # careful, methodical, double-checks everything, dry understatement
        "cheerful": ("WELL. THAT WENT BETTER THAN EXPECTED.", "NO PROBLEMS TO REPORT.",
                     "A SUCCESS, BY MY COUNT.",
                     "THAT MET MY EXPECTATIONS. BARELY EXCEEDED THEM, ACTUALLY.",
                     "I'LL MARK THIS ONE DOWN AS A WIN.",
                     "SURPRISINGLY, NOTHING WENT WRONG.",
                     "I'D CALL THAT A CLEAN OUTCOME.",
                     "NO NOTES. THAT'S RARE FROM ME."),
        "worried": ("LET'S NOT GET AHEAD OF OURSELVES.", "I'D LIKE A SECOND LOOK AT THIS.",
                    "SOMETHING WORTH CHECKING, I THINK.",
                    "THAT DOESN'T QUITE ADD UP. LET ME RECOUNT.",
                    "I'D FEEL BETTER WITH A CLOSER LOOK.",
                    "SOMETHING'S OFF BY A MARGIN I DON'T LIKE.",
                    "LET'S VERIFY BEFORE WE PROCEED.",
                    "I'VE GOT A BAD FEELING, WHICH IS RARE FOR ME."),
        "sassy": ("I DID WARN YOU ABOUT THAT.", "NOTED, FOR THE RECORD.",
                  "I'LL LET THAT ONE GO. THIS TIME.",
                  "I HAD THAT FILED UNDER 'PREDICTABLE.'", "NOTED. AGAIN.",
                  "I DID SAY SO, DIDN'T I.",
                  "THAT'S GOING IN MY REPORT, VERBATIM.",
                  "I'M CHOOSING NOT TO COMMENT FURTHER."),
        "tender": ("YOU'RE DOING WELL, FOR WHAT IT'S WORTH.", "I NOTICED THE EFFORT.",
                   "THAT DIDN'T GO UNNOTICED.",
                   "YOUR WORK HAS BEEN CONSISTENT. THAT MATTERS TO ME.",
                   "I KEEP BETTER RECORDS THAN I LET ON. YOURS ARE GOOD.",
                   "THAT WAS WELL HANDLED. I MEAN THAT.",
                   "I DON'T OFFER PRAISE LIGHTLY. TAKE THIS AS PRAISE.",
                   "YOU'VE EARNED MY CONFIDENCE. QUIETLY, BUT YOU HAVE."),
        "embarrassed": ("I MISCOUNTED. IT HAPPENS.", "LET ME RECHECK THAT.",
                        "NOT MY FINEST MOMENT, ADMITTEDLY.",
                        "MY NUMBERS WERE OFF. RARE, BUT NOT UNHEARD OF.",
                        "I'LL AMEND MY REPORT ACCORDINGLY.",
                        "EVEN I MISCALCULATE, APPARENTLY.",
                        "THAT ERROR IS NOW LOGGED AND CORRECTED.",
                        "LET'S PRETEND THAT DIDN'T HAPPEN. FOR MY RECORDS' SAKE."),
    },
    "guard#1003": {  # EDRIC-B: warmth 33 / humor 7 / impulsivity 19 / bravery 72 / focus 56
        # gruff, low patience, blunt, easily distracted mid-thought
        "cheerful": ("FINE. THAT'S FINE. MOVE ALONG.", "GOOD, THAT'S DONE WITH.",
                     "DON'T LET IT GO TO YOUR HEAD.",
                     "FINE. WHATEVER. GOOD JOB, I GUESS.", "THAT'LL DO.",
                     "DONE. NEXT.", "NOT BAD. DON'T EXPECT THAT OFTEN.",
                     "GOOD. NOW STOP TALKING ABOUT IT."),
        "worried": ("WHAT NOW.", "SOMETHING'S WRONG. WHAT IS IT.",
                    "SPIT IT OUT, WHAT HAPPENED.",
                    "WHAT. WHAT IS IT NOW.", "SOMETHING'S OFF. I CAN FEEL IT.",
                    "JUST TELL ME THE BAD PART FIRST.",
                    "GREAT. WHAT BROKE THIS TIME.",
                    "I DON'T HAVE PATIENCE FOR SURPRISES TODAY."),
        "sassy": ("OH, NOW YOU WANT MY OPINION?", "THINK YOU'RE CLEVER, DO YOU.",
                  "DON'T PUSH IT.",
                  "OH, NOW YOU'VE GOT SOMETHING TO SAY.",
                  "SURE. SURE. WHATEVER YOU SAY.",
                  "I'M NOT IN THE MOOD, BUT GO ON.", "THAT'S CUTE. MOVE ALONG.",
                  "DON'T TEST ME TODAY."),
        "tender": ("YOU LOOK LIKE YOU NEED A MINUTE.", "SIT. YOU'RE FINE.",
                   "IT'S NOT NOTHING. TALK.",
                   "YOU'VE HAD A ROUGH GO OF IT. SIT DOWN.",
                   "TALK TO ME. I'M LISTENING, SORT OF.",
                   "YOU DID FINE. DON'T MAKE ME SAY IT TWICE.",
                   "I NOTICED. I DON'T MISS MUCH.",
                   "TAKE A BREATH. I'LL WAIT. NOT LONG, BUT I'LL WAIT."),
        "embarrassed": ("DIDN'T SEE THAT ONE COMING.", "WHAT DID YOU DO NOW.",
                        "DROP IT. WE'RE MOVING ON.",
                        "THAT WAS A MESS. FORGET I SAID ANYTHING.",
                        "WHAT WAS THAT SUPPOSED TO BE.", "MOVING ON. FAST.",
                        "I DIDN'T SEE THAT. NEITHER DID YOU.",
                        "LET'S NOT BRING THAT UP AGAIN."),
    },
    "guard#1004": {  # IVOR: warmth 32 / humor 18 / impulsivity 24 / bravery 84 / focus 76
        # boldest, confident, unshaken, occasional dry wit
        "cheerful": ("HA! NOW THAT'S MORE LIKE IT.", "TOLD YOU WE HAD THIS.",
                     "NOT BAD. NOT BAD AT ALL.",
                     "SEE? NOTHING TO IT.", "THAT'S HOW IT'S DONE.",
                     "I LOVE DAYS LIKE THIS.", "TOLD YOU I HAD A GOOD FEELING.",
                     "THAT'S GOING IN THE GOOD COLUMN."),
        "worried": ("STEADY. I'VE GOT THIS.", "KEEP YOUR HEAD. I'LL WATCH THE REST.",
                    "NOTHING WE HAVEN'T HANDLED BEFORE.",
                    "STAY CALM. I'VE SEEN WORSE.",
                    "I DON'T RATTLE EASY. THIS IS CLOSE, THOUGH.",
                    "KEEP MOVING. I'LL HANDLE THE REST.",
                    "THIS ONE'S TRICKY. STAY SHARP.",
                    "EVEN I'M PAYING ATTENTION NOW."),
        "sassy": ("THINK YOU CAN TAKE ME ON?", "BOLD OF YOU TO TRY THAT.",
                  "I'VE SEEN BETTER.",
                  "THAT THE BEST YOU'VE GOT?", "CUTE ATTEMPT. TRY AGAIN.",
                  "I'VE FACED SCARIER THAN YOU BEFORE BREAKFAST.",
                  "BOLD MOVE. WRONG ONE, BUT BOLD.",
                  "I ADMIRE THE CONFIDENCE, MISPLACED AS IT IS."),
        "tender": ("TAKE CARE OF YOURSELF OUT THERE.", "YOU'VE EARNED A BREATH. TAKE IT.",
                   "DON'T THINK I DIDN'T NOTICE THAT.",
                   "YOU'VE GOT MORE NERVE THAN YOU GIVE YOURSELF CREDIT FOR.",
                   "I DON'T WORRY ABOUT MANY PEOPLE. YOU'RE AN EXCEPTION.",
                   "THAT TOOK GUTS. I NOTICED.",
                   "REST UP. YOU'VE EARNED IT TWICE OVER.",
                   "I'D STAND WITH YOU AGAIN, NO QUESTION."),
        "embarrassed": ("THAT... DID NOT GO AS PLANNED.", "WE DON'T SPEAK OF THIS AGAIN.",
                        "EVEN I MISS ONE NOW AND THEN.",
                        "OKAY, THAT ONE'S ON ME.", "NOBODY SAW THAT, RIGHT?",
                        "EVEN THE BEST HAVE OFF DAYS. THIS WAS MINE.",
                        "LET'S CALL THAT A LEARNING MOMENT.",
                        "I'LL PRETEND THAT DIDN'T HAPPEN IF YOU DO TOO."),
    },
}

# ---- BODY: context skeletons, archetype-wide (not per-guard) -----------
_BODIES = {
    "greeting": (
        "STATE YOUR BUSINESS. MAKE IT MORE INTERESTING THAN THE LAST ONE.",
        "PASS THROUGH, THEN. MIND THE {a}, IT'S CLAIMED FINER BOOTS THAN YOURS.",
        "TOWN'S QUIET TODAY. LET'S KEEP IT BORING, SHALL WE?",
        "WELCOME. WATCH YOUR STEP NEAR THE {a}. I'M NOT SCRAPING ANYONE OFF IT.",
        "ANOTHER TRAVELER. THAT'S THREE THIS WEEK. WE'RE PRACTICALLY A CITY.",
        "GATE'S OPEN. DON'T MAKE ME WRITE ANYTHING BUT 'UNEVENTFUL' TODAY.",
        "KEEP MOVING, NOTHING TO SEE HERE. TRUST ME, I'VE LOOKED. FOR HOURS.",
        "THROUGH YOU GO, MIND THE {a}. IT'S NEWER THAN MY BACK, ANYWAY.",
        "ANOTHER DAY, ANOTHER GATE TO WATCH. THE GATE AND I ARE CLOSE NOW.",
        "STATE YOUR NAME AND YOUR REASON. BONUS POINTS FOR A GOOD STORY.",
    ),
    "combat-banter": (
        "WATCH THE {a}, IT'S FASTER THAN IT LOOKS. AND IT LOOKS FAST.",
        "HOLD THE LINE AGAINST THE {a}! THIS IS THE FUN PART!",
        "ANOTHER {a}. OF COURSE. IT'S THAT KIND OF SHIFT.",
        "FALL BACK IF THE {a} GETS CLOSE. NO MEDALS FOR STANDING STILL.",
        "THAT {a} WON'T LAST LONG. NEITHER WILL MY PATIENCE.",
        "STEADY -- THE {a} IS ALMOST DOWN. ALMOST. DON'T JINX IT.",
        "EYES ON THE {a}, DON'T LOSE IT. I HATE PAPERWORK FOR ESCAPEES.",
        "HOLD STEADY, THE {a} IS TIRING. SO AM I, BUT IT'S TIRING FIRST.",
        "ANOTHER {a} DOWN. KEEP GOING. WE'RE ON A ROLL HERE.",
        "WATCH YOUR FLANK, THE {a} MOVES FAST. FASTER THAN MY LAST TRAINEE.",
    ),
    "quiet-moment": (
        "QUIET SHIFT TONIGHT. SUITS ME FINE. LESS PAPERWORK.",
        "YOU GET USED TO THE WATCH. THE OWLS AND I HAVE AN UNDERSTANDING.",
        "NOT MUCH HAPPENS OUT HERE MOST NIGHTS. I'VE NAMED A FEW STARS.",
        "GOOD NIGHT FOR STANDING STILL AND THINKING. MOSTLY ABOUT SNACKS.",
        "THE TOWN SLEEPS EASIER WITH SOMEONE AT THE GATE. THAT SOMEONE'S ME.",
        "STRANGE, HOW QUIET SUITS THIS JOB. DON'T TELL ANYONE I SAID SO.",
        "THE NIGHT WATCH HAS ITS OWN KIND OF PEACE. AND ITS OWN BOREDOM.",
        "NOTHING STIRRING TONIGHT. GOOD. I LIKE EXCITEMENT IN SMALL DOSES.",
        "I DON'T MIND THE QUIET SHIFTS. ME, THE STARS, AND A GOOD YAWN.",
        "STILLNESS LIKE THIS MAKES THE JOB EASIER. AND THE HOURS LONGER.",
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
            "KEEP YOUR DISTANCE, FOR NOW.",
            "I'M STILL ASSESSING YOU.", "NO OPINION FORMED YET.",
            "WE'LL SEE HOW THIS GOES.", "TRUST IS EARNED HERE, NOT GIVEN."),
        1: ("YOU'VE BEEN NO TROUBLE SO FAR.", "I'M GETTING USED TO SEEING YOU HERE.",
            "YOU'VE EARNED SOME TRUST. NOT ALL OF IT.",
            "YOU'VE SHOWN SOME RELIABILITY.",
            "I'M WILLING TO GIVE YOU MORE ROPE NOW.",
            "STEADY WORK. I NOTICE THAT.",
            "YOU'RE NOT ON MY WATCH LIST ANYMORE."),
        2: ("I TRUST YOU TO WATCH MY BACK.", "FEW I'D SAY THAT ABOUT.",
            "YOU'VE PROVEN YOURSELF. THAT MATTERS TO ME.",
            "I'D STAND POST WITH YOU ANY DAY.",
            "YOU'VE GOT MY FULL CONFIDENCE. THAT'S RARE.",
            "I DON'T TRUST EASILY. I TRUST YOU.",
            "YOU'RE THE KIND OF PARTNER I'D WANT AT MY BACK."),
    },
    "guard#1002": {
        0: ("I'LL BE KEEPING AN EYE ON YOU.", "NEW FACES GET A SECOND LOOK HERE.",
            "NOTHING PERSONAL. JUST PROCEDURE.",
            "YOU'RE UNDER OBSERVATION, LIKE EVERYONE NEW.",
            "I RESERVE JUDGMENT UNTIL I HAVE DATA.",
            "TOO EARLY TO SAY ANYTHING DEFINITIVE.",
            "STANDARD PROCEDURE APPLIES TO YOU TOO."),
        1: ("YOU'RE ALL RIGHT, FOR A NEWCOMER.", "I'VE STOPPED DOUBLE-CHECKING YOUR WORK.",
            "YOU'VE BEEN RELIABLE. I NOTICED.",
            "MY RECORDS ON YOU ARE TRENDING POSITIVE.",
            "YOU'VE CLEARED MY INITIAL CONCERNS.",
            "CONSISTENCY COUNTS FOR A LOT WITH ME.",
            "I'VE STOPPED FLAGGING YOUR REPORTS FOR REVIEW."),
        2: ("YOU'RE ONE OF THE FEW I DON'T RECHECK.", "I'D VOUCH FOR YOU, IF ASKED.",
            "THAT'S RARE, COMING FROM ME.",
            "MY CONFIDENCE IN YOU IS, BY MY STANDARDS, ABSOLUTE.",
            "I'D PUT YOUR NAME FORWARD WITHOUT HESITATION.",
            "YOU'VE EARNED A TRUST I DON'T EXTEND OFTEN.",
            "IF I HAD TO CHOOSE SOMEONE, IT WOULD BE YOU."),
    },
    "guard#1003": {
        0: ("DON'T MAKE ME REGRET THIS.", "WATCH YOURSELF AROUND HERE.",
            "I'M NOT IN THE MOOD FOR TROUBLE.",
            "DON'T GIVE ME A REASON TO REGRET THIS.",
            "I'M NOT IMPRESSED YET.", "PROVE YOURSELF. THEN WE'LL TALK.",
            "NEW FACES DON'T GET MY TRUST FOR FREE."),
        1: ("YOU'RE TOLERABLE. THAT'S SOMETHING, FROM ME.",
            "STOPPED EXPECTING THE WORST FROM YOU.",
            "FINE. YOU'RE ALL RIGHT.",
            "YOU'VE STOPPED ANNOYING ME, MOSTLY.",
            "FINE. YOU'VE GOT SOME SENSE AFTER ALL.",
            "I DON'T HATE HAVING YOU AROUND ANYMORE.",
            "YOU'RE GROWING ON ME. AGAINST MY WILL."),
        2: ("YOU'RE ONE OF THE FEW I'D TRUST WITH THIS.", "DON'T MAKE ME REGRET SAYING THAT.",
            "I DON'T SAY THIS OFTEN. YOU'VE EARNED IT.",
            "YOU'RE ONE OF THE FEW I'D ACTUALLY MISS.",
            "DON'T MAKE ME SAY SOMETHING SENTIMENTAL.",
            "I'D GO TO BAT FOR YOU. DON'T TELL ANYONE I SAID THAT.",
            "YOU'VE EARNED MORE RESPECT THAN I SHOW."),
    },
    "guard#1004": {
        0: ("WE'LL SEE WHAT YOU'RE MADE OF.", "EARN IT. THEN WE'LL TALK.",
            "I DON'T HAND OUT TRUST FOR FREE.",
            "SHOW ME SOMETHING WORTH REMEMBERING.",
            "EVERYONE STARTS AT ZERO WITH ME.",
            "I DON'T HAND OUT RESPECT ON CREDIT.",
            "WE'LL SEE WHAT YOU'RE REALLY MADE OF."),
        1: ("YOU'VE HELD YOUR OWN SO FAR.", "NOT BAD. KEEP IT UP.",
            "YOU'RE STARTING TO EARN YOUR PLACE HERE.",
            "YOU'VE GOT MY ATTENTION NOW.",
            "I'D FIGHT ALONGSIDE YOU. THAT'S NOT NOTHING FROM ME.",
            "YOU'RE PROVING YOURSELF, STEP BY STEP.",
            "KEEP THIS UP AND YOU'LL EARN THE REST TOO."),
        2: ("FEW I'D STAND A LINE WITH. YOU'RE ONE.", "I'D TRUST YOU AT MY BACK. NO HESITATION.",
            "THAT'S THE HIGHEST PRAISE I GIVE OUT.",
            "THERE'S NOBODY I'D RATHER HAVE AT MY BACK.",
            "YOU'VE EARNED EVERY BIT OF THIS.",
            "I DON'T SAY THIS LIGHTLY: YOU'RE ONE OF THE BEST I'VE SEEN.",
            "WHATEVER COMES, I'M STANDING WITH YOU."),
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
