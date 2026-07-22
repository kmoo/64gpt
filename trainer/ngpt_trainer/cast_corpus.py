"""M9 curated-cast corpus generator: attempt #2, template-grammar (like
selena_corpus.py/guard_corpus.py), replacing attempt #1's freeform LLM
per-persona generation which garbled at this model's scale -- see
docs/milestones/m9.md section 4 for the measured failure and the density
comparison table this design is built from (guard's ~123K chars/instance
is the proven benchmark; attempt #1 gave ~1,300 chars/persona).

Design: almost every phrase bank is SHARED across characters, keyed by
an axis (mood, personality descriptor, context, occupation, relationship
tier) rather than by character name -- the actual test of whether the
compositional mechanism generalizes, not just four relabeled fixed
voices. The only per-character content is a small catchphrase bank per
character (Fergus's Irish-flavored lines, M9; Kragan's menacing lines,
M9.2 -- docs/milestones/m9.2.md, targeting the coherence gap M9's own
DoD flagged live on hardware for Kragan specifically). Reuses
selena_corpus.py's mood-keyed OPENERS and context-keyed BODIES directly
rather than re-authoring them -- "sassy" openers work for any sassy
character, not just Selena.

Deliberate axis-crossing: a fraction of each character's lines use a
DIFFERENT descriptor's tic bank than their own -- without this, OCC:
and D: would correlate perfectly in the training data and the model
would have no evidence they're separable, independently composable
axes (the actual thing M9 is supposed to prove).
"""
import random

from ngpt_trainer import selena_corpus as sc
from ngpt_trainer.npc_service import personality_descriptor, prompt_fields
from ngpt_trainer.m9_corpus import _relationship_state, _TIER_MIDPOINT

# ---- cast --------------------------------------------------------------
# Traits verified against personality_descriptor() to land on the
# intended word (trainer/tests/test_cast_corpus.py locks this in).

CHARACTERS = {
    "bram": {
        "occupation": "guard", "age": 35, "gender": "male",
        "traits": {"warmth": 30, "humor": 15, "impulsivity": 20,
                  "bravery": 80, "focus": 70},
    },
    "fergus": {
        "occupation": "innkeeper", "age": 62, "gender": "male",
        "traits": {"warmth": 80, "humor": 75, "impulsivity": 50,
                  "bravery": 50, "focus": 40},
    },
    "kragan": {
        "occupation": "bandit", "age": 45, "gender": "male",
        "traits": {"warmth": 20, "humor": 15, "impulsivity": 40,
                  "bravery": 55, "focus": 60},
    },
    # M10: town-archetype representatives, pulled forward from M11's
    # planned cast (docs/milestones/m11.md). Unlike bram/fergus/kragan
    # these aren't individually-named display characters -- the
    # compositional prompt schema (P:/D:/OCC:/R:/M:/C:/EV:|) has no name
    # token at all, so a "character" entry here exists purely to generate
    # real (OCC:<occupation> D:<descriptor>) training coverage for that
    # occupation; the actual in-game instances are spawnInstance()'s
    # unlimited seed-jittered output (NPCDatabase::PUB_PATRON_ARCHETYPE
    # etc.), each with its own generated name/age/gender/traits, not this
    # one fixed profile. Traits verified against personality_descriptor()
    # (trainer/tests/test_cast_corpus.py locks this in) and deliberately
    # span genders/ages the existing all-male bram/fergus/kragan cast
    # doesn't cover.
    "patron_rep": {
        "occupation": "pub_patron", "age": 28, "gender": "female",
        "traits": {"warmth": 75, "humor": 65, "impulsivity": 50,
                  "bravery": 55, "focus": 35},
    },
    "smith_rep": {
        "occupation": "blacksmith", "age": 40, "gender": "female",
        "traits": {"warmth": 30, "humor": 25, "impulsivity": 25,
                  "bravery": 80, "focus": 78},
    },
    "tinker_rep": {
        "occupation": "wizard", "age": 68, "gender": "male",
        "traits": {"warmth": 60, "humor": 55, "impulsivity": 30,
                  "bravery": 45, "focus": 70},
    },
    "folk_rep": {
        "occupation": "villager", "age": 72, "gender": "female",
        "traits": {"warmth": 65, "humor": 70, "impulsivity": 55,
                  "bravery": 40, "focus": 50},
    },
    # M11: Briar Glen's General Store keeper and Herbalist
    # (docs/ideas-briar-glen-world.md). Same "representative point inside
    # NPCDatabase's archetype trait box" pattern as the M10 reps above --
    # midpoints of MERCHANT_ARCHETYPE/HEALER_ARCHETYPE's ranges
    # (game/src/user/NPCDatabase.cpp), verified to land on "measured"/
    # "gentle" respectively (trainer/tests/test_cast_corpus.py).
    "merchant_rep": {
        "occupation": "merchant", "age": 50, "gender": "male",
        "traits": {"warmth": 55, "humor": 42, "impulsivity": 24,
                  "bravery": 42, "focus": 77},
    },
    "healer_rep": {
        "occupation": "healer", "age": 45, "gender": "female",
        "traits": {"warmth": 80, "humor": 42, "impulsivity": 27,
                  "bravery": 27, "focus": 72},
    },
}

# ---- DESCRIPTOR-keyed tics (personality axis, shared/reusable) ---------
# "sassy" reuses Selena's own mood-opener bank directly -- same word,
# same voice, genuine cross-axis reuse rather than new authoring.

_DESCRIPTOR_TICS = {
    "sassy": sc._OPENERS["sassy"],
    "gruff": (
        "WHAT'S YOUR BUSINESS HERE?", "IDENTIFY YOURSELF, STRANGER.",
        "KEEP MOVING, NO LOITERING.", "STATE YOUR PURPOSE.",
        "I'VE HEARD ENOUGH.", "THIS ISN'T A PLACE FOR IDLE TALK.",
        "I WON'T ASK AGAIN.", "YOU'RE TESTING MY PATIENCE.",
        "SPEAK PLAINLY, I HAVE NO TIME.", "DON'T MAKE ME REPEAT MYSELF.",
        "KEEP YOUR HANDS WHERE I CAN SEE THEM.", "MOVE ALONG.",
        "I'VE SEEN YOUR KIND BEFORE.", "NO TROUBLE, UNDERSTOOD?",
    ),
    "cheerful": (
        "HEY THERE, GOOD TO SEE YOU!", "WELCOME, WELCOME, COME ON IN!",
        "WELL, IF IT ISN'T MY FAVORITE GUEST!", "AH, A WELCOME SIGHT!",
        "GREETINGS, TRAVELLER! REST YOUR WEARY FEET!",
        "WELL NOW, LOOK WHO IT IS!", "COME IN, COME IN, OUT OF THE COLD!",
        "AH, A FACE I HAVEN'T SEEN IN AGES!",
        "GRAND TO SEE YOU, TRULY GRAND.", "NOW THEN, WHAT'LL IT BE?",
        "SIT YOURSELF DOWN, MAKE YOURSELF AT HOME.",
        "AH, GO ON NOW, DON'T BE SHY.",
    ),
    "cold": (
        "YOU DARE DISTURB ME?", "WHAT DO YOU WANT?",
        "YOU'RE WASTING MY TIME.", "YOU'RE NOT WORTHY OF MY ATTENTION.",
        "YOU'RE INTRUDING ON MY TERRITORY.", "YOU'RE GOING TO REGRET THIS.",
        "STATE YOUR BUSINESS AND LEAVE.", "I HAVE NO PATIENCE FOR THIS.",
        "YOU SHOULDN'T HAVE COME HERE.", "MAKE THIS QUICK.",
        "I DON'T REPEAT MYSELF.", "YOU'RE ALREADY TESTING ME.",
    ),
    # M10: added for the new town-archetype representative entries below
    # (wizard_rep -> "measured", villager_rep -> "playful") -- shared/
    # reusable like every other descriptor bank, not tied to one character.
    "measured": (
        "LET'S THINK THIS THROUGH FIRST.", "NO NEED TO RUSH, WE HAVE TIME.",
        "I CHOOSE MY WORDS CAREFULLY.", "EVERYTHING IN ITS OWN TIME.",
        "A STEADY HAND MAKES FEWER MISTAKES.",
        "PATIENCE. THAT'S THE WHOLE SECRET.",
        "I LIKE TO WEIGH THINGS BEFORE I SPEAK.",
        "SLOW AND CAREFUL WINS OUT, USUALLY.", "THERE'S NO HURRY HERE.",
        "BETTER TO PAUSE THAN TO STUMBLE.",
    ),
    "playful": (
        "RACE YOU TO THE OTHER SIDE!",
        "BET YOU CAN'T GUESS WHAT I'M THINKING.",
        "EVERYTHING'S MORE FUN WITH A LITTLE MISCHIEF.",
        "I NEVER TURN DOWN A GOOD GAME.", "WANT TO HEAR SOMETHING SILLY?",
        "LIFE'S TOO SHORT TO BE SERIOUS ALL THE TIME.",
        "CATCH ME IF YOU CAN!", "I MAKE UP GAMES WHEREVER I GO.",
        "A LITTLE TROUBLE NEVER HURT ANYONE.", "LET'S SEE WHO LAUGHS FIRST.",
    ),
    # M11: added for healer_rep -> "gentle" (docs/ideas-briar-glen-world.md).
    "gentle": (
        "LET ME TAKE A LOOK AT THAT FOR YOU.",
        "THERE, THERE, THE WORST IS OVER NOW.",
        "REST A MOMENT, YOU'VE EARNED IT.",
        "NO NEED TO BE BRAVE ABOUT IT WITH ME.",
        "I'VE GOT SOMETHING FOR THAT, DON'T WORRY.",
        "SLOW BREATHS, THAT'S IT.", "YOU'RE SAFE HERE, I PROMISE.",
        "LET IT HEAL IN ITS OWN TIME.", "I'M RIGHT HERE, TAKE YOUR TIME.",
        "A LITTLE KINDNESS GOES A LONG WAY.",
    ),
}

# ---- OCCUPATION-flavored insertions (job axis, shared/reusable) --------
# No entry for "villager" (Selena) -- occupation flavor is optional per
# character, not required; her voice already carries plenty via the
# reused mood-opener + descriptor-tic banks.

_OCCUPATION_FLAVOR = {
    "guard": (
        "KEEP THE PEACE, ORDER MUST PREVAIL.", "THE CITIZENS DEPEND ON ME.",
        "SUSPICIOUS CHARACTERS, STAY BACK.", "MY POST IS MY DUTY.",
        "THE GATE STAYS SHUT AFTER DARK.", "THE LAW IS STRICT HERE.",
        "I WATCH THIS ROAD DAY AND NIGHT.",
        "THE SAFETY OF THIS TOWN IS ON ME.",
        "NOTHING GETS PAST THIS WATCH.", "TROUBLE FINDS ME FIRST, ALWAYS.",
    ),
    "innkeeper": (
        "FRESH ALE, HOT STEW, WARM BEDS.", "A WARM MEAL AND A SOFT BED.",
        "FRESH BAKED BREAD, HOT COFFEE.", "A SAFE PLACE FROM THE STORM.",
        "THIS INN IS KNOWN FOR GOOD FOOD.", "A PLACE WHERE TRAVELLERS REST.",
        "THE HEARTH'S ALWAYS LIT FOR GUESTS.", "YOUR TAB CAN WAIT, SIT DOWN.",
        "BEST ALE IN THE WHOLE TOWN, I SAY SO MYSELF.",
        "THIS INN'S SEEN A THOUSAND STORIES.",
    ),
    "bandit": (
        "YOUR VALUABLES, OVER HERE.", "HAND OVER THE GOLD, NOW.",
        "SURRENDER YOUR TREASURES.", "YOUR WEALTH IS OUR GOAL.",
        "WE TAKE WHAT WE WANT HERE.", "THIS ROAD BELONGS TO US NOW.",
        "NOBODY CROSSES THIS PASS FOR FREE.", "GIVE IT UP, EASY OR HARD.",
        "MY CREW DOESN'T MISS.", "THIS HIDEOUT ISN'T ON ANY MAP.",
    ),
    # M10, pulled forward from M11's planned town cast. Kid-appropriate:
    # pub_patron reads as giddy/goofy/overly-complimentary, never
    # intoxicated or flirtatious -- deliberate choice per the game's
    # audience.
    "pub_patron": (
        "OH WOW, IS THAT A NEW CLOAK? IT'S WONDERFUL!",
        "I COULD TALK ALL NIGHT, HONESTLY, ALL NIGHT!",
        "BEST CIDER IN THE WHOLE VILLAGE, I SWEAR IT!",
        "WAIT, WAIT -- HAVE I TOLD YOU THIS ONE?",
        "YOU HAVE THE KINDEST FACE I'VE EVER SEEN.",
        "SHH, DON'T TELL ANYONE, BUT I DANCED ON THE TABLE LAST NIGHT.",
        "I'M NOT EVEN TIRED, I COULD STAY UP FOREVER!",
        "THREE MUGS OF CIDER IN AND EVERYTHING IS HILARIOUS.",
        "YOU'RE MY FAVORITE PERSON HERE, DON'T TELL THE OTHERS.",
        "I LAUGH AT EVERYTHING WHEN I'M THIS HAPPY.",
        "IS THE ROOM SPINNING OR IS IT JUST ME BEING SILLY?",
        "I COULD HUG THE WHOLE TAVERN RIGHT NOW.",
    ),
    "blacksmith": (
        "THE FORGE DOESN'T WAIT FOR ANYONE.",
        "GOOD STEEL TAKES PATIENCE. SO DO YOU, APPARENTLY.",
        "DON'T TOUCH THE ANVIL, IT'S STILL HOT.",
        "I MAKE THINGS THAT LAST. THAT'S THE WHOLE JOB.",
        "SPARKS FLY, I KEEP WORKING.",
        "A DULL BLADE IS A WASTED ONE.",
        "MY HANDS REMEMBER EVERY HAMMER STRIKE.",
        "NO SHORTCUTS IN THIS TRADE.",
        "THE FIRE'S HONEST. PEOPLE AREN'T ALWAYS.",
        "STRONG METAL, STRONG WILL. SAME RULE.",
        "I'VE FORGED THROUGH WORSE DAYS THAN THIS.",
        "KEEP YOUR FINGERS AWAY FROM THE HEAT.",
    ),
    # Friendly town tinker-wizard -- distinct voice from Shadewrath (the
    # necromancer villain, a separate full-tier character, not this
    # archetype).
    "wizard": (
        "OH! OH, WAIT, I ALMOST HAD IT THAT TIME.",
        "THIS POTION EITHER GLOWS OR EXPLODES. LET'S FIND OUT.",
        "I INVENTED SOMETHING YESTERDAY. I THINK. I FORGET.",
        "MAGIC IS JUST PATIENCE WITH SPARKLES ON TOP.",
        "DON'T MIND THE SMOKE, THAT'S NORMAL. USUALLY.",
        "I ONCE TURNED A TEAPOT INTO A FROG. LONG STORY.",
        "EVERY SPELL STARTS WITH A GOOD QUESTION.",
        "MY BEARD CAUGHT FIRE AGAIN. WORTH IT, THOUGH.",
        "THE STARS TOLD ME SOMETHING TODAY. I FORGET WHAT.",
        "COME SEE MY WORKSHOP, IT'S MOSTLY ORGANIZED.",
        "I'M CLOSE TO A BREAKTHROUGH. VERY CLOSE. PROBABLY.",
        "CURIOSITY NEVER HURT ANYONE. MOSTLY.",
    ),
    "villager": (
        "AH, A NEW FACE! WE DON'T GET MANY OF THOSE.",
        "BACK IN MY DAY, WE WALKED EVERYWHERE, UPHILL.",
        "I'VE LIVED HERE SO LONG THE ROADS KNOW MY NAME.",
        "YOU LOOK LIKE TROUBLE. GOOD. WE NEEDED SOME.",
        "MY KNEES CREAK LOUDER THAN THE OLD MILL.",
        "EVERY VILLAGE NEEDS A GOOD GOSSIP. THAT'S ME.",
        "I'VE SEEN THREE FESTIVALS AND TWO FLOODS. BUSY YEAR.",
        "YOUNG FOLKS THESE DAYS, ALWAYS IN SUCH A HURRY.",
        "I TELL THE SAME JOKE EVERY YEAR. STILL FUNNY.",
        "THIS VILLAGE RAISED ME AND I'M NOT DONE YET.",
        "COME BACK ANYTIME, I'LL HAVE MORE STORIES.",
        "WISDOM COMES WITH AGE. SO DOES COMPLAINING.",
    ),
    # M11: Briar Glen's General Store and Herbalist
    # (docs/ideas-briar-glen-world.md).
    "merchant": (
        "EVERYTHING'S PRICED FAIR, ASK ANYONE.",
        "I KEEP A LEDGER FOR A REASON.",
        "GOOD STOCK MOVES ITSELF, IF YOU LET IT.",
        "I DON'T HAGGLE, BUT I DO LISTEN.",
        "A SHOP THAT KEEPS ITS WORD KEEPS ITS CUSTOMERS.",
        "COUNT YOUR CHANGE, I DON'T MIND.",
        "I RESTOCK EVERY MORNING, RAIN OR SHINE.",
        "THE SHELVES TELL YOU WHAT SELLS. I JUST LISTEN TO THEM.",
        "NOTHING LEAVES THIS SHOP WITHOUT A FAIR PRICE.",
        "I REMEMBER EVERY REGULAR'S ORDER.",
    ),
    "healer": (
        "SIT DOWN, LET ME LOOK AT THAT.",
        "THIS ONE'S BITTER, BUT IT WORKS.",
        "THE GARDEN OUT BACK GROWS MOST OF WHAT I NEED.",
        "REST IS PART OF THE CURE, NOT JUST THE HERBS.",
        "I'VE TREATED WORSE. YOU'LL BE FINE.",
        "DRINK IT SLOWLY, NOT ALL AT ONCE.",
        "EVERY PLANT IN THIS ROOM HAS A PURPOSE.",
        "COME BACK IF IT DOESN'T EASE BY MORNING.",
        "I LEARNED MOST OF THIS FROM MY OWN MOTHER.",
        "PAIN PASSES. LET ME HELP IT PASS FASTER.",
    ),
}

# ---- per-character catchphrase banks (the only per-character content) --
# Fergus's Irish-flavored lines (M9). Kragan's added M9.2, targeting the
# specific coherence gap M9's own DoD flagged live on real hardware
# ("GOTTAND", "RECAND", "NONDS") -- see docs/milestones/m9.2.md. Both
# banks are deliberately small so they reinforce rather than dominate the
# shared-bank repetition budget the compositional mechanism relies on.

_FERGUS_CATCHPHRASES = (
    "AH, GO ON NOW, DON'T BE SHY.", "THAT'S GRAND, THAT IS.",
    "SIT YOURSELF DOWN, LAD.", "NOW THEN, TELL ME A TALE.",
    "A ROUND FOR THE HOUSE, WHY NOT.", "GOOD FOOD AND GOOD COMPANY, THAT'S THE LIFE.",
    "GRAND DAY FOR IT, ISN'T IT.", "COME IN FROM THE COLD, GO ON.",
)

_KRAGAN_CATCHPHRASES = (
    "MY BLADE DOESN'T MISS.", "THIS PASS BELONGS TO ME NOW.",
    "NOBODY LEAVES WITHOUT PAYING.", "I'VE BURIED BETTER THAN YOU.",
    "SPEAK FAST, MY PATIENCE IS SHORT.", "THE SHADOWS ARE MY ONLY FRIENDS.",
    "CROSS ME ONCE, REGRET IT FOREVER.", "GOLD OR BLOOD, YOUR CHOICE.",
)

_CATCHPHRASES = {"fergus": _FERGUS_CATCHPHRASES, "kragan": _KRAGAN_CATCHPHRASES}

# ---- gossip (M11 section 2, docs/milestones/m11.md) ---------------------
# The town-gossip mechanism: a player-caused event (reaching max trust
# with Shadewrath or Korrath, game/src/user/DialogueDemo.cpp's real
# trigger) publishes to WorldState::currentGossip(); NpcService::eventFor()
# routes it into EV: for occupations trained to react to it secondhand.
# MUST match game/src/user/WorldState.cpp's GOSSIP_EVENTS exactly --
# these are the only EV: values a gossip-hub occupation was ever shown
# during training, so any other tag reaching them is out-of-distribution.
GOSSIP_EVENTS = ("shadewrath_allied", "korrath_pleaded")

# Only these two occupations' corpora get gossip content -- pub_patron and
# villager are the town's natural gossip hubs ("EVERY VILLAGE NEEDS A GOOD
# GOSSIP. THAT'S ME." is already villager's own stock line above). Every
# other occupation keeps reacting only to its own direct EVENTS_FOR_CONTEXT
# events, same as before this section existed.
GOSSIP_HUB_OCCUPATIONS = frozenset({"pub_patron", "villager"})

# Secondhand reactions -- deliberately phrased as hearsay ("I HEARD",
# "THEY SAY", "WORD IS"), not firsthand experience, since these
# characters didn't witness the event themselves.
_GOSSIP_LINES = {
    "shadewrath_allied": (
        "DID YOU HEAR? THE NECROMANCER OFFERED SOME KIND OF ALLIANCE. GIVES ME CHILLS.",
        "THEY SAY HE MADE AN OFFER TO SOMEONE, OF ALL THINGS. AN ALLIANCE.",
        "WORD IS THE NECROMANCER WANTS TO TALK TERMS NOW. NEVER THOUGHT I'D SEE THE DAY.",
        "SOMEONE TOLD ME THE NECROMANCER PROPOSED AN ALLIANCE. STRANGE TIMES, THESE.",
        "I HEARD HE ISN'T FIGHTING ANYMORE. OFFERING DEALS INSTEAD. CAN YOU IMAGINE?",
    ),
    "korrath_pleaded": (
        "I HEARD THE BOUND KNIGHT ASKED SOMEONE FOR HELP. POOR SOUL.",
        "THEY SAY HE BEGGED TO BE FREED. IMAGINE CARRYING THAT BURDEN SO LONG.",
        "WORD IS THE KNIGHT FINALLY SPOKE HIS TRUE WISH. IT BREAKS MY HEART.",
        "SOMEONE TOLD ME THE BOUND KNIGHT ASKED FOR HIS FREEDOM. I HOPE HE FINDS PEACE.",
        "I HEARD HE'S NOT JUST STANDING GUARD ANYMORE. HE ASKED FOR A WAY OUT.",
    ),
}

# Fraction of a gossip-hub character's draws that carry a gossip EV: tag
# instead of a direct EVENTS_FOR_CONTEXT one -- high enough for the model
# to actually learn the association (guard/cast density precedent), low
# enough that direct-event coverage for these two occupations isn't
# starved.
GOSSIP_FRACTION = 0.3

# ---- relationship-tier closers (shared/reusable) ------------------------

_TIER_CLOSERS = {
    "stranger": (
        "I DON'T RECOGNIZE YOU.", "YOU'RE A NEW FACE.",
        "I'VE NEVER SEEN YOU BEFORE.", "IT'S NICE TO MEET YOU.",
        "I'LL REMEMBER YOU.",
    ),
    "acquaintance": (
        "LONG TIME NO SEE.", "NICE TO SEE YOU AGAIN.",
        "WE'VE MET BEFORE, HAVEN'T WE?", "I REMEMBER OUR LAST TALK.",
        "WE'VE SHARED A FEW MOMENTS.",
    ),
    "neutral": (
        "HELLO AGAIN.", "NICE TO SEE YOU.", "WE'VE CROSSED PATHS BEFORE.",
        "I'VE SEEN YOU AROUND.", "LET'S SEE WHERE THIS GOES.",
    ),
    "friend": (
        "GREAT TO SEE YOU.", "IT'S ALWAYS A PLEASURE.",
        "YOU'RE ALWAYS WELCOME HERE.", "I'M GLAD WE'RE FRIENDS.",
        "WE'VE GOT A GOOD THING GOING.",
    ),
    "close_friend": (
        "MISSED YOU.", "I'M HERE FOR YOU.", "WE'VE GOT A STRONG BOND.",
        "YOU'RE OFTEN ON MY MIND.", "GOOD TO HAVE YOU BACK.",
    ),
    "best_friend": (
        "YOU KNOW YOU CAN ALWAYS COUNT ON ME.", "I'LL BE THERE FOR YOU.",
        "WE'VE GOT AN UNBREAKABLE BOND.", "YOU'RE PART OF THE FAMILY HERE.",
        "I'LL ALWAYS BE HERE FOR YOU.",
    ),
}


def _response(rng: random.Random, name: str, descriptor: str, occupation: str,
             tier: str, mood: str, context: str, crossed_descriptor: str | None,
             gossip_tag: str | None = None) -> str:
    """One line: mood-opener + descriptor-tic + body + occupation-flavor +
    (Fergus-only) catchphrase + tier-closer, each included probabilistically
    -- same discipline as selena_corpus._response()'s fixed draw order.
    gossip_tag (M11): when set, a secondhand-reaction line is appended
    UNCONDITIONALLY, not probabilistically like the other clauses -- this
    combo's entire purpose is teaching EV:<gossip_tag> -> this content, so
    it needs strong signal, not a coin flip that could omit it entirely."""
    parts = []
    if rng.random() < 0.5:
        parts.append(rng.choice(sc._OPENERS[mood]))
    tic_descriptor = crossed_descriptor if crossed_descriptor else descriptor
    if rng.random() < 0.6:
        parts.append(rng.choice(_DESCRIPTOR_TICS[tic_descriptor]))
    body = sc._fill(rng, context, rng.choice(sc._BODIES[context]))
    parts.append(body)
    if occupation in _OCCUPATION_FLAVOR and rng.random() < 0.4:
        parts.append(rng.choice(_OCCUPATION_FLAVOR[occupation]))
    if name in _CATCHPHRASES and rng.random() < 0.3:
        parts.append(rng.choice(_CATCHPHRASES[name]))
    if gossip_tag:
        parts.append(rng.choice(_GOSSIP_LINES[gossip_tag]))
    if rng.random() < 0.35:
        parts.append(rng.choice(_TIER_CLOSERS[tier]))
    return " ".join(parts)


# (occupation, descriptor) pairs deliberately EXCLUDED from every crossed
# example below -- the actual falsification test for compositional
# generalization (docs/milestones/m9.md Data Science Review: "the
# capacity-dilution hypothesis needs its own falsification test"). If the
# model can produce plausible text for these it has genuinely never seen,
# that's real evidence OCC:/D: are learned as separable axes, not just
# memorized per-character pairings.
HOLDOUT_COMBOS = frozenset({
    ("innkeeper", "sassy"), ("bandit", "gruff"), ("guard", "cheerful"),
})


def holdout_pairs() -> list[tuple[str, str]]:
    return sorted(HOLDOUT_COMBOS)


def assert_no_holdout_leak(pairs: list[tuple[str, str]]) -> None:
    """Safety net: fails loudly if any generated line's (occupation,
    descriptor) combo matches a held-out pair -- mirrors m9_corpus.py's
    same-named check for attempt #1's corpus."""
    leaked = []
    for prompt, _ in pairs:
        occ = prompt.split("OCC:")[1].split(" ")[0]
        d = prompt.split("D:")[1].split(" ")[0]
        if (occ, d) in HOLDOUT_COMBOS:
            leaked.append((occ, d))
    if leaked:
        raise AssertionError(
            f"{len(leaked)} corpus lines leaked from held-out combos: "
            f"{sorted(set(leaked))} -- capacity-check gate would be invalid")


def generate_pairs(seed: int = 0, per_combo: int = 3,
                   cross_fraction: float = 0.2,
                   combo_count: int | None = None) -> list[tuple[str, str]]:
    """per_combo pairs for each character x tier x mood x context combo
    (6 tiers x 5 moods x 8 contexts = 240 combos/character -- per_combo=3
    lands at ~125-146K chars/character, matching guard's own proven
    ~123K/instance benchmark, docs/milestones/m9.md section 4). Real
    repetition comes from the shared phrase banks (~10-15 items each)
    getting reused across all 240 combos, not from repeating one exact
    combo many times the way Selena's single-character corpus does.

    combo_count (m9.1 density-structure experiment, docs/milestones/
    m9.1.md): if set, uses only this many combos per character (a seeded
    random subset of the 240) instead of all of them -- guard's own
    corpus hits its ~123K/instance density via a SMALL combo space
    repeated MANY times (GUARD_PER_COMBO=24 across ~45 combos/instance),
    not cast_corpus's default of many combos repeated few times. This
    isolates which structure actually drives coherence: total character
    volume, or how many times the exact same combo repeats. Caller
    should raise per_combo to hold total volume roughly constant when
    narrowing combo_count.

    cross_fraction of draws use a DIFFERENT descriptor's tic bank than the
    character's own, with D: relabeled to match -- deliberate axis-crossing
    so OCC: and D: don't perfectly correlate in the training data
    (docs/milestones/m9.md section 4's "why this matters" note). Draws
    landing on a HOLDOUT_COMBOS pair are skipped (no crossing that line)
    so those combos never appear in training -- generalization_check()
    (make_m9_blob.py) probes them afterward."""
    rng = random.Random(seed)
    descriptors = list(_DESCRIPTOR_TICS.keys())
    pairs = []
    for name, profile in CHARACTERS.items():
        descriptor = personality_descriptor(profile["traits"])
        occupation = profile["occupation"]
        combos = [(t, m, c) for t in _TIER_MIDPOINT for m in sc.MOODS
                 for c in sc.CONTEXTS]
        if combo_count is not None:
            # Python's hash() on strings is randomized per-process by
            # default -- must not be used where a reproducible seed is
            # required (same lesson as make_m9_blob.py's generalization_
            # check()). A trivial deterministic checksum instead.
            name_checksum = sum(ord(c) for c in name)
            subset_rng = random.Random(seed + name_checksum)
            combos = subset_rng.sample(combos, min(combo_count, len(combos)))
        for _ in range(per_combo):
            for tier, mood, context in combos:
                crossed = None
                if rng.random() < cross_fraction:
                    other = [d for d in descriptors if d != descriptor
                            and (occupation, d) not in HOLDOUT_COMBOS]
                    crossed = rng.choice(other)
                gossip_tag = None
                if occupation in GOSSIP_HUB_OCCUPATIONS and rng.random() < GOSSIP_FRACTION:
                    gossip_tag = rng.choice(GOSSIP_EVENTS)
                    event = gossip_tag
                else:
                    event = rng.choice(sc.EVENTS_FOR_CONTEXT[context])
                relationship = _relationship_state(tier)
                prompt = prompt_fields(profile, relationship, mood, context, event)
                if crossed:
                    # Relabel D: to match the crossed descriptor -- the
                    # response text below is drawn from THAT descriptor's
                    # tic bank, so prompt and response must agree, or this
                    # would teach the model D: is unreliable noise instead
                    # of teaching it OCC:/D: are independently composable.
                    prompt = prompt.replace(f"D:{descriptor} ", f"D:{crossed} ")
                response = _response(rng, name, descriptor, profile["occupation"],
                                     tier, mood, context, crossed, gossip_tag)
                pairs.append((prompt, response))
    return pairs


def corpus_text(seed: int = 0, per_combo: int = 3) -> str:
    return "".join(p + r for p, r in generate_pairs(seed, per_combo))


def combo_key(prompt: str) -> tuple:
    """Parses (person, descriptor, occupation, tier, mood, context) back
    out of a prompt string -- mirrors selena_corpus.combo_key()."""
    fields = {}
    for tok in prompt.rstrip("|").split(" "):
        k, _, v = tok.partition(":")
        fields[k] = v
    return (fields["P"], fields["D"], fields["OCC"], fields["R"],
           fields["M"], fields["C"])
