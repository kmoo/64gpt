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
voices. The only per-character content is a small catchphrase bank
(Fergus's Irish-flavored lines). Reuses selena_corpus.py's mood-keyed
OPENERS and context-keyed BODIES directly rather than re-authoring them
-- "sassy" openers work for any sassy character, not just Selena.

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
}

# ---- Fergus's catchphrases (the ONLY per-character bank, kept small) ---

_FERGUS_CATCHPHRASES = (
    "AH, GO ON NOW, DON'T BE SHY.", "THAT'S GRAND, THAT IS.",
    "SIT YOURSELF DOWN, LAD.", "NOW THEN, TELL ME A TALE.",
    "A ROUND FOR THE HOUSE, WHY NOT.", "GOOD FOOD AND GOOD COMPANY, THAT'S THE LIFE.",
    "GRAND DAY FOR IT, ISN'T IT.", "COME IN FROM THE COLD, GO ON.",
)

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
             tier: str, mood: str, context: str, crossed_descriptor: str | None) -> str:
    """One line: mood-opener + descriptor-tic + body + occupation-flavor +
    (Fergus-only) catchphrase + tier-closer, each included probabilistically
    -- same discipline as selena_corpus._response()'s fixed draw order."""
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
    if name == "fergus" and rng.random() < 0.3:
        parts.append(rng.choice(_FERGUS_CATCHPHRASES))
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
                   cross_fraction: float = 0.2) -> list[tuple[str, str]]:
    """per_combo pairs for each character x tier x mood x context combo
    (6 tiers x 5 moods x 8 contexts = 240 combos/character -- per_combo=3
    lands at ~125-146K chars/character, matching guard's own proven
    ~123K/instance benchmark, docs/milestones/m9.md section 4). Real
    repetition comes from the shared phrase banks (~10-15 items each)
    getting reused across all 240 combos, not from repeating one exact
    combo many times the way Selena's single-character corpus does.

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
        for _ in range(per_combo):
            for tier, mood, context in combos:
                crossed = None
                if rng.random() < cross_fraction:
                    other = [d for d in descriptors if d != descriptor
                            and (occupation, d) not in HOLDOUT_COMBOS]
                    crossed = rng.choice(other)
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
                                     tier, mood, context, crossed)
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
