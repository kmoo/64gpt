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
        "YOU RETURNED. I HARDLY DARED BELIEVE YOU WOULD.",
        "EACH VISIT IS A SMALL VICTORY, AND I TREASURE EVERY ONE.",
        "I HAD NEARLY FORGOTTEN THE SOUND OF GOOD NEWS.",
        "SOMETHING IN ME KINDLES WHEN THAT DOOR OPENS AND IT IS YOU.",
        "TODAY FEELS BRIGHTER. I HAVE LEARNED NOT TO QUESTION SUCH GIFTS.",
        "I HAVE COUNTED THE DAYS, AND THIS ONE SHINES AMONG THEM.",
        "YOU CANNOT KNOW WHAT YOUR RETURNING MEANS TO ME. TRULY.",
        "I PERMITTED MYSELF TO HOPE TODAY. IT FELT PERILOUS AND WONDERFUL.",
        "I FOUND MYSELF SMILING BEFORE I KNEW WHY. THEN CAME YOUR STEP.",
        "THIS IS THE FINEST HOUR OF MY WEEK, EVERY TIME WITHOUT FAIL.",
        "I DID NOT THINK I WOULD FEEL SO LIGHT OF HEART AGAIN.",
        "YOU CARRY SOMETHING I CANNOT QUITE NAME. HOPE, PERHAPS.",
        "I ALLOWED MYSELF GLADNESS TODAY. RARER THAN IT OUGHT TO BE.",
        "I HAVE BEEN SMILING SINCE YOUR FOOTSTEPS REACHED THIS DOOR.",
        "SOMETHING GRACIOUS STIRS TODAY. I CAN FEEL IT IN THE AIR.",
        "YOU MAKE THIS CHAMBER FEEL SMALLER AND THE WORLD BEYOND FEEL VAST.",
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
        "I HEARD HIS VOICE THROUGH THE WALLS AGAIN LAST NIGHT.",
        "SOMETHING FEELS DIFFERENT TODAY AND I DON'T TRUST DIFFERENT.",
        "PLEASE, BE QUICK. I DON'T WANT HIM TO KNOW YOU WERE HERE.",
        "I'VE LEARNED TO FEAR THE QUIET MORE THAN THE SHOUTING.",
        "WHAT IF HE FINDS OUT ABOUT THIS? ABOUT US TALKING LIKE THIS?",
        "I DON'T SLEEP WELL ANYMORE. THERE'S ALWAYS SOMETHING TO LISTEN FOR.",
        "I'M SCARED THIS IS TOO GOOD TO LAST.",
        "SOMETHING IN THE AIR FEELS WRONG TODAY. I CAN'T EXPLAIN IT BETTER THAN THAT.",
    ),
    "sassy": (
        "OH. SO YOU GRACE ME WITH YOUR PRESENCE AT LAST.",
        "A PRINCESS IN CHAINS IS STILL A PRINCESS. DO MIND YOUR MANNERS.",
        "I HAVE ENDURED WORSE COMPANY. THOUGH NOT, I CONFESS, BY MUCH.",
        "CAPTIVITY HAS NOT COST ME MY GRACE. HE'D LOVE IT IF IT HAD.",
        "I MAY BE IMPRISONED, BUT I AM NOT WITHOUT DIGNITY.",
        "I WOULD CURTSY PROPERLY, BUT THE IRONS RATHER SPOIL THE EFFECT.",
        "YOU ARE LATE. A PRINCESS NOTICES SUCH THINGS, YOU KNOW.",
        "IMPRESSIVE. YOU FOUND THE ONE CHAMBER HE GUARDS MOST JEALOUSLY.",
        "AH. MY FAVORED VISITOR RETURNS. I HAD BEGUN TO WONDER.",
        "A CELL FOR A HOME, AND STILL FINER BRED THAN HALF THE COURT I LEFT.",
        "ONE WOULD THINK CAPTIVITY MIGHT HUMBLE ME. ONE WOULD BE MISTAKEN.",
        "I HAVE MASTERED THE ART OF LOOKING REGAL EVEN IN IRONS.",
        "PAY THE COBWEBS NO MIND. I HAVE CHOSEN TO CALL THEM DECOR.",
        "YOU KEEP SLIPPING PAST HIS GUARD. HOW CARELESS OF HIM.",
        "I WOULD OFFER YOU TEA, HAD THIS PLACE ANY CUPS, OR COMFORT, OR EXITS.",
        "A PRINCESS STILL, EVEN HERE. I'D THANK YOU NOT TO FORGET IT.",
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
        "I DREAM ABOUT HOME MORE THAN I ADMIT, EVEN TO MYSELF.",
        "YOU LISTEN LIKE IT MATTERS WHAT I SAY. THAT'S RARER THAN YOU KNOW.",
        "I DIDN'T THINK ANYONE WOULD COME LOOKING. I'M GLAD I WAS WRONG ABOUT THAT TOO.",
        "SOMETIMES I FORGET WHAT MY OWN VOICE SOUNDS LIKE WHEN IT'S NOT AFRAID.",
        "YOU'VE GIVEN ME SOMETHING I THOUGHT I'D LOST. HOPE, I THINK IT'S CALLED.",
        "I DON'T KNOW HOW TO THANK YOU FOR TREATING ME LIKE A PERSON AND NOT A PRIZE.",
        "MY FAMILY FEELS SO FAR AWAY SOME DAYS. YOU MAKE THE DISTANCE FEEL SMALLER.",
        "I HOLD ONTO LITTLE THINGS DOWN HERE. YOUR VISITS ARE THE BIGGEST LITTLE THING.",
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
        "OH, IGNORE ME, I GET LIKE THIS WHEN I'M NERVOUS.",
        "I DIDN'T MEAN TO CRY JUST THEN. IT SNUCK UP ON ME.",
        "THAT WAS MORE HONEST THAN I MEANT TO BE. FORGIVE ME.",
        "I SOUND SILLY WHEN I TALK ABOUT HOME. I KNOW.",
        "PLEASE FORGET I SAID THAT. IT WAS THE HOPE TALKING.",
        "I'M NOT USUALLY THIS OPEN WITH ANYONE. I DON'T KNOW WHAT CAME OVER ME.",
        "THAT WAS EMBARRASSINGLY SENTIMENTAL OF ME. SORRY.",
        "I GET FLUSTERED WHEN SOMEONE'S ACTUALLY KIND TO ME. IT'S BEEN A WHILE.",
    ),
}

# ---- BODY: context-specific lines, always included ---------------------
# Not a fighter -- her "combat-banter" is encouragement from behind the
# chamber door, not her own fighting; "damage-taken" is HER reaction to
# the PLAYER being hurt, same convention shadewrath_corpus/korrath_corpus
# use for a non-combatant/observer character.

_BODIES = {
    "greeting": (
        "YOU HAVE FOUND YOUR WAY DOWN TO ME ONCE MORE. I AM GRATEFUL.",
        "I HEAR FOOTSTEPS AND HOPE, EACH TIME, THAT THEY ARE YOURS.",
        "THE GUARD PERMITTED YOU THROUGH. THAT IS NO SMALL THING.",
        "I HAVE KEPT ONLY MY OWN THOUGHTS FOR COMPANY SINCE YOU LEFT.",
        "WELCOME BACK TO MY VERY SMALL KINGDOM.",
        "YOU LOOK WEARY. BE SEATED, IF THIS PLACE WILL PERMIT IT.",
        "YOU CAME BACK. I CONFESS I ALWAYS HALF-FEAR YOU WILL NOT.",
        "I HAVE COUNTED THE HOURS SINCE YOUR LAST VISIT. I ALWAYS DO.",
        "THIS ROOM FEELS FAR LESS A CAGE WHEN YOU STAND WITHIN IT.",
        "THE KNIGHT INCLINED HIS HEAD AS YOU PASSED. HE HAS GROWN USED TO YOU.",
        "I HAVE SAVED UP SO MUCH TO TELL YOU. I ALWAYS DO.",
        "YOU LOOK WORN. REST, IF THERE IS ANYWHERE FIT LEFT TO REST.",
        "EVERY VISIT FEELS BORROWED FROM SOME KINDER LIFE.",
        "I WAS NOT CERTAIN TODAY WOULD BRING A VISITOR. I AM GLAD IT HAS.",
    ),
    "combat-banter": (
        "BE CAREFUL OUT THERE. I CAN HEAR MORE THAN YOU'D THINK FROM HERE.",
        "I WISH I COULD HELP. I'M NOT ALLOWED EVEN THAT.",
        "YOU FIGHT LIKE SOMEONE WHO HAS SOMETHING TO COME BACK FOR.",
        "I LISTEN FOR THE SOUNDS OF A FIGHT AND PRAY IT'S NOT YOU LOSING.",
        "WHATEVER'S OUT THERE, YOU'VE SURVIVED WORSE TO REACH ME.",
        "I HEARD THE CLASH OF IT ALL THE WAY DOWN HERE. I WAS TERRIFIED FOR YOU.",
        "I WISH I HAD A SWORD TO GIVE YOU, OR ANYTHING AT ALL TO HELP.",
        "YOU FIGHT LIKE SOMEONE WHO STILL BELIEVES IN SOMETHING. DON'T LOSE THAT.",
        "EVERY SOUND OF BATTLE UP THERE MAKES MY HEART STOP DOWN HERE.",
        "YOU'VE SURVIVED WORSE THAN THIS TO GET THIS FAR. REMEMBER THAT.",
        "I PRAY FOR YOU IN THE OLD WAYS, THE ONES MY MOTHER TAUGHT ME.",
        "BE CAREFUL. I DON'T HAVE MANY PEOPLE LEFT TO WORRY ABOUT.",
        "I WISH COURAGE COULD BE LENT LIKE A CLOAK. I'D GIVE YOU ALL OF MINE.",
        "COME BACK TO ME IN ONE PIECE. THAT'S ALL I ASK.",
    ),
    "item-found": (
        "THAT LOOKS LIKE SOMETHING FROM RAVENDALE, ACTUALLY.",
        "KEEP IT SAFE. NOTHING DOWN HERE IS GIVEN FREELY.",
        "I RECOGNIZE THAT CRAFTSMANSHIP. MY PEOPLE MADE THINGS LIKE THAT.",
        "SOMEONE BEFORE YOU MUST HAVE DROPPED THAT. I HOPE THEY GOT OUT.",
        "IT'S STRANGE, SEEING SOMETHING BEAUTIFUL IN A PLACE LIKE THIS.",
        "MY GRANDMOTHER HAD SOMETHING LIKE THAT, ONCE, BACK HOME.",
        "THAT'S ELVEN WORK, UNMISTAKABLY. IT MAKES ME HOMESICK JUST LOOKING AT IT.",
        "KEEP THAT CLOSE. THIS PLACE DOESN'T LET GO OF THINGS EASILY.",
        "I WONDER WHO HAD THAT BEFORE YOU. I WONDER IF THEY MADE IT OUT.",
        "IT'S ODD, FINDING BEAUTY DOWN HERE. IT FEELS LIKE A SMALL REBELLION.",
        "THAT REMINDS ME OF HOME MORE THAN I CAN EXPLAIN.",
        "HOLD ONTO IT. SOMETHING GOOD SHOULD SURVIVE THIS PLACE.",
        "MY PEOPLE VALUED CRAFT LIKE THAT ABOVE GOLD. I SEE WHY.",
        "IT'S BEAUTIFUL. I'D FORGOTTEN THINGS COULD STILL BE BEAUTIFUL.",
    ),
    "damage-taken": (
        "YOU'RE HURT. PLEASE TELL ME YOU'RE ALRIGHT.",
        "I HATE THAT I CAN'T DO ANYTHING BUT WORRY WHEN THAT HAPPENS.",
        "SIT FOR A MOMENT. YOU DON'T HAVE TO PROVE ANYTHING TO ME.",
        "EVERY TIME YOU'RE HURT I WONDER IF THIS IS TOO MUCH TO ASK OF YOU.",
        "PLEASE BE MORE CAREFUL. I DON'T KNOW WHAT I'D DO IF YOU STOPPED COMING.",
        "OH -- YOU'RE HURT. LET ME SEE, PLEASE, LET ME SEE.",
        "I WISH I HAD HERBS, BANDAGES, ANYTHING. ALL I HAVE IS WORRY.",
        "YOU DON'T HAVE TO BE STRONG FOR ME. NOT HERE. NOT EVER.",
        "I FEEL EVERY WOUND YOU CARRY LIKE IT'S MY OWN FAULT SOMEHOW.",
        "PLEASE. I CAN'T LOSE THE ONE GOOD THING LEFT IN MY LIFE.",
        "SIT DOWN BEFORE YOU FALL DOWN. I INSIST, FOR ONCE.",
        "I HATE THIS PLACE FOR WHAT IT DOES TO YOU EVERY TIME YOU COME.",
        "YOU KEEP GETTING HURT FOR ME. I DON'T KNOW HOW TO CARRY THAT.",
        "REST. JUST FOR A MOMENT. I'LL STAY RIGHT HERE WITH YOU.",
    ),
    "quiet-moment": (
        "THE QUIET DOWN HERE USED TO TERRIFY ME. NOW IT'S JUST FAMILIAR.",
        "I COUNT THE CRACKS IN THE CEILING WHEN I CAN'T SLEEP.",
        "I TRY TO REMEMBER MY MOTHER'S VOICE. IT'S HARDER THAN IT USED TO BE.",
        "SOME DAYS THE SILENCE IS ALMOST PEACEFUL. ALMOST.",
        "I'VE HAD A LOT OF TIME TO THINK DOWN HERE. TOO MUCH, PROBABLY.",
        "I'VE MEMORIZED EVERY STONE IN THIS ROOM. IT'S A STRANGE KIND OF COMPANY.",
        "I TALK TO MYSELF SOMETIMES, JUST TO REMEMBER WHAT MY VOICE SOUNDS LIKE.",
        "THE QUIET USED TO SCARE ME MORE THAN HIM. NOW I'M NOT SURE WHICH IS WORSE.",
        "I THINK ABOUT WHO I WAS BEFORE ALL THIS, SOMETIMES. SHE FEELS FAR AWAY.",
        "SITTING WITH YOU LIKE THIS IS THE CLOSEST THING TO PEACE I GET.",
        "I'VE LEARNED TO FIND SMALL COMFORTS. A CRACK OF LIGHT. A KIND VOICE.",
        "SOME NIGHTS I JUST LISTEN FOR ANY SOUND THAT ISN'T HIM.",
        "I DON'T MIND THE QUIET WHEN YOU'RE THE ONE SHARING IT WITH ME.",
        "I'VE HAD SO MUCH TIME TO THINK. MOSTLY ABOUT GETTING HOME.",
    ),
    "joke": (
        "I'VE HAD TO GET CREATIVE FOR ENTERTAINMENT DOWN HERE.",
        "THE GUARD DOESN'T LAUGH AT MY JOKES EITHER. IT'S NOT JUST YOU.",
        "THAT ACTUALLY MADE ME LAUGH. IT'S BEEN A WHILE.",
        "I'VE NAMED THE SPIDERS IN THE CORNER. DON'T JUDGE ME.",
        "CAPTIVITY HUMOR IS AN ACQUIRED TASTE. I'M SORRY IN ADVANCE.",
        "I'VE TAKEN TO NARRATING MY OWN LIFE LIKE A BAD BALLAD. IT HELPS.",
        "THE KNIGHT ALMOST SMILED ONCE. I CONSIDER THAT MY GREATEST ACHIEVEMENT HERE.",
        "THAT WAS ACTUALLY FUNNY. DON'T LET IT GO TO YOUR HEAD.",
        "I'VE GIVEN THE RATS DOWN HERE TITLES OF NOBILITY. IT PASSES THE TIME.",
        "MY SENSE OF HUMOR IS THE ONE THING HE HASN'T MANAGED TO TAKE.",
        "I LAUGH SO I DON'T DO THE OTHER THING. YOU UNDERSTAND.",
        "CAPTIVITY GIVES YOU A STRANGE SENSE OF HUMOR. I'M STILL WORKING ON MINE.",
        "THAT ONE ACTUALLY GOT A REAL LAUGH OUT OF ME. RARE THESE DAYS.",
        "I HAVE JOKES. THEY'RE ALL ABOUT THIS ROOM. THEY'RE NOT VERY GOOD.",
    ),
    "encouragement": (
        "YOU'VE COME THIS FAR. DON'T STOP NOW.",
        "I BELIEVE YOU CAN DO THIS. I DON'T SAY THAT LIGHTLY.",
        "GET UP. I'M NOT READY TO STOP HOPING YET, AND NEITHER SHOULD YOU BE.",
        "WHATEVER HAPPENED OUT THERE, YOU'RE STILL HERE. THAT COUNTS.",
        "I'VE SEEN WHAT YOU'RE CAPABLE OF. TRUST THAT, EVEN WHEN IT'S HARD.",
        "DON'T GIVE UP. I HAVEN'T, AND I'VE HAD LONGER TO WANT TO.",
        "YOU CAN DO THIS. I'VE PINNED EVERYTHING I HAVE LEFT ON THAT BELIEF.",
        "RISE. I NEED YOU TO. THAT'S SELFISH OF ME, BUT IT'S TRUE.",
        "YOU'VE CARRIED SO MUCH ALREADY. A LITTLE FURTHER, THAT'S ALL.",
        "I KNOW IT'S HARD. I KNOW. BUT YOU'RE STILL HERE, AND SO AM I.",
        "DON'T LET THIS BE WHERE IT ENDS. I'M STILL WAITING ON THE OTHER SIDE OF THIS.",
        "YOU'VE GIVEN ME EVERY REASON TO HOPE. GIVE YOURSELF THE SAME.",
        "WHATEVER THIS TAKES OUT OF YOU, I PROMISE IT MATTERS.",
        "STAND UP. FOR YOURSELF, IF NOT FOR ME. THOUGH I HOPE A LITTLE FOR ME TOO.",
    ),
    "farewell": (
        "GO CAREFULLY. I SHALL BE HERE. I AM ALWAYS HERE.",
        "COME BACK TO ME. I ASK IT PLAINLY, AS A FRIEND, NOT A PRINCESS.",
        "I WILL COUNT THE HOURS UNTIL YOUR RETURN, AS I ALWAYS DO.",
        "BE SAFE OUT THERE. YOU CARRY THE LAST OF MY HOPE WITH YOU.",
        "UNTIL NEXT WE MEET. I MEAN THAT MORE DEEPLY THAN YOU KNOW.",
        "GO SAFELY. I SHALL BE HERE, COUNTING THE MINUTES, AS EVER.",
        "PLEASE RETURN TO ME. I DARE NOT THINK WHAT IF YOU DID NOT.",
        "I WILL WAIT. IT IS MOSTLY WHAT I DO, AND I MIND IT LESS FOR YOU.",
        "TAKE GREAT CARE OF YOURSELF. YOU BEAR MORE HOPE THAN YOU KNOW.",
        "UNTIL NEXT TIME, THEN. I SHALL HOLD FAST TO THAT PROMISE.",
        "GO. AND PLEASE, COME BACK. I MEAN IT EACH AND EVERY TIME.",
        "I SHALL COUNT THE DAYS. IT IS EASIER WITH SOMETHING TO COUNT TOWARD.",
        "SAFE TRAVELS. YOU CARRY MORE OF ME WITH YOU THAN YOU REALIZE.",
        "UNTIL YOU RETURN. I SHALL BE HERE, HOPING, AS ALWAYS.",
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
        "FORGIVE MY CAUTION. IT'S KEPT ME ALIVE THIS LONG.",
        "I DON'T KNOW YET IF THIS IS KINDNESS OR ANOTHER TRICK.",
        "I'VE BEEN FOOLED BEFORE BY KIND FACES. GIVE ME TIME.",
        "TRUST IS EXPENSIVE DOWN HERE. I'M STILL DECIDING YOUR PRICE.",
        "I WANT TO BELIEVE YOU. I'M NOT THERE YET.",
        "ASK ME AGAIN, LATER. I'LL KNOW BETTER BY THEN.",
    ),
    1: (
        "HE TALKS TO HIMSELF WHEN HE THINKS NO ONE'S LISTENING. ABOUT A DOOR.",
        "THE KNIGHT WHO GUARDS ME ISN'T CRUEL. I THINK HE'S TRAPPED TOO, SOMEHOW.",
        "I'VE BEEN HERE LONG ENOUGH TO LEARN HIS PATTERNS. THAT MIGHT MATTER.",
        "I'M STARTING TO BELIEVE YOU MIGHT ACTUALLY MEAN IT.",
        "THERE'S SOMETHING BENEATH RAVENDALE HE WON'T STOP TALKING ABOUT.",
        "I HAVEN'T TOLD ANYONE ELSE THAT. I DON'T KNOW WHY I'M TELLING YOU.",
        "I'VE STARTED SAVING THINGS UP TO TELL YOU. THAT'S NEW FOR ME.",
        "THE KNIGHT BRINGS ME WATER SOMETIMES WITHOUT BEING ASKED. SMALL KINDNESSES ADD UP.",
        "I THINK HE'S AFRAID OF SOMETHING TOO. I DON'T KNOW WHAT YET.",
        "YOU'VE EARNED SOME OF MY TRUST. I DIDN'T EXPECT TO GIVE IT SO SOON.",
        "I'VE STARTED PAYING ATTENTION TO THINGS I THINK MIGHT HELP YOU.",
        "I FEEL SAFER TELLING YOU THINGS THAN I HAVE IN A LONG TIME.",
    ),
    2: (
        "PLEASE. GET ME OUT OF HERE. I CAN'T DO IT MYSELF.",
        "I TRUST YOU. THAT'S NOT A WORD I USE LIGHTLY ANYMORE.",
        "WHATEVER IT COSTS, I WANT TO GO HOME. TAKE ME WITH YOU.",
        "I'M READY. I'VE BEEN READY FOR LONGER THAN YOU KNOW.",
        "FREE ME, AND I'LL TELL YOU EVERYTHING I'VE LEARNED DOWN HERE.",
        "THANK YOU. WHATEVER HAPPENS NEXT, THANK YOU FOR NOT GIVING UP ON ME.",
        "TAKE ME HOME. I DON'T CARE WHAT IT COSTS EITHER OF US ANYMORE.",
        "I'VE TRUSTED YOU WITH EVERYTHING I HAVE LEFT. DON'T MAKE ME REGRET IT.",
        "I'M READY TO LEAVE THIS ROOM BEHIND, WHATEVER WAITS OUTSIDE IT.",
        "WHATEVER HAPPENS NEXT, I WANT IT TO HAPPEN OUTSIDE THESE WALLS.",
        "YOU'VE GIVEN ME SOMETHING I THOUGHT WAS GONE FOR GOOD: A REASON TO LEAVE.",
        "FREE ME, AND I PROMISE YOU WON'T REGRET WHAT I HAVE TO OFFER IN RETURN.",
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
