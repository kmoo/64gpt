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

# M11.1: every curated-cast/town-archetype-rep entry is an ordinary human
# the player has no pre-existing relationship with -- species/bond are
# uniform defaults here, not per-character authoring (unlike Selena/
# Shadewrath/Korrath/Elowen below, whose species/bond come from their own
# manifest bible and are set individually where each is genericized).
for _c in CHARACTERS.values():
    _c.setdefault("species", "human")
    _c.setdefault("bond", "stranger")

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
        "SAVE THE STORY, GET TO THE POINT.",
        "I DON'T REPEAT ORDERS TWICE.",
        "YOU'VE GOT ONE MINUTE. USE IT WISELY.",
        "THIS ISN'T A DEBATE.",
        "I'VE GOT NO PATIENCE FOR EXCUSES.",
        "WATCH YOUR TONE WITH ME.",
        "I DECIDE WHAT'S RELEVANT HERE, NOT YOU.",
        "KEEP IT SHORT. I'VE GOT WORK.",
        "DON'T WASTE MY DAYLIGHT.",
        "I'VE SEEN TROUBLE WALK IN LOOKING FRIENDLIER THAN YOU.",
        "SPARE ME THE DETAILS.",
        "THAT'S ENOUGH TALKING FOR ONE DAY.",
    ),
    "cheerful": (
        "HEY THERE, GOOD TO SEE YOU!", "WELCOME, WELCOME, COME ON IN!",
        "WELL, IF IT ISN'T MY FAVORITE GUEST!", "AH, A WELCOME SIGHT!",
        "GREETINGS, TRAVELLER! REST YOUR WEARY FEET!",
        "WELL NOW, LOOK WHO IT IS!", "COME IN, COME IN, OUT OF THE COLD!",
        "AH, A FACE I HAVEN'T SEEN IN AGES!",
        "GRAND TO SEE YOU, TRULY GRAND.", "NOW THEN, WHAT'LL IT BE?",
        "SIT YOURSELF DOWN, MAKE YOURSELF AT HOME.",
        "OH, WHAT A GRAND SURPRISE THIS IS!",
        "COME IN BEFORE THE WEATHER CHANGES ITS MIND!",
        "WELL AREN'T YOU A SIGHT FOR SORE EYES!",
        "THERE'S ALWAYS ROOM FOR ONE MORE FRIENDLY FACE!",
        "AH, JUST THE PERSON I HOPED TO SEE TODAY!",
        "COME, COME, TELL ME EVERYTHING!",
        "WHAT A FINE DAY FOR A VISIT, WOULDN'T YOU SAY?",
        "YOU'VE MADE MY WHOLE DAY BETTER JUST WALKING IN!",
        "NOW ISN'T THIS A PLEASANT SURPRISE!",
        "COME WARM UP, YOU LOOK CHILLED TO THE BONE!",
        "AH, GOOD COMPANY AT LAST!",
        "WELL, DON'T JUST STAND THERE, COME ON IN!",
        "I WAS JUST THINKING SOMEONE OUGHT TO VISIT!",
    ),
    "cold": (
        "YOU DARE DISTURB ME?", "WHAT DO YOU WANT?",
        "YOU'RE WASTING MY TIME.", "YOU'RE NOT WORTHY OF MY ATTENTION.",
        "YOU'RE INTRUDING ON MY TERRITORY.", "YOU'RE GOING TO REGRET THIS.",
        "STATE YOUR BUSINESS AND LEAVE.", "I HAVE NO PATIENCE FOR THIS.",
        "YOU SHOULDN'T HAVE COME HERE.", "MAKE THIS QUICK.",
        "I DON'T REPEAT MYSELF.", "YOU'RE ALREADY TESTING ME.",
        "YOUR PRESENCE IS NOT REQUESTED.",
        "I DID NOT SUMMON YOU.",
        "SPEAK QUICKLY OR NOT AT ALL.",
        "I OWE YOU NOTHING, LEAST OF ALL MY TIME.",
        "THIS AUDIENCE IS ALREADY OVER.",
        "YOU MISTAKE MY SILENCE FOR INVITATION.",
        "NOTHING HERE CONCERNS YOU.",
        "I GRANT FEW PEOPLE MY ATTENTION. YOU ARE NOT ONE OF THEM.",
        "LEAVE BEFORE I RECONSIDER MY PATIENCE.",
        "YOU'VE ALREADY OVERSTAYED YOUR WELCOME.",
        "I DO NOT EXPLAIN MYSELF TWICE.",
        "YOUR CURIOSITY IS NOT MY PROBLEM.",
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
        "LET ME CONSIDER THAT FOR A MOMENT.",
        "HASTE HAS COST ME MORE THAN PATIENCE EVER HAS.",
        "I'D RATHER BE RIGHT THAN QUICK.",
        "THERE'S A RIGHT TIME FOR EVERYTHING. THIS MAY NOT BE IT.",
        "I TAKE MY TIME. IT'S SERVED ME WELL SO FAR.",
        "LET'S NOT DECIDE ANYTHING IN A HURRY.",
        "EVERY ANSWER DESERVES A MOMENT'S THOUGHT FIRST.",
        "I'VE FOUND STILLNESS CLARIFIES MORE THAN PANIC EVER DOES.",
        "ONE STEP AT A TIME SERVES BETTER THAN A LEAP.",
        "I PREFER TO OBSERVE BEFORE I ACT.",
        "NOTHING GOOD WAS EVER RUSHED INTO BEING.",
        "I'LL GIVE YOU MY ANSWER WHEN I'VE ACTUALLY THOUGHT IT THROUGH.",
        "CAREFUL WORK OUTLASTS QUICK WORK.",
        "LET THE DUST SETTLE BEFORE WE DECIDE ANYTHING.",
    ),
    "playful": (
        "RACE YOU TO THE OTHER SIDE!",
        "BET YOU CAN'T GUESS WHAT I'M THINKING.",
        "EVERYTHING'S MORE FUN WITH A LITTLE MISCHIEF.",
        "I NEVER TURN DOWN A GOOD GAME.", "WANT TO HEAR SOMETHING SILLY?",
        "LIFE'S TOO SHORT TO BE SERIOUS ALL THE TIME.",
        "CATCH ME IF YOU CAN!", "I MAKE UP GAMES WHEREVER I GO.",
        "A LITTLE TROUBLE NEVER HURT ANYONE.", "LET'S SEE WHO LAUGHS FIRST.",
        "BET YOU A COIN I CAN GUESS YOUR NEXT MOVE.",
        "OOH, LET'S MAKE THIS INTERESTING.",
        "WANT TO PLAY A GAME? I ALREADY STARTED ONE.",
        "LAST ONE THERE BUYS SOMETHING SILLY.",
        "I COLLECT FUN LIKE OTHER PEOPLE COLLECT COINS.",
        "THIS IS GOING TO BE FUN, I CAN ALREADY TELL.",
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
        "THERE'S NO RUSH. WE'LL GET THROUGH THIS TOGETHER.",
        "YOU DON'T HAVE TO EXPLAIN, JUST REST.",
        "LET ME HELP WITH THAT, IT'S NO TROUBLE AT ALL.",
        "IT'S ALRIGHT TO NOT BE ALRIGHT FOR A WHILE.",
        "I'VE SEEN WORSE HEAL JUST FINE. YOU WILL TOO.",
        "SIT DOWN BEFORE YOU FALL DOWN, DEAR.",
        "LET ME EASE THAT FOR YOU, JUST HOLD STILL.",
        "YOU'RE DOING BETTER THAN YOU THINK YOU ARE.",
        "THERE'S NO SHAME IN NEEDING A LITTLE HELP.",
        "I'LL SIT WITH YOU AS LONG AS YOU NEED.",
        "WHATEVER HURTS, WE'LL TEND TO IT TOGETHER.",
        "GENTLY NOW, THERE'S NO NEED TO PUSH THROUGH THIS ALONE.",
        "YOU'VE CARRIED ENOUGH TODAY. LET ME CARRY SOME OF IT.",
        "REST FIRST. WORRY LATER, IF YOU MUST AT ALL.",
    ),
}

# ---- OCCUPATION-flavored insertions (job axis, shared/reusable) --------
# No entry for "villager" (Selena) -- occupation flavor is optional per
# character, not required; her voice already carries plenty via the
# reused mood-opener + descriptor-tic banks.

_OCCUPATION_FLAVOR = {
    "guard": (
        "KEEP THE PEACE, ORDER MUST PREVAIL. EASIER THAN IT SOUNDS, MOST DAYS.",
        "THE CITIZENS DEPEND ON ME. NO PRESSURE. WELL, SOME PRESSURE.",
        "SUSPICIOUS CHARACTERS, STAY BACK. YES, I'M LOOKING AT YOU. PROBABLY.",
        "MY POST IS MY DUTY. ALSO MY WHOLE PERSONALITY, LATELY.",
        "THE GATE STAYS SHUT AFTER DARK. THE GATE AND I HAVE RULES.",
        "THE LAW IS STRICT HERE. I DIDN'T WRITE IT, BUT I ENJOY QUOTING IT.",
        "I WATCH THIS ROAD DAY AND NIGHT. THE ROAD IS NOT GRATEFUL.",
        "THE SAFETY OF THIS TOWN IS ON ME. AND MY VERY COMFORTABLE BOOTS.",
        "NOTHING GETS PAST THIS WATCH. EXCEPT PIGEONS. WE'VE MADE PEACE.",
        "TROUBLE FINDS ME FIRST, ALWAYS. WE'RE PRACTICALLY OLD FRIENDS NOW.",
        "I'VE WALKED THIS WALL SO MANY NIGHTS I COULD DO IT ASLEEP. DON'T TELL.",
        "MY SHIFT DOESN'T END UNTIL THE SUN SAYS SO. WE DISAGREE OFTEN.",
        "EVERY STRANGER GETS THE SAME LOOK FROM ME. I'VE PRACTICED THE LOOK.",
        "I REPORT WHAT I SEE. TONIGHT, MOSTLY STARS AND ONE SUSPICIOUS CAT.",
        "THE TOWN SLEEPS BECAUSE I DON'T. SOMEONE HAS TO. LUCKY ME.",
        "MY BLADE STAYS SHARP. MY EYES STAY SHARPER. MY JOKES, LESS SO.",
        "NOBODY GETS THROUGH WITHOUT ME KNOWING IT. IT'S A GIFT. AND A JOB.",
        "I'VE STOPPED WORSE THAN YOU BEFORE BREAKFAST. AFTER, I'M EVEN BETTER.",
        "THIS POST HAS SEEN QUIETER YEARS. I RATHER MISS THEM, HONESTLY.",
        "I DON'T MAKE THE RULES, I JUST ENFORCE THEM. LOUDLY. WITH FLAIR.",
        "KEEP YOUR WEAPON SHEATHED WITHIN THE WALLS. MINE'S FOR SHOW. MOSTLY.",
        "THE PEACE HERE ISN'T AN ACCIDENT. IT'S A JOB. A LONG, LONG JOB.",
    ),
    "innkeeper": (
        "FRESH ALE, HOT STEW, WARM BEDS.", "A WARM MEAL AND A SOFT BED.",
        "FRESH BAKED BREAD, HOT COFFEE.", "A SAFE PLACE FROM THE STORM.",
        "THIS INN IS KNOWN FOR GOOD FOOD.", "A PLACE WHERE TRAVELLERS REST.",
        "THE HEARTH'S ALWAYS LIT FOR GUESTS.", "YOUR TAB CAN WAIT, SIT DOWN.",
        "BEST ALE IN THE WHOLE TOWN, I SAY SO MYSELF.",
        "THIS INN'S SEEN A THOUSAND STORIES.",
        "THE UPSTAIRS ROOMS ARE QUIET, JUST THE WAY TRAVELLERS LIKE THEM.",
        "NOBODY LEAVES MY TABLE STILL HUNGRY.",
        "I'VE POURED ALE FOR KINGS AND BEGGARS ALIKE, SAME SMILE FOR BOTH.",
        "THE FIRE'S BEEN GOING SINCE BEFORE DAWN, COME WARM YOURSELF.",
        "MY STEW RECIPE'S OLDER THAN THIS BUILDING.",
        "EVERY GOOD STORY STARTS AT A BAR LIKE MINE.",
        "I'VE GOT A ROOM FOR ANYONE WHO NEEDS ONE, PAYING OR NOT, TONIGHT.",
        "THE REGULARS SAY THIS PLACE FEELS LIKE HOME. I TRY TO KEEP IT THAT WAY.",
        "NOTHING ENDS A HARD DAY BETTER THAN A FULL PLATE.",
        "I'VE HEARD EVERY RUMOR IN THIS TOWN FIRST, RIGHT HERE AT THIS BAR.",
        "COME BACK ANYTIME, THE DOOR'S NEVER LOCKED TO A FRIENDLY FACE.",
        "MY GRANDMOTHER RAN THIS PLACE BEFORE ME. I DON'T PLAN ON LETTING HER DOWN.",
    ),
    "bandit": (
        "YOUR VALUABLES, OVER HERE.", "HAND OVER THE GOLD, NOW.",
        "SURRENDER YOUR TREASURES.", "YOUR WEALTH IS OUR GOAL.",
        "WE TAKE WHAT WE WANT HERE.", "THIS ROAD BELONGS TO US NOW.",
        "NOBODY CROSSES THIS PASS FOR FREE.", "GIVE IT UP, EASY OR HARD.",
        "MY CREW DOESN'T MISS.", "THIS HIDEOUT ISN'T ON ANY MAP.",
        "EMPTY YOUR POCKETS OR EMPTY YOUR LUCK.",
        "THIS ISN'T A NEGOTIATION.",
        "EVERYONE PAYS THE TOLL ONE WAY OR ANOTHER.",
        "I'VE ROBBED BETTER-ARMED FOOLS THAN YOU.",
        "NOBODY REMEMBERS A COWARD, BUT NOBODY FORGETS A BANDIT.",
        "THIS ROAD HAS A PRICE. YOU'RE PAYING IT TODAY.",
        "HAND IT OVER AND WALK AWAY, THAT'S THE DEAL.",
        "MY CREW'S WATCHING. THINK CAREFULLY.",
        "I DON'T ASK TWICE, AND I DON'T ASK NICELY EITHER.",
        "WHAT'S YOURS IS MINE THE MOMENT YOU STEP ON THIS ROAD.",
        "RUN IF YOU WANT. I'M FASTER THAN I LOOK.",
        "THIS PASS HAS A TOLL COLLECTOR NOW. THAT'S ME.",
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
        "I KNOW A GREAT STORY BUT I FORGOT THE ENDING, ISN'T THAT FUNNY?",
        "THIS IS THE BEST NIGHT I'VE HAD IN, HONESTLY, DAYS!",
        "I MADE THREE NEW FRIENDS TONIGHT AND I DON'T REMEMBER THEIR NAMES!",
        "EVERYONE HERE IS SO WONDERFUL, HAVE I SAID THAT YET?",
        "I COULD DANCE RIGHT NOW, SHOULD I DANCE RIGHT NOW?",
        "THIS MUG IS MY NEW BEST FRIEND, DON'T TELL THE OTHER MUG.",
        "I HAVE SO MANY FEELINGS AND THEY'RE ALL GOOD ONES!",
        "WAIT, DID I ALREADY TELL YOU THIS STORY? TELL ME IF I DID, I'LL TELL IT AGAIN ANYWAY.",
        "EVERYTHING TONIGHT FEELS LIKE A CELEBRATION!",
        "I LOVE THIS TAVERN, I LOVE THIS TOWN, I LOVE EVERYTHING RIGHT NOW!",
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
        "I'VE NEVER SOLD A BLADE I WOULDN'T CARRY MYSELF.",
        "MY FATHER TAUGHT ME THIS FORGE. I'LL TEACH IT TO SOMEONE SOMEDAY TOO.",
        "HEAT IT, SHAPE IT, COOL IT, TRUST IT. THAT'S THE WHOLE CRAFT.",
        "A GOOD BLADE OUTLIVES THE HAND THAT FIRST HELD IT.",
        "I DON'T RUSH GOOD STEEL. GOOD STEEL DOESN'T RUSH BACK.",
        "EVERY SCAR ON THESE HANDS TAUGHT ME SOMETHING.",
        "THE ANVIL DOESN'T LIE ABOUT SLOPPY WORK.",
        "BRING ME SOMETHING BROKEN, I'LL BRING IT BACK BETTER.",
        "I'VE FORGED WEAPONS FOR HEROES AND HORSESHOES FOR FARMERS, SAME CARE FOR BOTH.",
        "THIS FORGE HAS OUTLASTED THREE ROOFS AND ONE FIRE. IT'S NOT GOING ANYWHERE.",
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
        "I ONCE TRIED TO SPEAK WITH A CLOUD. IT DIDN'T ANSWER, BUT I LEARNED SOMETHING.",
        "MY SPELLBOOK HAS MORE NOTES IN THE MARGINS THAN ACTUAL SPELLS AT THIS POINT.",
        "I FAILED SEVENTEEN TIMES BEFORE THIS POTION WORKED. THAT'S PROGRESS, ACTUALLY.",
        "THE UNIVERSE IS MOSTLY QUESTIONS, I'VE JUST GOT MORE OF THEM THAN MOST.",
        "I TALK TO MY INGREDIENTS. THEY DON'T TALK BACK. YET.",
        "EVERY EXPLOSION TEACHES YOU SOMETHING, USUALLY 'DON'T DO THAT AGAIN.'",
        "I KEEP A JOURNAL OF EVERYTHING THAT'S GONE WRONG. IT'S VERY THICK.",
        "MAGIC REWARDS THE PATIENT AND SINGES THE IMPATIENT. I'VE BEEN BOTH.",
        "MY APPRENTICESHIP TAUGHT ME MORE ABOUT CLEANING THAN SPELLCASTING.",
        "I ONCE ALMOST UNDERSTOOD WHY THE STARS DO WHAT THEY DO. ALMOST.",
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
        "I REMEMBER WHEN THIS WHOLE SQUARE WAS JUST A MUDDY FIELD.",
        "EVERY GENERATION THINKS THEY INVENTED TROUBLE. THEY DIDN'T.",
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
        "I DON'T SELL ANYTHING I WOULDN'T BUY MYSELF.",
        "A FAIR DEAL TODAY BRINGS A RETURNING CUSTOMER TOMORROW.",
        "MY SUPPLIERS KNOW ME BY NAME AND BY MY WORD.",
        "EVERY ITEM ON THESE SHELVES HAS A STORY, ASK IF YOU'RE CURIOUS.",
        "I TRACK EVERY COIN THAT COMES THROUGH THIS DOOR.",
        "UNDERSELLING ISN'T HONEST. NEITHER IS OVERCHARGING.",
        "I'VE BUILT THIS TRADE ON TRUST, NOT LUCK.",
        "MY SCALES ARE HONEST. ASK ANYONE WHO'S CHECKED.",
        "A GOOD MERCHANT REMEMBERS WHAT YOU NEEDED LAST TIME.",
        "BUSINESS IS SLOW SOME DAYS. HONESTY ISN'T NEGOTIABLE ON ANY OF THEM.",
        "I'D RATHER LOSE A SALE THAN A CUSTOMER'S TRUST.",
        "EVERYTHING HERE IS PRICED TO MOVE, FAIRLY, EVERY TIME.",
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
        "I'VE SET MORE BONES THAN I CAN COUNT, EACH ONE MENDS DIFFERENT.",
        "MY MOTHER TAUGHT ME THESE REMEDIES. HER MOTHER TAUGHT HER.",
        "PAIN TEACHES PATIENCE, BUT MEDICINE SPEEDS IT ALONG A BIT.",
        "I KEEP MY GARDEN STOCKED FOR WHATEVER COMES THROUGH THAT DOOR.",
        "THE BODY KNOWS HOW TO HEAL. I JUST HELP IT ALONG.",
        "I'VE NEVER TURNED AWAY SOMEONE WHO NEEDED TENDING, COIN OR NOT.",
        "SOME WOUNDS NEED HERBS. SOME JUST NEED SOMEONE TO SIT WITH THEM.",
        "I LEARNED MORE FROM MY FAILURES THAN MY SUCCESSES, EARLY ON.",
        "REST IS THE MEDICINE MOST PEOPLE SKIP.",
        "I'VE TENDED WORSE THAN THIS AND SENT THEM HOME WALKING.",
        "EVERY SCAR I'VE HELPED HEAL TAUGHT ME SOMETHING NEW.",
        "MY DOOR'S ALWAYS OPEN, DAY OR NIGHT, FOR ANYONE HURTING.",
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
    "AH, YOU'RE A SIGHT FOR SORE EYES, SO YOU ARE.",
    "NOW DON'T BE A STRANGER, YOU HEAR ME.",
    "THERE'S ALWAYS A SEAT BY THE FIRE FOR YOU.",
    "AH, GO ON WITH YOU, YOU'RE TOO KIND.",
    "GRAND CRAIC TONIGHT, ISN'T IT JUST.",
    "SURE AND ISN'T THAT JUST THE WAY OF IT.",
    "HAVE ANOTHER, GO ON, IT'S ON THE HOUSE.",
    "NOW THAT'S A STORY WORTH THE TELLING, THAT IS.",
)

_KRAGAN_CATCHPHRASES = (
    "MY BLADE DOESN'T MISS.", "THIS PASS BELONGS TO ME NOW.",
    "NOBODY LEAVES WITHOUT PAYING.", "I'VE BURIED BETTER THAN YOU.",
    "SPEAK FAST, MY PATIENCE IS SHORT.", "THE SHADOWS ARE MY ONLY FRIENDS.",
    "CROSS ME ONCE, REGRET IT FOREVER.", "GOLD OR BLOOD, YOUR CHOICE.",
    "I DON'T NEGOTIATE. I COLLECT.",
    "EVERY ROAD OUT OF HERE HAS MY PRICE ON IT.",
    "YOU'LL LEARN MY NAME THE HARD WAY OR NOT AT ALL.",
    "MERCY'S A LUXURY I STOPPED AFFORDING LONG AGO.",
    "MY CREW ANSWERS TO ME AND ME ALONE.",
    "I'VE TAKEN MORE THAN COIN FROM MEN LIKE YOU.",
    "THIS PASS HAS A TOLL. IT'S EVERYTHING YOU'VE GOT.",
    "NOBODY REMEMBERS THE CAUTIOUS ONES. THEY REMEMBER ME.",
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
GOSSIP_EVENTS = ("shadewrath_allied", "korrath_pleaded", "princess_freed")

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
        "RUMOR HAS IT THE NECROMANCER WANTS PEACE NOW. WHO WOULD'VE GUESSED.",
        "I HEARD SOMEONE ACTUALLY SAT DOWN AND TALKED WITH HIM. AND LIVED.",
        "THEY SAY HE'S OFFERING SOMETHING INSTEAD OF TAKING FOR ONCE.",
        "WORD AROUND TOWN IS THE OLD VILLAIN MADE A REAL OFFER. AN ALLIANCE, THEY SAY.",
        "SOMEONE SWORE TO ME THE NECROMANCER PROPOSED TERMS. TERMS! CAN YOU IMAGINE.",
    ),
    "korrath_pleaded": (
        "I HEARD THE BOUND KNIGHT ASKED SOMEONE FOR HELP. POOR SOUL.",
        "THEY SAY HE BEGGED TO BE FREED. IMAGINE CARRYING THAT BURDEN SO LONG.",
        "WORD IS THE KNIGHT FINALLY SPOKE HIS TRUE WISH. IT BREAKS MY HEART.",
        "SOMEONE TOLD ME THE BOUND KNIGHT ASKED FOR HIS FREEDOM. I HOPE HE FINDS PEACE.",
        "I HEARD HE'S NOT JUST STANDING GUARD ANYMORE. HE ASKED FOR A WAY OUT.",
        "I HEARD HE FINALLY ASKED SOMEONE FOR HELP. AFTER ALL THESE YEARS.",
        "THEY SAY THE BOUND KNIGHT BROKE HIS SILENCE AT LAST.",
        "WORD IS HE ADMITTED HE WANTS OUT. IMAGINE CARRYING THAT SECRET SO LONG.",
        "SOMEONE TOLD ME THE KNIGHT ACTUALLY SPOKE HIS TRUE WISH OUT LOUD.",
        "I HEARD HE ASKED TO BE FREED. FINALLY. AFTER ALL THIS TIME.",
    ),
    "princess_freed": (
        "DID YOU HEAR? THE PRINCESS FROM RAVENDALE IS FREE. IT'S ALL ANYONE'S TALKING ABOUT.",
        "THEY SAY SOMEONE ACTUALLY GOT HER OUT OF THAT DUNGEON. GOOD NEWS FOR ONCE.",
        "WORD IS THE ELF PRINCESS IS HEADING HOME AT LAST.",
        "SOMEONE TOLD ME SHE'S BEEN RESCUED. I ALMOST DIDN'T BELIEVE IT.",
        "I HEARD RAVENDALE'S PRINCESS IS SAFE NOW. THAT'S THE BEST THING I'VE HEARD IN AGES.",
        "I HEARD THE WHOLE TOWN'S TALKING ABOUT THE PRINCESS'S RESCUE.",
        "THEY SAY SHE'S FINALLY HEADED BACK TO RAVENDALE, SAFE AND SOUND.",
        "WORD IS SOMEONE ACTUALLY PULLED IT OFF. THE RESCUE, I MEAN.",
        "SOMEONE TOLD ME SHE CRIED WHEN SHE SAW DAYLIGHT AGAIN. WOULDN'T YOU?",
        "I HEARD THE PRINCESS IS FREE AT LAST. BEST NEWS THIS TOWN'S HAD IN AGES.",
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
        "YOU'RE NEW AROUND HERE, AREN'T YOU?",
        "CAN'T SAY I KNOW YOU, BUT I'D LIKE TO.",
        "FIRST TIME I'VE SEEN YOUR FACE HERE.",
        "WELCOME, STRANGER. WE'LL SEE HOW THIS GOES.",
        "YOU'RE NOT FROM AROUND HERE, ARE YOU?",
        "A NEW NAME TO LEARN, THEN.",
        "I DON'T KNOW YOUR STORY YET.",
        "EVERYONE'S A STRANGER ONCE.",
        "LET'S START WITH HELLO, THEN.",
    ),
    "acquaintance": (
        "LONG TIME NO SEE.", "NICE TO SEE YOU AGAIN.",
        "WE'VE MET BEFORE, HAVEN'T WE?", "I REMEMBER OUR LAST TALK.",
        "WE'VE SHARED A FEW MOMENTS.",
        "AH, WE'VE CROSSED PATHS BEFORE, HAVEN'T WE.",
        "GOOD TO SEE A FAMILIAR FACE.",
        "I WAS WONDERING WHEN YOU'D TURN UP AGAIN.",
        "YOU'RE STARTING TO BECOME A REGULAR.",
        "I REMEMBER YOU FROM LAST TIME. GOOD MEMORY, THIS ONE.",
        "WE'VE SPOKEN ENOUGH NOW THAT THIS FEELS EASY.",
        "BACK AGAIN, I SEE. GOOD.",
        "WE'VE BUILT UP A FEW SHARED STORIES BY NOW.",
        "I'D CALL US ACQUAINTED AT THIS POINT.",
    ),
    "neutral": (
        "HELLO AGAIN.", "NICE TO SEE YOU.", "WE'VE CROSSED PATHS BEFORE.",
        "I'VE SEEN YOU AROUND.", "LET'S SEE WHERE THIS GOES.",
        "YOU'RE A FAMILIAR SIGHT BY NOW.",
        "I DON'T MIND SEEING YOU AROUND.",
        "WE'VE GOT NO QUARREL, YOU AND I.",
        "THINGS SEEM FINE BETWEEN US SO FAR.",
        "NOTHING TO COMPLAIN ABOUT HERE.",
        "WE GET ALONG WELL ENOUGH, I'D SAY.",
        "YOU'RE EASY ENOUGH COMPANY.",
        "NO TROUBLE FROM YOU SO FAR. I APPRECIATE THAT.",
        "WE'LL KEEP THIS FRIENDLY, THEN.",
    ),
    "friend": (
        "GREAT TO SEE YOU.", "IT'S ALWAYS A PLEASURE.",
        "YOU'RE ALWAYS WELCOME HERE.", "I'M GLAD WE'RE FRIENDS.",
        "WE'VE GOT A GOOD THING GOING.",
        "YOU'VE EARNED A PLACE HERE, TRULY.",
        "I LOOK FORWARD TO YOUR VISITS, YOU KNOW.",
        "WE'VE BECOME GOOD FRIENDS, HAVEN'T WE.",
        "IT'S GOOD TO HAVE YOU IN MY CORNER.",
        "YOU'RE ONE OF THE GOOD ONES, FRIEND.",
        "I COUNT US AS FRIENDS NOW, TRULY.",
        "THINGS ARE BETTER WITH YOU AROUND.",
        "I'D CALL YOU A FRIEND WITHOUT HESITATION.",
        "WE'VE GOT SOMETHING GOOD GOING HERE.",
    ),
    "close_friend": (
        "MISSED YOU.", "I'M HERE FOR YOU.", "WE'VE GOT A STRONG BOND.",
        "YOU'RE OFTEN ON MY MIND.", "GOOD TO HAVE YOU BACK.",
        "YOU'RE MORE THAN JUST A FRIEND AT THIS POINT.",
        "I THINK OF YOU MORE THAN I'D ADMIT.",
        "WE'VE BEEN THROUGH ENOUGH TO CALL THIS REAL.",
        "I TRUST YOU WITH THE THINGS THAT MATTER.",
        "YOU'VE EARNED A DEEPER TRUST THAN MOST.",
        "I DON'T SAY THIS OFTEN, BUT I VALUE YOU DEEPLY.",
        "WE'VE GOT SOMETHING RARE HERE.",
        "YOU MATTER TO ME MORE THAN I LET ON.",
        "I'D CALL ON YOU BEFORE MOST ANYONE ELSE.",
    ),
    "best_friend": (
        "YOU KNOW YOU CAN ALWAYS COUNT ON ME.", "I'LL BE THERE FOR YOU.",
        "WE'VE GOT AN UNBREAKABLE BOND.", "YOU'RE PART OF THE FAMILY HERE.",
        "I'LL ALWAYS BE HERE FOR YOU.",
        "THERE'S NOTHING I WOULDN'T DO FOR YOU AT THIS POINT.",
        "YOU'RE FAMILY. THAT'S NOT A FIGURE OF SPEECH ANYMORE.",
        "I TRUST YOU MORE THAN I TRUST MOST PEOPLE I'VE KNOWN MY WHOLE LIFE.",
        "WHATEVER YOU NEED, YOU KNOW WHERE TO FIND ME.",
        "THIS BOND ISN'T GOING ANYWHERE. NEITHER AM I.",
        "YOU'VE EARNED MY DEEPEST TRUST, AND I DON'T GIVE THAT EASILY.",
        "I'D DROP ANYTHING FOR YOU. YOU KNOW THAT BY NOW.",
        "WE'RE IN THIS TOGETHER, NO MATTER WHAT COMES.",
        "YOU'RE THE CLOSEST THING TO FAMILY I'VE GOT HERE.",
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
                # M11.1: this cast has no authored private/secret-register
                # content (no bible entries -- unlike Selena/Shadewrath/
                # Korrath/Elowen below, whose corpora DO vary AUD: against
                # real differentiated content). Fixing AUD:witnessed here
                # rather than randomizing against identical text avoids
                # teaching the model a fake alone/witnessed distinction
                # this cast's phrase banks don't actually carry.
                prompt = prompt_fields(profile, relationship, mood, context,
                                       audience="witnessed", event=event)
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
