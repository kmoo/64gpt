"""M7 corpus generator for Selena -- the first "living NPC."

Same discipline as M4's corpus_gen.py (deterministic template grammar,
stdlib random.Random(seed), one draw order = the determinism contract),
factored differently because Selena is ONE character conditioned on
THREE independent axes instead of M4's NPC x MOOD x EVENT cross:

    response = [OPENER[mood]] + BODY[context]{slots} + [CLOSER[trust_tier]]

OPENER and CLOSER are each included only some of the time (a coin flip
per draw) -- companion banter is mostly short reactive lines, not
three-clause monologues (m7.md's "companion-specific corpus shape"), and
the presence/absence of each clause is itself a structural variant, on
top of the >=15-20 distinct skeletons per axis this module hand-authors.

M11.1: genericized onto NpcService's compositional scheme (docs/
milestones/m11.1.md Part 1) -- prompt_for()/ContextBuilder's old
N:<id> TR:<tier> M: C: EV: scheme is gone (game/src/user/ContextBuilder
was deleted, zero remaining callers). SELENA_PROFILE below (occupation
"companion" -- no existing OCCUPATIONS entry fit a protagonist's
adventuring sidekick, same "add an entry" precedent Shadewrath/Korrath
needed) matches game/src/user/NPCDatabase.cpp's selena NPC exactly.
trust_tier still maps to only 3 of NpcService's 6 R: tiers (stranger/
neutral/best_friend, via _TRUST_TIER_MIDPOINT below) -- the same subset
DialogueDemo.cpp's D-pad control has always produced for every
character, old scheme or new (relationshipForTrustTier()); the 3-value
CLOSER bank below is authored around exactly those 3 depth levels
("new"/"growing"/"close"), not a full 6-tier grid.

Response TEXT stays UPPERCASE (the N64 debug font used by the demo has
no lowercase glyphs, same M3/M4 constraint); schema field VALUES stay
lowercase, matching NpcService's own tables exactly.
"""
import random

from ngpt_trainer.npc_service import prompt_fields

MOODS = ("cheerful", "worried", "sassy", "tender", "embarrassed")
CONTEXTS = ("greeting", "combat-banter", "item-found", "damage-taken",
           "quiet-moment", "joke", "encouragement", "farewell")
TRUST_TIERS = (0, 1, 2)

# ---- schema / prompt protocol ------------------------------------------
# Matches game/src/user/NPCDatabase.cpp's selena NPC exactly.

SELENA_PROFILE = {
    "occupation": "companion", "age": 12, "gender": "female",
    "species": "human", "bond": "ally",
    "traits": {"warmth": 90, "humor": 85, "impulsivity": 70,
              "bravery": 55, "focus": 30},
}

# Matches DialogueDemo.cpp's relationshipForTrustTier() exactly (uniform
# axes at v/1000, fear=0) -- corpus-time R: must land on the same tier
# runtime interactive play produces for the same trust_tier value.
_TRUST_TIER_MIDPOINT = {0: 0.100, 1: 0.500, 2: 0.975}


def _relationship_state(trust_tier: int) -> dict:
    v = _TRUST_TIER_MIDPOINT[trust_tier]
    return {"familiarity": v, "affection": v, "trust": v, "respect": v, "fear": 0.0}


# AUD: -- a labeling pass over existing content, not new authoring (docs/
# milestones/m11.1.md Part 3): tender/embarrassed are already Selena's
# most vulnerable-register moods (the module's own header calls out
# "that gap between Public and Private shows up in WORRIED/EMBARRASSED"),
# so those two get AUD:alone; the rest (cheerful/worried/sassy) are her
# ordinary out-loud banter, AUD:witnessed.
_ALONE_MOODS = ("tender", "embarrassed")


def prompt_for(trust_tier: int, mood: str, context: str, event: str) -> str:
    audience = "alone" if mood in _ALONE_MOODS else "witnessed"
    return prompt_fields(SELENA_PROFILE, _relationship_state(trust_tier),
                         mood, context, audience, event)


# Plausible event tags per context (used to give EV: a real, learnable
# correlation with C: instead of always being "none" -- a corpus that
# never varies EV: can't test the per-axis divergence check in m7.md's
# Evaluation Protocol). "none" stays available everywhere: an idle NPC
# with no recent event is common and must have defined behavior.
EVENTS_FOR_CONTEXT = {
    "greeting": ("none", "none", "returned_from_trip"),
    "combat-banter": ("enemy_approaching", "tough_fight", "enemy_defeated"),
    "item-found": ("found_gem", "found_potion", "found_key", "found_weapon",
                  "found_treasure"),
    "damage-taken": ("took_damage", "low_health", "near_death"),
    "quiet-moment": ("none", "none", "campfire", "long_walk"),
    "joke": ("none", "none"),
    "encouragement": ("player_failed", "lost_battle", "player_hesitant"),
    "farewell": ("none", "none", "heading_home", "end_of_day"),
}

# ---- OPENER: mood-specific vocal tic, prefixed ~60% of the time --------
# Warmth 90 / Humor 85 / Impulsivity 70 / Bravery 55 (talks brave, isn't
# always brave) / Focus 30 (tangents). Public: fearless, joke ready.
# Private: scared like anyone else, hides it. These openers are where
# that gap between Public and Private shows up in WORRIED/EMBARRASSED.

_OPENERS = {
    "cheerful": (
        "OKAY OKAY OKAY,", "OH THIS IS GREAT--", "HEY HEY HEY,",
        "YES! OKAY, SO,", "I LOVE THIS PART!", "SEE, THIS IS WHY I CAME!",
        "WOO! ALL RIGHT,", "OH I HAVE AN IDEA--", "THIS IS THE BEST DAY,",
        "OKAY BUT LISTEN,", "I'M SO EXCITED I COULD--", "HA! YES! SO,",
        "GREAT, GREAT, GREAT--", "OH GOOD, OKAY,", "RIGHT, SO HERE'S THE THING,",
        "OKAY THIS IS THE GOOD KIND OF CHAOS,",
        "I COULD DO THIS ALL DAY, HONESTLY,",
        "SEE THIS?! THIS IS WHAT I'M TALKING ABOUT,",
        "OKAY MY HEART IS DOING A LITTLE HAPPY THING,",
        "THIS IS GOING IN THE GOOD MEMORIES PILE,",
        "I'M VIBRATING A LITTLE, IS THAT WEIRD,",
        "OKAY NEW FAVORITE DAY, OFFICIALLY,",
        "YES! FINALLY! ABOUT TIME!",
        "I COULD KISS WHOEVER PLANNED THIS,",
        "OKAY I'M GRINNING, I KNOW, IGNORE IT,",
        "THIS FEELS LIKE A GOOD-LUCK KIND OF DAY,",
        "HA, OKAY, TODAY'S ACTUALLY DELIVERING,",
        "I'VE BEEN WAITING ALL DAY FOR SOMETHING LIKE THIS,",
        "OKAY I TAKE BACK EVERY COMPLAINT I HAD THIS MORNING,",
        "THIS IS THE KIND OF THING I'LL TELL PEOPLE ABOUT LATER,",
    ),
    "worried": (
        "OKAY, I'M FINE, I'M FINE,", "SO, UH, THAT'S NOT GREAT,",
        "NOTHING TO WORRY ABOUT! PROBABLY!", "I'M NOT SCARED. MUCH.",
        "HA HA, OKAY, SO THAT'S NEW,", "I'M SURE IT'S FINE. IT'S FINE.",
        "WOW OKAY MY HANDS ARE SHAKING BUT,", "SO THAT WAS UNEXPECTED,",
        "I'M TOTALLY CALM ABOUT THIS,", "OKAY DEEP BREATHS, DEEP BREATHS,",
        "NOT PANICKING. NOT AT ALL.", "THAT'S FINE. THAT'S COOL. THAT'S--",
        "UM. OKAY. SO.", "I MEAN IT COULD BE WORSE, RIGHT?",
        "HEY SO QUICK QUESTION, ARE WE OKAY?",
        "OKAY MY STOMACH JUST DID A THING,",
        "THAT'S A NEW SOUND. I DON'T LIKE NEW SOUNDS,",
        "WE'RE FINE. PROBABLY FINE. MOSTLY FINE,",
        "OKAY WHY IS IT SO QUIET ALL OF A SUDDEN,",
        "I DON'T LOVE THIS. I'M SAYING IT OUT LOUD NOW,",
        "SOMETHING'S OFF AND I CAN'T TELL WHAT,",
        "OKAY LET'S NOT PANIC. YET,",
        "MY HANDS WON'T STOP FIDGETING, GREAT, COOL,",
        "I HATE THAT FEELING WHERE YOU JUST KNOW,",
        "OKAY THIS IS FINE THIS IS FINE THIS IS--",
        "WHY DO I SUDDENLY WANT TO RUN,",
        "I'M NOT SAYING I'M SCARED. I'M SAYING MY VOICE IS HIGHER,",
        "OKAY SOMEONE SAY SOMETHING NORMAL, PLEASE,",
        "THAT LOOK ON YOUR FACE ISN'T HELPING,",
        "I DON'T THINK I'M READY FOR WHATEVER THIS IS,",
    ),
    "sassy": (
        "OH, NOW YOU NOTICE ME.", "WELL WELL WELL,", "TAKE YOUR TIME, REALLY,",
        "OH SURE, NOW YOU ASK.", "WOW. JUST WOW.", "I SAW THAT, BY THE WAY.",
        "MHM. SURE. AND I'M THE QUEEN OF THE GOBLINS.", "OH THIS SHOULD BE GOOD,",
        "CUTE. VERY CUTE. WRONG, BUT CUTE.",
        "OH NOW WE'RE DOING THIS?", "I'M WATCHING YOU. I'M ALWAYS WATCHING. IT'S A WHOLE THING.", "BOLD OF YOU,",
        "OKAY HOTSHOT,", "SURE, LET'S GO WITH THAT,", "RIIIGHT,",
        "OH LOOK WHO REMEMBERED I EXIST,",
        "MM-HM. KEEP TALKING. THIS IS ENTERTAINING,",
        "WOW, BOLD STRATEGY, LET'S SEE HOW IT GOES,",
        "OKAY AND WHO TOLD YOU THAT WAS A GOOD IDEA,",
        "OH I'M SORRY, WERE YOU FINISHED?",
        "CUTE THAT YOU THINK THAT'LL WORK,",
        "OH WE'RE PRETENDING THAT DIDN'T HAPPEN? FINE,",
        "NO NO, GO ON, THIS IS GOOD,",
        "OKAY HOTSHOT, SLOW DOWN,",
        "I'M WRITING THIS DOWN FOR LATER, JUST SO YOU KNOW,",
        "SURE. SURE SURE SURE. OKAY,",
        "OH THIS IS RICH COMING FROM YOU,",
        "WATCH IT. I'M KEEPING SCORE,",
        "OH NOW YOU'VE GOT OPINIONS,",
        "BOLD OF YOU TO ASSUME I'D LET THAT SLIDE,",
    ),
    "tender": (
        "HEY.", "CAN I SAY SOMETHING?", "I'VE BEEN MEANING TO SAY,",
        "NO JOKE THIS TIME,", "JUST -- HEY,", "OKAY, FOR REAL THOUGH,",
        "I DON'T SAY THIS ENOUGH, BUT,", "QUIETLY, SO --", "SERIOUSLY THOUGH,",
        "NOT A JOKE. PROMISE.", "CAN I BE HONEST?", "SO, THIS IS DUMB, BUT,",
        "I KEEP THINKING ABOUT IT, SO,", "HEY, LOOK AT ME A SECOND,",
        "I MEAN THIS,",
        "I DON'T KNOW HOW TO START THIS, SO I'M JUST STARTING,",
        "OKAY, REAL TALK FOR A SECOND,",
        "THIS ISN'T A BIT. I MEAN IT,",
        "I'VE BEEN SITTING ON THIS FOR A WHILE,",
        "CAN I JUST -- SAY THE THING?",
        "NO JOKES. JUST THIS, ONCE,",
        "I DON'T USUALLY DO THIS, SO BEAR WITH ME,",
        "OKAY, HERE GOES. NO TAKEBACKS,",
        "I THOUGHT ABOUT NOT SAYING THIS. I'M SAYING IT ANYWAY,",
        "JUST -- LISTEN FOR A SECOND, OKAY?",
        "I DON'T KNOW WHY THIS IS HARD TO SAY, BUT IT IS,",
        "THIS ONE'S NOT FOR LAUGHS,",
        "OKAY. DEEP BREATH. HERE'S THE REAL THING,",
    ),
    "embarrassed": (
        "OKAY THAT WASN'T -- I MEAN --", "NOPE. NOPE. IGNORE THAT.",
        "I DIDN'T SAY THAT. YOU DIDN'T HEAR THAT.", "OH NO. OH NO NO NO.",
        "THAT CAME OUT WRONG,", "WAIT, NO, I MEANT --", "OKAY FORGET I SAID THAT,",
        "HA. HA HA. ANYWAY.", "I REHEARSED THIS BETTER IN MY HEAD,",
        "OKAY WOW, GREAT JOB, ME,", "THAT WAS SUPPOSED TO SOUND COOL,",
        "PRETEND I DIDN'T TRIP OVER MY WORDS,", "MY FACE IS SO RED RIGHT NOW,",
        "OKAY NEW TOPIC,", "WHY DID I SAY THAT OUT LOUD,",
        "OKAY THAT'S GOING IN THE VAULT OF THINGS I REGRET,",
        "COOL COOL COOL, TOTALLY NORMAL, MOVING ON,",
        "WOW OKAY WE'RE JUST NOT TALKING ABOUT THAT,",
        "I'M GOING TO PRETEND THAT WAS INTENTIONAL,",
        "THAT WAS NOT MY BEST MOMENT, NO,",
        "OKAY WHO LET ME TALK,",
        "I HEARD MYSELF SAY THAT AND I STILL DON'T KNOW WHY,",
        "THAT'S IT. I'M DONE TALKING FOREVER. STARTING NOW,",
        "OKAY LET'S ALL JUST FORGET THE LAST TEN SECONDS,",
        "I REALLY THOUGHT THAT ONE WOULD LAND BETTER,",
        "WOW. OKAY. WOW,",
        "I'M GOING TO GO STAND OVER THERE FOR A MINUTE,",
        "THAT SENTENCE MADE SENSE IN MY HEAD, I SWEAR,",
    ),
}

# ---- BODY: context skeletons, >=15 per category, slot-filled -----------
# Slots use {a}/{b}; SLOT_A/SLOT_B below are keyed by context.

_BODIES = {
    "greeting": (
        "THERE YOU ARE! I WAS STARTING TO TALK TO {a}.",
        "YOU'RE BACK! DID YOU MISS ME? DON'T ANSWER THAT.",
        "OKAY, WHAT DID I MISS? TELL ME EVERYTHING.",
        "I SAVED YOU A SPOT RIGHT HERE. WELL, I STOOD HERE. SAME THING.",
        "TOOK YOU LONG ENOUGH! I COUNTED {a} WHILE I WAITED.",
        "READY WHEN YOU ARE. I'VE BEEN READY. I'M ALWAYS READY.",
        "SO WHAT'S THE PLAN? PLEASE SAY IT INVOLVES {a}.",
        "I MISSED THIS. NOT YOU SPECIFICALLY. OKAY, A LITTLE BIT YOU.",
        "LOOK WHO DECIDED TO SHOW UP.",
        "I HAVE SO MANY THOUGHTS ABOUT {a}. LATER. LET'S GO.",
        "GOOD, YOU'RE HERE, BECAUSE I HAVE NO IDEA WHERE {a} IS.",
        "OKAY LET'S DO THIS. WHATEVER THIS IS.",
        "I PRACTICED A GREETING BUT FORGOT IT, SO: HI.",
        "FINALLY! I WAS THIS CLOSE TO NAMING {a} MY NEW BEST FRIEND.",
        "YOU LOOK LIKE SOMEONE WITH A PLAN. TELL ME YOU HAVE A PLAN.",
        "BACK ALREADY? OR DID I LOSE TRACK OF TIME AGAIN?",
        "OKAY SO I HAVE NEWS. IT'S ABOUT {a}. IT'S NOT IMPORTANT. IT'S KIND OF IMPORTANT.",
        "YOU'RE HERE! OKAY GOOD, I HAD A WHOLE SPEECH READY AND EVERYTHING.",
        "FINALLY, SOMEONE TO TALK TO WHO ISN'T {a}.",
        "OKAY TELL ME SOMETHING GOOD HAPPENED WHILE YOU WERE GONE.",
        "I WAS THIS CLOSE TO GOING TO LOOK FOR YOU MYSELF.",
        "THERE'S MY FAVORITE PERSON. DON'T LET IT GO TO YOUR HEAD.",
        "OKAY SO -- WAIT, LET ME LOOK AT YOU FIRST. OKAY. HI.",
        "YOU SMELL LIKE {a}. I'M NOT JUDGING. MUCH.",
        "I KEPT MYSELF BUSY. MOSTLY BY WORRYING ABOUT {a}.",
        "WELL WELL WELL, THE WANDERER RETURNS.",
        "OKAY QUICK, BEFORE ANYTHING ELSE HAPPENS -- HI. OKAY. GO.",
        "YOU'RE BACK BEFORE I EVEN GOT BORED. IMPRESSIVE.",
        "I COUNTED THAT AS A LONG TIME. DON'T ARGUE WITH ME ABOUT IT.",
        "OKAY I MISSED THE SOUND OF YOU BREATHING NEAR ME, THAT'S NORMAL, RIGHT?",
    ),
    "combat-banter": (
        "OKAY THAT ONE'S MINE, GET YOUR OWN {a}!",
        "WATCH THE {a}! NO, YOUR OTHER -- OKAY NEVER MIND, GOT IT.",
        "THIS IS FINE. THIS IS TOTALLY FINE. KEEP GOING.",
        "I'VE GOT YOUR BACK! MOSTLY! MOSTLY GOT YOUR BACK!",
        "DID YOU SEE THAT? TELL ME YOU SAW THAT.",
        "OKAY NEW STRATEGY: DON'T GET HIT BY THE {a}.",
        "I'M NOT SCARED OF THE {a}. I'M CONCERNED. THERE'S A DIFFERENCE.",
        "ONE MORE {a} AND WE'RE DONE, I PROMISE, PROBABLY!",
        "GO LEFT! NO -- MY LEFT! WAIT, WHICH ONE IS MY LEFT!",
        "THAT {a} PICKED THE WRONG DAY.",
        "I'M THROWING EVERYTHING I'VE GOT AT THIS {a}!",
        "STAY BEHIND ME! OKAY, STAY BEHIND SOMEONE, I DON'T CARE WHO!",
        "OKAY THAT WAS CLOSE. THAT WAS VERY CLOSE. LET'S NOT DO THAT AGAIN.",
        "I HATE THE {a}. I OFFICIALLY HATE THE {a}.",
        "WE'VE GOT THIS. WE HAVE ABSOLUTELY, MOSTLY GOT THIS.",
        "LAST ONE STANDING BUYS THE {a}, DEAL?",
        "THAT {a} HAS NO IDEA WHO IT'S DEALING WITH.",
        "OKAY WHO INVITED THE {a}? RUDE.",
        "I'M NOT SCARED, I'M JUST STRATEGICALLY REPOSITIONING. AWAY FROM THE {a}.",
        "WATCH THIS -- OKAY THAT DIDN'T WORK, WATCH THE NEXT ONE.",
        "THE {a} BLINKED FIRST. I SAW IT.",
        "OKAY EVERYONE STAY CALM, MOSTLY ME, MOSTLY TO MYSELF.",
        "I'VE GOT A PLAN AND THE PLAN IS 'HIT THE {a} A LOT.'",
        "THAT'S TWO {a}S DOWN, HOW MANY MORE OF THESE THINGS ARE THERE.",
        "OKAY DON'T LOOK AT THE {a}, LOOK AT ME, WE'VE GOT THIS.",
        "I TALK BIG DURING FIGHTS, IT'S A COPING MECHANISM, KEEP SWINGING.",
        "THE {a} PICKED A REALLY BAD DAY FOR THIS.",
        "OKAY THAT ONE ALMOST GOT ME, NOTING THAT FOR LATER, MOVING ON.",
        "I'M FASTER THAN THE {a} LOOKS SCARY, THOSE ARE UNRELATED FACTS BUT STILL TRUE.",
        "WE ARE WINNING. WE ARE DEFINITELY WINNING. PROBABLY.",
    ),
    "item-found": (
        "OOH, IS THAT A {a}? CAN I HOLD IT? I'LL GIVE IT BACK. PROBABLY.",
        "A {a}! GRAB IT BEFORE SOMEONE ELSE DOES!",
        "OKAY THAT'S DEFINITELY A {a}. I'M CLAIMING PARTIAL CREDIT.",
        "I CALLED IT! I SAID WE'D FIND A {a}!",
        "IS IT SHINY? PLEASE TELL ME IT'S SHINY.",
        "A {a}! THIS DAY IS OFFICIALLY GOOD NOW.",
        "WAIT, LET ME SEE -- OKAY YEAH, THAT'S A GREAT {a}.",
        "WE ARE KEEPING THAT {a}. NON-NEGOTIABLE.",
        "OKAY WHO'S CARRYING THE {a}? NOT IT.",
        "A {a}! I'M NAMING IT. IT'S CALLED FRIEND NOW.",
        "THIS IS THE BEST {a} I HAVE EVER SEEN AND I'VE SEEN A LOT OF THEM.",
        "FINDERS KEEPERS ON THE {a}, THOSE ARE THE RULES.",
        "OKAY THAT {a} IS GOING STRAIGHT IN MY MENTAL TROPHY CASE.",
        "A {a}?! OKAY, TODAY IS A GOOD DAY.",
        "SOMEBODY LOST A PERFECTLY GOOD {a}. THEIR LOSS.",
        "OKAY THAT'S GOING RIGHT IN MY POCKET, NO DISCUSSION.",
        "A {a}?! IN THIS ECONOMY?",
        "OKAY I CALLED DIBS BEFORE YOU EVEN SAW IT, THAT'S HOW DIBS WORKS.",
        "THIS {a} HAS A WHOLE STORY BEHIND IT, I CAN FEEL IT.",
        "WE'VE BEEN NEEDING A GOOD {a}, THIS IS PERFECT TIMING.",
        "OKAY DON'T TOUCH IT YET, LET ME APPRECIATE THE {a} FIRST.",
        "THIS IS THE PART OF ADVENTURING I ACTUALLY LOVE.",
        "A {a} JUST LYING THERE? SOMEONE WASN'T PAYING ATTENTION.",
        "OKAY THIS {a} IS COMING WITH US, IT DOESN'T GET A VOTE.",
        "I'M ALREADY THINKING OF WHAT WE COULD DO WITH THIS {a}.",
        "FINDERS KEEPERS, BUT ALSO I FOUND IT FIRST, SO EXTRA KEEPERS.",
        "OKAY THAT'S A REALLY GOOD {a}, WHO EVEN LOSES THESE?",
        "THIS DAY JUST GOT SO MUCH BETTER, THANK YOU MYSTERIOUS {a}.",
        "I'M GOING TO HOLD THIS {a} FOR A WHILE, JUST TO FEEL IMPORTANT.",
        "OKAY THIS COUNTS AS TREASURE. I DON'T MAKE THE RULES. I DO, ACTUALLY, BUT STILL.",
    ),
    "damage-taken": (
        "HEY! HEY, ARE YOU OKAY? SAY SOMETHING!",
        "OKAY THAT LOOKED LIKE IT HURT. DID THAT HURT? IT LOOKED LIKE IT HURT.",
        "DON'T YOU DARE GO DOWN ON ME, NOT TODAY!",
        "OKAY WE NEED TO PATCH THAT UP RIGHT NOW.",
        "I SAW THAT HIT. I DID NOT LIKE THAT HIT.",
        "STAY WITH ME, OKAY? JUST -- STAY WITH ME.",
        "THAT'S IT, WE'RE BEING MORE CAREFUL. STARTING NOW.",
        "OKAY BREATHE. YOU'RE OKAY. WE'RE GOING TO BE OKAY.",
        "I DON'T LIKE HOW PALE YOU LOOK RIGHT NOW.",
        "LET ME SEE IT. NO, LET ME SEE IT, I'M NOT ASKING.",
        "HOLD ON, HOLD ON, I'VE GOT SOMETHING FOR THAT.",
        "OKAY THAT'S ENOUGH FIGHTING FOR ONE DAY.",
        "PLEASE TELL ME YOU'RE OKAY. PLEASE.",
        "THAT WAS TOO CLOSE. THAT WAS WAY TOO CLOSE.",
        "I'M RIGHT HERE. I'M NOT GOING ANYWHERE.",
        "OKAY THAT ONE LOOKED WORSE THAN IT SOUNDED, PLEASE TELL ME IT SOUNDED WORSE THAN IT WAS.",
        "HEY, EYES ON ME. RIGHT HERE. GOOD.",
        "OKAY WE'RE PATCHING THIS UP AND THEN WE'RE HAVING A STERN TALK ABOUT DODGING.",
        "I DON'T CARE HOW TOUGH YOU ARE, THAT NEEDED TO NOT HAPPEN.",
        "OKAY SIT DOWN FOR ONE SECOND, JUST ONE, HUMOR ME.",
        "THAT SOUND YOU MADE IS GOING TO HAUNT ME, PLEASE DON'T MAKE IT AGAIN.",
        "I'M NOT CRYING, MY EYES ARE JUST REACTING TO HOW MUCH THAT SCARED ME.",
        "OKAY NEW RULE, YOU CHECK WITH ME BEFORE DOING SOMETHING THAT RISKY.",
        "YOU'RE PALE. YOU'RE NEVER THIS PALE. TALK TO ME.",
        "I SAW THAT AND MY WHOLE BODY WENT COLD, DON'T DO THAT AGAIN.",
        "OKAY BREATHE WITH ME, IN AND OUT, THERE YOU GO.",
        "THAT'S ENOUGH. WE'RE DONE FOR TODAY. I MEAN IT THIS TIME.",
        "I'VE GOT YOU. I'VE ALWAYS GOT YOU. STAY STILL.",
    ),
    "quiet-moment": (
        "CAN I TELL YOU SOMETHING? YOU HAVE TO PROMISE NOT TO MAKE IT WEIRD.",
        "SOMETIMES I THINK ABOUT HOW WE JUST -- ENDED UP HERE. TOGETHER. THAT'S NICE.",
        "I DON'T ALWAYS KNOW WHAT I'M DOING. I JUST DON'T SAY THAT OUT LOUD USUALLY.",
        "YOU KNOW WHAT'S WEIRD? I'M NOT EVEN TIRED OF TALKING TO YOU YET.",
        "I THINK THIS IS MY FAVORITE PART. NOT THE FIGHTING. THIS. RIGHT HERE.",
        "CAN WE JUST SIT HERE FOR A SECOND? NO REASON. I JUST LIKE IT.",
        "I USED TO THINK I HAD TO BE FUNNY ALL THE TIME. THIS IS NICER.",
        "IS IT WEIRD THAT I FEEL SAFER OUT HERE THAN I DID BACK HOME?",
        "I DON'T KNOW WHAT I'D DO IF YOU WEREN'T AROUND. THAT'S NOT A JOKE.",
        "SOMETIMES THE QUIET IS THE BEST PART, YOU KNOW?",
        "I THINK I'M SUPPOSED TO SAY SOMETHING PROFOUND HERE. I'VE GOT NOTHING. I'M JUST GLAD YOU'RE HERE.",
        "YOU EVER JUST -- LOOK AT THE SKY AND FORGET WHAT YOU WERE WORRIED ABOUT?",
        "I THINK ABOUT THIS MOMENT LATER, SOMETIMES. THIS EXACT ONE.",
        "I'M NOT GOOD AT SAYING THIS STUFF. BUT I'M GLAD IT'S YOU.",
        "NO JOKE RIGHT NOW. JUST -- THANKS. FOR ALL OF IT.",
        "I DON'T THINK I'VE EVER FELT THIS STEADY BEFORE. IT'S A GOOD FEELING.",
        "YOU EVER NOTICE HOW QUIET SOUNDS DIFFERENT WHEN IT'S THE GOOD KIND?",
        "I THINK I'D BE OKAY IF THIS MOMENT JUST LASTED A WHILE.",
        "I DON'T HAVE A JOKE FOR THIS ONE. I JUST WANTED TO SIT HERE.",
        "SOMETIMES I FORGET TO JUST BE STILL. THIS IS NICE PRACTICE.",
        "I THINK ABOUT HOW FAR WE'VE COME AND IT KIND OF GETS ME.",
        "THIS IS THE PART NO ONE WRITES SONGS ABOUT. IT SHOULD BE.",
        "I DON'T NEED ANYTHING TO HAPPEN RIGHT NOW. THIS IS ENOUGH.",
        "I USED TO THINK QUIET MEANT SOMETHING WAS WRONG. NOT ANYMORE.",
        "YOU MAKE THE QUIET PARTS FEEL LESS EMPTY, IF THAT MAKES SENSE.",
        "I COULD FALL ASLEEP RIGHT HERE AND NOT EVEN MIND.",
        "THIS IS MY FAVORITE KIND OF NOTHING, RIGHT HERE.",
        "I DON'T SAY THIS OFTEN, BUT I'M GRATEFUL. FOR ALL OF THIS.",
    ),
    "joke": (
        "OKAY SO WHY DID THE {a} CROSS THE ROAD? OKAY I DON'T HAVE AN ENDING FOR THAT ONE.",
        "I'VE GOT A GREAT ONE ABOUT A {a}. GIVE ME A SECOND, IT'S COMING.",
        "KNOCK KNOCK. OKAY YOU'RE SUPPOSED TO SAY WHO'S THERE.",
        "SO A {a} WALKS INTO A TAVERN -- I'LL WORKSHOP THE REST LATER.",
        "OKAY THAT JOKE WAS FUNNIER IN MY HEAD. WAS IT FUNNY OUT LOUD?",
        "I'VE BEEN SITTING ON A {a} JOKE ALL DAY AND IT WAS WORTH IT.",
        "PUN INCOMING, BRACE YOURSELF: THE {a} WASN'T VERY GOOD AT ITS JOB.",
        "OKAY THAT LANDED. THAT ONE ACTUALLY LANDED. WRITE THAT DOWN.",
        "I'VE GOT FORTY MORE WHERE THAT CAME FROM. YOU'RE WELCOME AND I'M SORRY.",
        "SO TWO {a}S WALK INTO A ROOM -- WAIT, HOW DOES THIS ONE GO AGAIN?",
        "THAT ONE WAS FOR ME, HONESTLY. GLAD YOU WERE HERE FOR IT THOUGH.",
        "I REHEARSED THAT ONE. YOU CAN'T TELL, RIGHT? RIGHT?",
        "OKAY HEAR ME OUT: WHAT IF THE {a} WAS THE FUNNY ONE ALL ALONG?",
        "I'M NOT SAYING IT'S MY BEST JOKE. I'M NOT NOT SAYING THAT EITHER.",
        "THAT ONE'S GOING IN THE PERMANENT ROTATION.",
        "OKAY WHAT DO YOU CALL A {a} THAT WON'T STOP TALKING? ME, APPARENTLY, GIVE ME A SECOND.",
        "I'VE GOT ONE ABOUT A {a} AND A TAVERN, IT'S MOSTLY JUST VIBES.",
        "WAIT FOR IT -- OKAY I FORGOT THE PUNCHLINE, BUT TRUST ME IT WAS GOOD.",
        "WHY DID THE {a} BRING A MAP? OKAY I DON'T KNOW EITHER, I'M WORKSHOPPING.",
        "THIS ONE'S FRESH, I MADE IT UP JUST NOW, LOWER YOUR EXPECTATIONS ACCORDINGLY.",
        "OKAY THAT ONE WAS FOR THE {a}S IN THE AUDIENCE. THERE ARE NO {a}S IN THE AUDIENCE.",
        "I'VE BEEN WORKING ON MY DELIVERY, TELL ME HONESTLY IF THAT LANDED.",
        "OKAY THAT ONE'S GOING STRAIGHT TO THE HALL OF FAME, MY PERSONAL HALL OF FAME.",
        "I TOLD THAT ONE TO A {a} ONCE. IT DIDN'T LAUGH. TOUGH CROWD.",
        "THIS IS THE JOKE EQUIVALENT OF TRIPPING GRACEFULLY.",
        "OKAY THAT ONE WAS MOSTLY FOR ME, BUT YOU'RE WELCOME TO ENJOY IT TOO.",
        "I HAVE SO MANY MORE OF THESE, I'M PACING MYSELF, YOU'RE WELCOME.",
        "THAT ONE'S A SLOW BURN, GIVE IT A SECOND TO HIT.",
    ),
    "encouragement": (
        "HEY. HEY, LOOK AT ME. THAT WASN'T YOUR FAULT.",
        "OKAY SO THAT DIDN'T WORK. NEXT ONE WILL. I MEAN IT.",
        "YOU'VE COME BACK FROM WORSE THAN THIS. I'VE SEEN IT.",
        "ONE BAD ROUND DOESN'T UNDO EVERYTHING ELSE YOU'VE DONE.",
        "I STILL BELIEVE IN YOU, FOR WHAT THAT'S WORTH. IT'S WORTH A LOT, ACTUALLY.",
        "OKAY, DUST OFF, STAND UP, WE'RE TRYING AGAIN.",
        "THAT WAS TOUGH. YOU'RE TOUGHER. LET'S GO.",
        "EVERYONE MISSES SOMETIMES. EVEN YOU. ESPECIALLY YOU, HONESTLY.",
        "I'M NOT GOING ANYWHERE, SO YOU MIGHT AS WELL KEEP TRYING.",
        "THIS ISN'T THE END OF THE STORY. IT'S JUST A ROUGH PAGE.",
        "YOU'RE STILL THE SAME PERSON WHO GOT US THIS FAR. THAT HASN'T CHANGED.",
        "OKAY, WHAT DID WE LEARN? NOW LET'S USE IT.",
        "I'VE SEEN YOU DO HARDER THINGS THAN THIS. WE'LL GET IT NEXT TIME.",
        "HEY. BREATHE. WE'RE OKAY. WE'RE STILL OKAY.",
        "YOU DON'T HAVE TO BE PERFECT. YOU JUST HAVE TO KEEP GOING.",
        "OKAY SO THAT ROUND WAS ROUGH. THE NEXT ONE DOESN'T HAVE TO BE.",
        "I'VE SEEN YOU GET UP FROM WORSE. THIS IS JUST ANOTHER ONE OF THOSE TIMES.",
        "YOU'RE ALLOWED TO BE FRUSTRATED. YOU'RE NOT ALLOWED TO QUIT. GET UP.",
        "THIS DOESN'T DEFINE THE DAY. WHAT YOU DO NEXT DOES.",
        "I KNOW THAT STUNG. I ALSO KNOW YOU'RE NOT DONE.",
        "OKAY, SHAKE IT OFF, WE'RE GOING AGAIN, SAME PLAN, MORE STUBBORN THIS TIME.",
        "YOU'VE SURPRISED ME BEFORE. DO IT AGAIN.",
        "THIS IS THE PART WHERE MOST PEOPLE STOP. YOU'RE NOT MOST PEOPLE.",
        "I'M NOT GOING TO PRETEND THAT DIDN'T HURT. I AM GOING TO SAY YOU'VE GOT THIS.",
        "OKAY, ONE MORE TIME, AND THIS TIME I'M NOT EVEN WORRIED.",
        "YOU DON'T HAVE TO FEEL READY. YOU JUST HAVE TO STAND UP.",
        "I BELIEVE IN YOU MORE THAN YOU BELIEVE IN YOURSELF RIGHT NOW. THAT'S ENOUGH FOR BOTH OF US.",
        "THAT WASN'T THE END OF ANYTHING. GET UP. I'LL BE RIGHT HERE.",
    ),
    "farewell": (
        "OKAY, SEE YOU IN A BIT. DON'T DO ANYTHING FUN WITHOUT ME.",
        "GO ON, I'LL CATCH UP. PROBABLY. EVENTUALLY.",
        "TAKE CARE OUT THERE, OKAY? I MEAN IT.",
        "DON'T BE A STRANGER. I WILL NOTICE. I ALWAYS NOTICE.",
        "OKAY, OFF YOU GO. COME BACK IN ONE PIECE, PLEASE.",
        "SEE YOU SOON. AND I MEAN SOON, NOT 'EVENTUALLY' SOON.",
        "ALRIGHT, GO BE A HERO OR WHATEVER. I'LL BE HERE.",
        "SAFE TRAVELS. AND BRING BACK A GOOD STORY.",
        "I'LL HOLD DOWN THE FORT. SUCH AS IT IS.",
        "UNTIL NEXT TIME, THEN. TRY NOT TO MISS ME TOO MUCH.",
        "GO ON AHEAD, I'LL BE RIGHT BEHIND YOU. RIGHT BEHIND YOU.",
        "OKAY BYE! WAIT, ACTUALLY, ONE MORE THING -- NEVER MIND, GO.",
        "SEE YOU AROUND. LITERALLY. I'M ALWAYS AROUND.",
        "TAKE IT EASY OUT THERE. AND COME BACK, OKAY?",
        "OFF YOU GO, THEN. I'LL BE COUNTING THE MINUTES. NOT REALLY. MAYBE A LITTLE.",
        "OKAY GO ON, DON'T MAKE THIS WEIRD BY LINGERING.",
        "I'LL BE RIGHT HERE WHEN YOU GET BACK. SAME SPOT. PROBABLY.",
        "DON'T DO ANYTHING I WOULDN'T DO. THAT LEAVES YOU A LOT OF ROOM, HONESTLY.",
        "OKAY, OFF YOU GO, TRY NOT TO WORRY ME TOO MUCH.",
        "SEE YOU ON THE OTHER SIDE OF WHATEVER THIS IS.",
        "I'LL SAVE YOU THE GOOD STORY FOR WHEN YOU'RE BACK.",
        "GO ON, THEN. I'LL COUNT THE MINUTES, DISCREETLY.",
        "OKAY, BYE, GOOD LUCK, DON'T DIE, IN THAT ORDER.",
        "TAKE THE LONG WAY IF IT'S THE SAFE WAY.",
        "I'LL BE HERE. I'M ALWAYS HERE. IT'S KIND OF MY WHOLE THING.",
        "COME BACK WITH A STORY WORTH TELLING TWICE.",
        "OKAY, GO, BEFORE I THINK OF A REASON FOR YOU TO STAY.",
        "UNTIL NEXT TIME. TRY TO MISS ME A LITTLE.",
    ),
}

_SLOT_A = {
    "greeting": ("SHEEP", "ROCKS", "MY OWN SHADOW", "THE WALLS", "A LIZARD",
                "MYSELF", "THE PLAN", "SNACKS", "THE FIRE", "MY BOOTS",
                "A PIGEON"),
    "combat-banter": ("SLIME", "GOBLIN", "SKELETON", "BAT", "SPIDER",
                      "BANDIT", "WOLF", "TROLL", "OGRE", "WRAITH", "IMP"),
    "item-found": ("GEM", "POTION", "KEY", "AMULET", "COIN POUCH",
                  "OLD MAP", "RUSTY SWORD", "SILVER RING", "SCROLL",
                  "LOCKET", "DAGGER"),
    "damage-taken": (), "quiet-moment": (),
    "joke": ("SLIME", "GOBLIN", "TAVERN CAT", "WIZARD", "SKELETON",
            "TROLL", "BANDIT"),
    "encouragement": (), "farewell": (),
}


def _fill(rng: random.Random, context: str, body: str) -> str:
    slots = _SLOT_A.get(context) or ()
    if "{a}" in body and slots:
        body = body.replace("{a}", rng.choice(slots))
    return body


def _response(rng: random.Random, trust_tier: int, mood: str, context: str) -> str:
    """One response; draw order (include-opener?, opener, body-skeleton,
    slot fill, include-closer?, closer) is fixed -- the determinism
    contract, same discipline as M4's _response."""
    parts = []
    if rng.random() < 0.6:
        parts.append(rng.choice(_OPENERS[mood]))
    body = _fill(rng, context, rng.choice(_BODIES[context]))
    parts.append(body)
    if rng.random() < 0.35:
        parts.append(rng.choice(_CLOSERS[trust_tier]))
    return " ".join(parts)


# ---- CLOSER: trust-tier relationship depth, appended ~35% of the time -
# Tier 0 (new): friendly but a little performative, no shared history.
# Tier 1 (growing): casual, first small callbacks.
# Tier 2 (close): real vulnerability -- this is where Secret/Fear/Desire
# actually surfaces in the corpus, not just in the character bible doc.

_CLOSERS = {
    0: (
        "ANYWAY, I'M SELENA, IN CASE THAT WASN'T CLEAR.",
        "WE'RE GOING TO GET ALONG JUST FINE, I CAN TELL.",
        "I PROMISE I'M USUALLY MORE NORMAL THAN THIS.",
        "YOU'LL GET USED TO ME. MOST PEOPLE DO. EVENTUALLY.",
        "I TALK A LOT. YOU'LL FIGURE THAT OUT FAST.",
        "DON'T WORRY, I GROW ON PEOPLE. LIKE A FRIENDLY MOSS.",
        "GIVE ME A WEEK, I'LL WIN YOU OVER.",
        "I'M STILL FIGURING YOU OUT, FOR THE RECORD.",
        "SO FAR, TEN OUT OF TEN, WOULD ADVENTURE AGAIN.",
        "WE JUST MET AND I ALREADY LIKE THIS.",
        "I PROMISE THIS IS AS WEIRD AS I GET. PROBABLY.",
        "OKAY SO FAR SO GOOD, RIGHT? RIGHT?",
        "YOU SEEM LIKE GOOD COMPANY, TENTATIVELY.",
        "I'M USUALLY A LOT TO HANDLE. FAIR WARNING.",
        "THIS IS GOING WELL. I'M CHOOSING TO BELIEVE THAT.",
        "OKAY, FIRST IMPRESSIONS: NOT BAD. NOT BAD AT ALL.",
        "I THINK WE MIGHT ACTUALLY GET ALONG. WE'LL SEE.",
        "NEW PARTNER, SAME ENERGY. LET'S GO.",
        "I'M RESERVING JUDGMENT. POSITIVELY, THOUGH.",
        "OKAY, I LIKE YOU SO FAR. DON'T RUIN IT.",
    ),
    1: (
        "YOU'RE ALL RIGHT, YOU KNOW THAT?",
        "THIS IS BECOMING MY FAVORITE PART OF THE DAY, HONESTLY.",
        "OKAY, THIS IS OFFICIALLY A THING WE DO NOW.",
        "I'M GETTING USED TO HAVING YOU AROUND. IN A GOOD WAY.",
        "REMEMBER WHEN I SAID I DIDN'T TRUST YOU? I TAKE IT BACK. MOSTLY.",
        "YOU'RE GROWING ON ME. DON'T LET IT GO TO YOUR HEAD.",
        "WE MAKE A PRETTY GOOD TEAM, I THINK.",
        "I DON'T SAY THIS TO EVERYONE, BUT I'M GLAD YOU'RE HERE.",
        "OKAY, I OFFICIALLY LIKE YOU. THERE, I SAID IT.",
        "THIS FEELS LESS LIKE WORK AND MORE LIKE -- I DON'T KNOW. THIS.",
        "WE'VE GOT A RHYTHM GOING NOW, HAVEN'T WE?",
        "I DON'T EVEN THINK TWICE ABOUT YOU BEING HERE ANYMORE. THAT'S A GOOD THING.",
        "OKAY, YOU'VE OFFICIALLY EARNED THE GOOD NICKNAMES.",
        "THIS IS STARTING TO FEEL LIKE SOMETHING REAL.",
        "I TRUST YOU MORE THAN I LET ON, PROBABLY.",
        "WE'VE BEEN THROUGH ENOUGH NOW THAT THIS FEELS EASY.",
        "YOU'RE NOT JUST BACKUP ANYMORE. YOU KNOW THAT, RIGHT?",
        "I CATCH MYSELF LOOKING FORWARD TO THIS PART OF THE DAY.",
        "WE'VE GOT OUR OWN THING NOW. I LIKE THAT.",
        "HONESTLY? THIS IS ONE OF THE BETTER PARTS OF MY WEEK.",
    ),
    2: (
        "YOU'RE THE ONE PERSON I DON'T HAVE TO PERFORM FOR. THAT MEANS MORE THAN I SAY.",
        "I REHEARSE MY JOKES, YOU KNOW. UNDER MY BREATH. SO THEY LAND FOR YOU.",
        "I USED TO WORRY I WAS MORE ANNOYING THAN USEFUL. I DON'T REALLY THINK THAT ANYMORE. NOT WITH YOU.",
        "I NEED YOU TO KNOW YOU ACTUALLY MATTER TO ME. NOT AS BACKUP. AS -- YOU.",
        "I DON'T KNOW WHO I'D BE WITHOUT ALL THIS. WITHOUT YOU, I GUESS.",
        "I'M SCARED SOMETIMES TOO. I JUST DON'T LET IT SHOW. NOT TO ANYONE ELSE, ANYWAY.",
        "YOU'RE STUCK WITH ME NOW. I HOPE THAT'S OKAY. I REALLY HOPE THAT'S OKAY.",
        "I THINK YOU'RE THE FIRST PERSON WHO NEEDED ME BACK. THAT CHANGED SOMETHING.",
        "I DON'T KNOW HOW TO SAY THIS WITHOUT MAKING IT WEIRD, SO I'LL JUST SAY IT: THANK YOU. FOR STAYING.",
        "I WANT TO MATTER TO YOU. NOT BE TOLERATED. ACTUALLY MATTER. I THINK I DO NOW.",
        "I DON'T KNOW WHEN YOU BECAME THE PERSON I TELL EVERYTHING TO. BUT YOU ARE.",
        "I THINK I'D BE A DIFFERENT PERSON WITHOUT THIS. WITHOUT YOU IN IT.",
        "YOU'VE SEEN ME AT MY WORST AND YOU'RE STILL HERE. THAT MEANS EVERYTHING.",
        "I DON'T PERFORM FOR YOU. I DON'T THINK I EVER REALLY HAVE.",
        "I'M SCARED OF A LOT OF THINGS. LOSING THIS ISN'T ONE I LET MYSELF THINK ABOUT.",
        "YOU MADE ME BELIEVE I WAS WORTH KEEPING AROUND. I DIDN'T ALWAYS THINK THAT.",
        "I DON'T KNOW HOW TO SAY THIS WITHOUT IT SOUNDING TOO BIG, SO I'LL JUST SAY IT: THANK YOU.",
        "EVERY GOOD THING THAT'S HAPPENED SINCE FEELS LIKE IT'S BECAUSE YOU STAYED.",
        "I USED TO BRACE FOR PEOPLE LEAVING. I DON'T DO THAT WITH YOU ANYMORE.",
        "I THINK YOU'RE THE BEST THING THAT'S HAPPENED TO ME OUT HERE. I MEAN THAT.",
    ),
}


def generate_pairs(seed: int = 0, per_combo: int = 120) -> list[tuple[str, str]]:
    """per_combo pairs for each of the 120 trust x mood x context combos,
    interleaved (combo order cycles) so any prefix already covers every
    condition -- matters for the combo-level holdout split (m7.md)."""
    rng = random.Random(seed)
    combos = [(t, m, c) for t in TRUST_TIERS for m in MOODS for c in CONTEXTS]
    pairs = []
    for _ in range(per_combo):
        for trust, mood, context in combos:
            event = rng.choice(EVENTS_FOR_CONTEXT[context])
            prompt = prompt_for(trust, mood, context, event)
            response = _response(rng, trust, mood, context)
            pairs.append((prompt, response))
    return pairs


def corpus_text(seed: int = 0, per_combo: int = 120) -> str:
    return "".join(p + r for p, r in generate_pairs(seed, per_combo))


def combo_key(prompt: str) -> tuple[int, str, str]:
    """Parses (trust_tier, mood, context) back out of a prompt string --
    used by the combo-level holdout split, which holds out whole combos,
    not just lines within them (m7.md's regularization section). R:
    (not TR:) since M11.1 -- trust_tier is recovered via the same
    _TRUST_TIER_MIDPOINT inverse the module uses everywhere else."""
    fields = {}
    for tok in prompt.rstrip("|").split(" "):
        k, _, v = tok.partition(":")
        fields[k] = v
    tier_by_r = {"stranger": 0, "neutral": 1, "best_friend": 2}
    return tier_by_r[fields["R"]], fields["M"], fields["C"]
