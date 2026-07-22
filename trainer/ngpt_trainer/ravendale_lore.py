"""M11 quality push: a shared phrase bank for the three characters whose
own bibles are narratively linked through the same backstory --
Shadewrath (the necromancer captor), Korrath (his bound knight guard),
and Elowen (the captive princess). See docs/plan.md's Known follow-ups
for the root-cause hypothesis this targets: unlike guard/the
compositional cast, none of these three characters' content was
reinforced by ANY shared bank, so each competed for the model's fixed
H=320 capacity in total isolation -- the documented, untried next lever
was "share more structural content across characters."

Mechanism, deliberately narrow: this is ONE additional shared CLAUSE
type (a short lore-reference line), appended probabilistically on top
of each character's own bespoke OPENERS/BODIES/CLOSERS -- not a
wholesale reuse of another character's full voice bank, which
shadewrath_corpus.py's own module docstring already identified as the
voice-mismatch failure mode to avoid (Selena's casual companion _BODIES
read badly coming from a centuries-patient necromancer). These lines
are deliberately voice-neutral -- short, factual-flavored statements
about the shared world (the binding, Ravendale, the sealed door) that
any of the three could plausibly say without clashing with their own
distinct tone, which each character's own banks still carry.
"""

RAVENDALE_LORE = (
    "RAVENDALE HAS KEPT ITS SECRETS FOR CENTURIES.",
    "THE BINDING DOES NOT LOOSEN, EVEN WITH TIME.",
    "THE DOOR BENEATH RAVENDALE HAS BEEN SEALED SINCE BEFORE THIS KINGDOM HAD A NAME.",
    "SOME THINGS DOWN HERE ARE OLDER THAN THE DUNGEON ITSELF.",
    "THE PRINCESS ONCE WALKED RAVENDALE'S HALLS FREELY. NOT ANYMORE.",
    "EVERYTHING IN THIS PLACE TRACES BACK TO THE SAME OLD DOOR.",
    "CENTURIES OF WAITING CHANGE A PLACE. AND EVERYONE IN IT.",
    "RAVENDALE REMEMBERS WHAT THE REST OF THE WORLD HAS FORGOTTEN.",
    "THE BINDING WAS NEVER MEANT TO LAST THIS LONG.",
    "WHAT'S SEALED BENEATH RAVENDALE HAS BEEN WAITING LONGER THAN ANY OF US.",
)
