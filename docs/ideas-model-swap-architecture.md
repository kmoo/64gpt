# Ideas — a three-tier *model* architecture (post-M10 brainstorm, 2026-07-17)

**Not a roadmap.** Like `docs/ideas.md`, this is a speculative idea for
later, not a milestone plan — nothing here is scheduled, and the current
single-shared-model approach is not considered broken or superseded by
anything in this document.

## Background — why this came up

M10 added Shadewrath (a bespoke full-tier villain) and four new thin-tier
town archetypes (`pub_patron`/`blacksmith`/`wizard`/`villager`) to the
*same* shared model Selena and guard already live in. That's a content
decision, not an architecture one — every character in this project, from
M2 onward, has been a prompt-conditioned voice inside one resident set of
weights. Working through *how many* characters that one model was being
asked to carry raised a genuinely separate question this doc exists to
capture before it's lost: at what point, if ever, does it stop being one
model for everyone, and how would swapping between multiple *model files*
actually work on this hardware.

**This is not the same idea as the existing three-tier character system**
(`full`/`mid`/`thin`, `docs/08-manifest-schema.md`, `docs/milestones/
m10.md`). That system is about *prompt content* — how much bespoke bible
and corpus a character gets, all inside one set of weights. The idea
below is about *weight files* — whether some characters eventually get
their own separately-loaded model instead of sharing one. The names
rhyme; the mechanisms don't. Worth remembering both are "three tiers" for
different things.

## The core idea

Map model granularity to the actual cost structure of an EverDrive-64 +
SD card setup, in three tiers:

### 1. One general/shared model (today's architecture)

Single resident model, every character prompt-conditioned via identity/
mood/context/occupation tags (M7-M10's whole mechanism). Zero swap cost
— nothing ever loads mid-session. Covers unlimited archetype instances
for free via seeded personality jitter (`NPCDatabase::spawnInstance()`).
Ceiling is the RSP's fixed 4KB DMEM tile budget and console RAM (4MB
stock / 8MB with the Expansion Pak), not storage — this is the thing M9's
H=320 kernel is already up against, 320 bytes of DMEM headroom left
(`docs/spikes/rsp-matvec-h320.md`).

### 2. One model per base archetype/NPC-type

A middle tier: `guard`, `merchant`, `wizard`-the-tinker, etc. each get
their own smaller, more specialized model file on the SD card, swapped
into RAM/DMEM only while that type's NPC is actually being talked to.
Tradeoff to weigh if this is ever pursued: swap latency per archetype-
type change (SD/flashcart DMA is fast but not instant — there's no
measured number for this yet, `docs/hardware-checklist.md`'s own
per-step timings are all single-resident-model numbers) against
potentially sharper, more specialized voice per archetype than one
shared model's prompt-conditioning capacity has to spend on all of them
at once.

### 3. One model per named individual

The most granular tier — Selena, Shadewrath, any other full-tier
character — for the handful of cases where a fully dedicated model might
actually justify its own swap cost. Named individuals are met
infrequently relative to how often a fresh archetype instance gets
spawned, so this is the tier where a swap's fixed cost is easiest to
amortize, if it's ever worth paying at all.

## Constraints that actually anchor this tradeoff

- **SD card capacity solves storage, nothing else.** An EverDrive setup
  can hold far more weight data than a real N64 cartridge ever could —
  the current ROM is 704,512 bytes total and that's "not remotely the
  binding constraint yet" (`docs/hardware-checklist.md`). "Too many
  model files" is not a real limit under this architecture.
- **It does not solve the RAM problem.** Console RAM is fixed at 4MB (8MB
  with the Expansion Pak) regardless of how much sits on the SD card.
  Whichever model is actively resident still has to fit inside that.
- **It does not solve the RSP DMEM problem.** The RSP's scratchpad is a
  hardwired 4KB, already down to 320 bytes of headroom at H=320
  (`docs/spikes/rsp-matvec-h320.md`) — a bigger *or additional*
  concurrently-resident model doesn't get more DMEM just because the SD
  card has room for its weights.
- **Swap latency is reduced, not eliminated, versus a real cartridge's
  ROM reads.** DMA-loading a fresh weight blob costs real time relative
  to a model that's already resident. If this is ever built, the actual
  swap-triggering events (which archetype type, which named character
  you've started talking to) need to be chosen deliberately to avoid a
  visible hitch — this needs a measured number before it's a real
  design, not an assumed one.
- **This is a "future, if constraints loosen or the cast grows large
  enough to justify it" idea, not a near-term need.** The single-shared-
  model-plus-conditioning approach was chosen after M7 specifically
  measured that one model can carry more than one distinct voice
  (identity-swap divergence ≥ mood-swap divergence, the actual decision
  gate that unblocked M8's archetype system) — nothing in M10's work or
  in this document invalidates that result. This idea is worth having
  written down for when/if the cast outgrows what prompt-conditioning
  alone can carry, not a signal that day has arrived.
