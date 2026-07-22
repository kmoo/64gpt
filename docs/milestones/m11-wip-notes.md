# M11 finish-out — working notes (crash recovery)

**Not a milestone doc.** Live scratch log for the "finish M11" push
(plan approved 2026-07-22, session plan file "reflective-wishing-
yeti" under `~/.claude/plans/`). Delete this file once M11 lands and
`docs/milestones/m11.md` has the real writeup. Updated as work lands,
not just at the end, so a crash mid-session doesn't lose the state of
what was actually done vs. still pending.

## Done, committed, pushed to `main`

1. **Stale version-label fix** (`c920869`) — `DialogueDemo.cpp` had 6
   hardcoded per-slot strings like `"64GPT V1.2 - SELENA (M7)"`, frozen
   at whichever milestone introduced that slot. Replaced with one
   `NGPT_VERSION = "V1.6"` constant + role-only labels (no bare
   milestone numbers in any slot's display string anymore).
2. **Milestone split** (already landed prior session, `a633c08`) — M11
   keeps town cast/dungeon loop/quality push; portability packaging
   moved to `docs/milestones/m12.md`.

## Done, NOT yet committed (working tree)

- `docs/ideas-briar-glen-world.md` (new) + `docs/img/briar-glen-map.jpg`
  (new, resized/compressed from the original 3.5MB source) — the Briar
  Glen vision doc, cross-linked from `docs/plan.md` and `docs/ideas.md`.
- **Two new archetypes**: `MERCHANT_ARCHETYPE` (occupation `"merchant"`,
  descriptor calibrates to `"measured"`) and `HEALER_ARCHETYPE`
  (occupation `"healer"`, descriptor calibrates to `"gentle"`) in
  `game/src/user/NPCDatabase.h`/`.cpp`. Both verified via
  `./build/test_npc_service` output: `D:measured OCC:merchant` /
  `D:gentle OCC:healer`, exactly as calibrated by hand.
  - Wired into `DungeonGenerator.cpp`'s `ARCHETYPE_POOL` (5→7 entries)
    — `tests/test_dungeon_generator.cpp`'s own local pool copy updated
    to match (this broke once, fixed).
  - Wired into `DialogueDemo.cpp`'s fixed roster: `NEW_ARCHETYPE_COUNT`
    4→6, arrays extended (labels `"MERCHANT"`/`"HEALER"`, seeds
    `0x2005`/`0x2006`).
  - `manifests/dungeon_crawler.json`'s `archetypes[]` extended with both.
  - `cast_corpus.py`: new `merchant_rep`/`healer_rep` CHARACTERS
    entries, new `_OCCUPATION_FLAVOR["merchant"/"healer"]` banks, new
    `_DESCRIPTOR_TICS["gentle"]` bank (measured already existed).
  - Tests updated: `tests/test_npc_service.cpp`'s archetype-coverage
    loop, `trainer/tests/test_cast_corpus.py`'s `_CANON_DESCRIPTOR` +
    density-set assertion.
  - **Host suite: 10/10 green. Trainer suite: 176/176 green** (last
    checked after this batch).
- **The princess (Elowen)**: `trainer/ngpt_trainer/princess_corpus.py`
  (new) — mid-tier, old N: scheme like Korrath, `per_combo=4` (same as
  Korrath). Own OPENERS/BODIES/CLOSERS (hopeful/scared-but-brave/
  grateful voice), no catchphrase bank (same reasoning as Korrath's).
  `trainer/tests/test_princess_corpus.py` (new, 9 tests, mirrors
  `test_korrath_corpus.py`) — all passing, including a check that her
  voice banks are fully disjoint from Shadewrath's/Korrath's own.
  **NOT yet wired into**: `NPCDatabase.h`/`.cpp` (no `NPC princess`/
  `NPC elowen` global yet), `manifests/dungeon_crawler.json` (no bible
  entry yet), `DialogueDemo.cpp` (no dungeon slot yet), `SaveData.h`/
  `.cpp` (no `princessHighestTier` field yet), `WorldState.h`/`.cpp`
  (no 3rd gossip event yet).
- **Quality-push shared lore bank**: `trainer/ngpt_trainer/
  ravendale_lore.py` (new) — `RAVENDALE_LORE` tuple, 10 voice-neutral
  lore lines. Spliced into `shadewrath_corpus.py`'s and
  `korrath_corpus.py`'s `_response()` (20% draw chance, one more
  optional clause alongside their own bespoke banks — NOT a wholesale
  voice-bank reuse). Verified via `test_princess_corpus.py`'s
  `test_ravendale_lore_lines_actually_appear_in_generated_output` that
  the splice actually fires for all three modules at realistic sample
  sizes.

## Not started yet

- Design refinement vs. original plan text: SaveData gets
  `princessHighestTier` (uint8, mirrors `shadewrathHighestTier`/
  `korrathHighestTier`), NOT a plain boolean `princessRescued` byte as
  the plan file literally says — reaching her TR:2 for the first time
  IS the rescue event, same mechanism Shadewrath/Korrath already use,
  rather than inventing a second kind of persisted state. Still fits
  the existing 6 bytes of `SaveFile` padding, no format change.
- `NPCDatabase::princess` (or `elowen`) NPC global, tier `MID`.
- `manifests/dungeon_crawler.json` princess bible entry.
- `DialogueDemo.cpp`: princess dungeon slot (`isDungeonPrincessSlot()`,
  `DUNGEON_SLOT_COUNT` `NPCS_PER_LEVEL+1` → `+2`), gossip trigger #3
  (`"princess_freed"`), display line (a `RESCUED`-style suffix once her
  tier maxes, mirroring `MET TR:N`). **Currently being delegated to
  opencoder (local model) — mechanical, pattern-following C++ mirroring
  the existing `isDungeonBadGuySlot()` shape.**
- `WorldState::GOSSIP_EVENTS` 2→3 (`"princess_freed"`), matching
  `cast_corpus.py`'s `GOSSIP_EVENTS`/`_GOSSIP_LINES` additions.
- `SaveData.h`/`.cpp`: `princessHighestTier` field + recorder function.
- One combined retrain (new blob script, exact filename TBD — see
  plan's naming note) covering: merchant/healer archetypes, the
  princess, the shared lore bank. Measure against baselines: val loss
  0.0992, agreement 0.9771, the divergence table, generalization check.
- Re-run/extend `eval_shadewrath_long_horizon.py` against the new
  checkpoint.
- Real ROM build, Ares boot, `NGPT_SELFTEST_ENABLED=1` confirmed,
  screenshot proving `SELFTEST PASS`/`RSP ON`/`XCHK PASS`.
- Manual interactive Ares check: new archetype slots, princess rescue
  in a generated dungeon level, gossip `NEWS` suffix, Shadewrath/
  Korrath `MET TR:N` still correct.
- `docs/milestones/m11.md` final writeup + DoD checkboxes + version bump
  in README/`versions/README.md` + `versions/m11_64gpt.z64` + `talk/`
  screenshot + tag `m11`.
- Delete this file once the real writeup lands.

## Key IDs/constants to remember across a crash

- Princess name: **Elowen**, `N:elowen` (NPC_ID in `princess_corpus.py`).
- `NGPT_VERSION = "V1.6"` (bump again if another milestone ships before
  M11 tags).
- New archetype seeds: merchant `0x2005`, healer `0x2006` (fixed
  roster showcase instances, `DialogueDemo.cpp`).
- `RAVENDALE_LORE` draw chance: 20% in all three modules (shadewrath/
  korrath/princess `_response()`).
- Baselines to beat: val loss 0.0992, agreement 0.9771 (M11's gossip
  retrain, already shipped/tagged... **not tagged**, just pushed to
  `main` — M11 itself is still open).
