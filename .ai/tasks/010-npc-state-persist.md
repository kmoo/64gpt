# Task 010 — npc-state-persist

## CONTRACT

```yaml
id: 010-npc-state-persist
goal: >
  Implement the pure, host-testable core of "living NPC state" step 1
  (docs/ideas-living-npc-state.md section 6): a per-NPC relationship
  vector (familiarity/affection/trust/respect/fear) and an 8-slot
  episodic memory block, plus the deterministic logic to update them --
  apply a relationship delta, record a new memory (with eviction when
  full), age/decay memories over time, and select the top-N memories by
  salience. No EEPROM/SD I/O, no serialization format, no EventBus
  wiring yet -- those are separate, later, hardware-only work. This
  piece must compile and be fully tested on the host with zero
  hardware dependency.
background: >
  This project (64GPT) already has one persistence mechanism,
  game/src/user/SaveData.h, split per ADR 0001
  (docs/adr/0001-host-test-portable-cpp-separate-from-libdragon.md):
  the real EEPROM read/write code depends on libdragon and has NO host
  test coverage, but the pure decision logic
  (SaveData::isNewHighWaterMark) is extracted into a header-only,
  libdragon-free function and IS host-tested
  (tests/test_save_data.cpp). This task follows that exact same split:
  everything in this task is the pure-logic half, header-only, no
  libdragon include anywhere. game/src/user/NPCDatabase.h is this
  project's existing per-NPC data-struct style reference (plain
  `int` fields, no floats, no heap) -- match that style, not
  core/'s int8/fixed-point conventions (core/ is model-inference-only
  and off limits to this task entirely).
constraints: |
  - Portable C++, header-only (a single .h, no .cpp) -- must compile
    with zero libdragon includes, exactly like SaveData.h's pure half.
  - No floats, no heap allocation (new/delete/malloc), no exceptions.
    Fixed-size arrays only (MEMORY_SLOTS below).
  - Plain `int`/`uint32_t` fields, matching NPCDatabase.h's existing
    style (its `personality[TRAIT_COUNT]` array is `int`, not a
    fixed-point type -- this task's fields are the same kind of
    plain game-state int, not the core/ int8-quantized model weights).
  - Do NOT touch core/, game/src/user/SaveData.h/.cpp,
    game/src/user/NPCDatabase.h/.cpp, or any other existing file --
    this task only adds one new header.
  - Do NOT design or implement any serialization/byte-packing format,
    EEPROM/SD I/O, or EventBus wiring -- explicitly out of scope for
    this task (see docs/ideas-living-npc-state.md's own "suggested
    first spike" list -- this task is step 1 only, the data structures
    and their pure update logic).
allowed_files:
  - game/src/user/NPCState.h
  - tests/test_npc_state.cpp
reference_files:
  - game/src/user/SaveData.h
  - game/src/user/NPCDatabase.h
  - tests/test_save_data.cpp
  - tests/test_util.h
test_files:
  - tests/test_npc_state.cpp
acceptance_criteria:
  - >
    `namespace NPCState { constexpr int MEMORY_SLOTS = 8; }` and a
    `struct Relationship { int familiarity; int affection; int trust;
    int respect; int fear; };` with exactly these five fields, these
    exact names, in this order.
  - >
    `struct Memory { uint32_t eventId; int salience; int confidence;
    uint32_t ageTicks; };` -- `eventId == 0` means an empty/unused slot
    (0 is never a valid real event id, matching NPCDatabase's own
    `memorySlot`'s "0 = empty/unused" convention).
  - >
    `struct MemoryBlock { Memory slots[MEMORY_SLOTS]; };` -- a
    default-constructed (or zero-initialized) MemoryBlock has every
    slot's eventId == 0 (all empty).
  - >
    `void applyDelta(Relationship &rel, int dFamiliarity, int dAffection,
    int dTrust, int dRespect, int dFear);` adds each delta to the
    matching field then clamps: familiarity/trust/respect/fear to
    [0, 100] inclusive, affection to [-100, 100] inclusive. Order of
    operations is add-then-clamp, not clamp-then-add.
  - >
    `void recordMemory(MemoryBlock &block, uint32_t eventId, int salience,
    int confidence);` clamps salience and confidence to [0, 100] first.
    If any slot has eventId == 0 (empty), write into the FIRST such slot
    (lowest array index) with ageTicks = 0. If no slot is empty, evict
    the slot with the lowest `salience`; if multiple slots tie for
    lowest salience, evict the one with the highest `ageTicks` among
    those tied (oldest); if still tied, evict the lowest array index.
    The evicted slot is overwritten with the new memory, ageTicks = 0.
  - >
    `void ageMemories(MemoryBlock &block, uint32_t ticksElapsed);` for
    every slot with eventId != 0: `ageTicks += ticksElapsed`; then
    `salience -= ticksElapsed / 10` using integer division (so
    `DECAY_TICKS_PER_SALIENCE_POINT = 10` ticks costs exactly 1
    salience point), clamped to a floor of 0. Empty slots (eventId == 0)
    are left completely untouched (ageTicks and salience stay whatever
    they were, since the slot is unused).
  - >
    `int selectTopMemories(const MemoryBlock &block, int n,
    uint32_t outEventIds[]);` considers only slots with eventId != 0,
    sorts them by descending `salience`; ties broken by ascending
    `ageTicks` (more recently recorded/refreshed wins); remaining ties
    broken by ascending array index. Writes up to `n` winning eventIds
    into `outEventIds` in that sorted (best-first) order and returns
    the actual number written (`min(n, count of non-empty slots)`).
    Does not modify `block`.
  - >
    tests/test_npc_state.cpp (matching tests/test_save_data.cpp's style
    -- plain `static void test_*()` functions, `CHECK`/`CHECK_EQ_INT`
    from test_util.h, a `main()` that calls each test and returns
    `test_summary(...)`) covers at minimum: applyDelta adds and clamps
    correctly at both ends of every axis's range (including affection's
    negative floor); recordMemory fills empty slots in index order
    before evicting anything; recordMemory evicts the correct slot
    under a real lowest-salience tie broken by ageTicks (construct a
    MemoryBlock with two equal-lowest-salience slots, differing
    ageTicks, and confirm the older one is evicted); ageMemories
    advances ageTicks and decays salience by exactly the documented
    integer-division rule (including a case where `ticksElapsed` is not
    an exact multiple of 10) and never drops salience below 0; empty
    slots are untouched by ageMemories; selectTopMemories returns
    memories in the documented sorted order including a salience tie
    broken by ageTicks, and returns fewer than `n` when fewer than `n`
    slots are occupied.
verification:
  - "rm -rf build && cmake -B build tests -DNGPT_SANITIZE=OFF && cmake --build build --target test_npc_state && ctest --test-dir build -R test_npc_state --output-on-failure"
```

## COMPLETION

```yaml
status: done
summary: |
  1 dispatch attempt, discarded, module written directly by lead. The
  test-designer role (blind to the implementation, per split-mode)
  misunderstood the split: instead of writing tests that #include
  "NPCState.h" and exercise it, it redeclared the entire SUT (struct
  Relationship/Memory/MemoryBlock, all four function signatures) directly
  inside tests/test_npc_state.cpp, then opened the file with `#pragma
  once` as if the .cpp itself were the header -- a hard compile error
  under -Werror -Wpragma-once-outside-header. Separately, the
  programmer's NPCState.h (attempt 1) only had function DECLARATIONS,
  no definitions, and was missing #include <stdint.h> -- would have
  failed to link even past the .cpp's compile error. Given the structural
  misunderstanding (not a small format slip) plus a related real gap in
  task 011 that day, wrote NPCState.h + test_npc_state.cpp directly,
  mirroring SaveData.h/test_save_data.cpp's real style.
files_changed:
  - game/src/user/NPCState.h
  - tests/test_npc_state.cpp
  - tests/CMakeLists.txt (test_npc_state target wiring, pre-existing from
    worktree setup, carried through the main merge)
verification: |
  rm -rf build && cmake -B build tests -DNGPT_SANITIZE=OFF && cmake --build build && ctest --test-dir build --output-on-failure
  100% tests passed out of 15 (full suite, no regressions)
risks: []
needs_review: []
```

## METRICS

- dispatches / retries / escalated: 1 / 0 / no (lead-authored after attempt 1)
- claude tokens spent (contract + review, est.) vs doing it directly: the acceptance_criteria for this task were already fully deterministic (exact struct layout, exact tie-break rules for recordMemory/selectTopMemories) -- writing directly from that spec was likely cheaper than a contract-authoring + dispatch + review cycle would have been, in hindsight
- defects: caught in review = 2 (test file redeclaring the SUT instead of including it; header with declarations but no definitions), slipped past review = 0

## WORKER RESULT (qwen-worker) — attempt 1, discarded

test-designer redeclared the whole SUT inside the test .cpp (see
COMPLETION summary above) and opened it with `#pragma once`, which
fails to compile under -Werror -Wpragma-once-outside-header
(`#pragma once in main file`). programmer's NPCState.h separately had
only function declarations, no bodies. Both discarded; written directly.
