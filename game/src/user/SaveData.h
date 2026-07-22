#pragma once
#include <stdint.h>

// M10: the walking-skeleton persistence mechanism -- the smallest real
// piece that proves "the bad guy remembers you across separate
// encounters" (docs/milestones/m10.md section 3) rather than leaving
// isPersistent() a correct rule with nothing plugged into it. Saves
// exactly one 8-byte EEPROM block: the highest trust tier ever reached
// with Shadewrath and Korrath (the two named M10 individuals whose tier
// carries real narrative weight -- his CLOSERS escalate by tier, see
// trainer/ngpt_trainer/shadewrath_corpus.py). Not a general save-
// anything system; deliberately scoped to the one piece of state that
// demonstrates the mechanism end to end.
//
// Built on libdragon's EEPROM filesystem (eepromfs.h), not raw EEPROM
// block access -- eepfs handles checksums, corruption detection, and
// signature validation (garbage/first-boot/foreign-cart EEPROM) so this
// module doesn't have to reinvent any of that.
//
// SaveFile/init()/recordXTier() below have no host-test coverage possible
// (real hardware peripheral, same as DialogueDemo.cpp) -- verified via a
// real Pyrite64 build + Ares boot, exercised through an actual write-
// then-relaunch cycle, not just a clean compile. MAX_TRUST_TIER and
// isNewHighWaterMark() below are the exception: pure, header-only, no
// eepromfs.h dependency -- host-tested in tests/test_save_data.cpp per
// ADR 0001 (docs/adr/0001-host-test-portable-cpp-separate-from-
// libdragon.md), the same "push decision logic into a portable module"
// discipline every other game/src/user/ header follows.
namespace SaveData
{
  struct SaveFile
  {
    uint8_t shadewrathHighestTier; // 0..2, highest trust tier ever reached
    uint8_t korrathHighestTier;    // 0..2
    uint8_t _pad[6];               // pad to the 8-byte EEPROM block size
  };
  static_assert(sizeof(SaveFile) == 8, "must fit one EEPROM block exactly");

  extern SaveFile current;

  // Initializes the EEPROM filesystem and loads `current`. Falls back to
  // all-zero defaults (nothing persisted yet) if: no EEPROM is present
  // (e.g. an emulator/flashcart run without the save type advertised),
  // the EEPROM signature doesn't match (first boot, or a cart reused
  // from another game -- wiped and reinitialized in that case), or the
  // stored file fails its checksum. Never crashes on missing/bad save
  // data; the demo runs identically to a fresh save either way, it just
  // won't remember anything yet.
  void init();

  // Records a newly-reached trust tier for Shadewrath/Korrath and
  // persists immediately if it's a new high-water mark (never lowers
  // the stored value -- "remembers the best you've done", not the most
  // recent visit). No-op, no write, if tier isn't higher than what's
  // already stored.
  void recordShadewrathTier(uint8_t tier);
  void recordKorrathTier(uint8_t tier);

  // The TR: field's ceiling -- NPCDatabase's trust tiers are always
  // 0..2 (game/src/user/NPCDatabase.h). Named so "fully won over" isn't
  // a bare magic number at each call site that cares (recordXTier()'s
  // own comparison below, and DialogueDemo.cpp's gossip trigger).
  constexpr uint8_t MAX_TRUST_TIER = 2;

  // Pure: true if newTier is strictly higher than storedHighest -- the
  // exact rule recordShadewrathTier()/recordKorrathTier() use to decide
  // whether to persist, extracted so a caller can ask "would this be a
  // NEW high-water mark" (M11's gossip trigger: fire only on the FIRST
  // time trust reaches MAX_TRUST_TIER, not every re-visit) without
  // duplicating the comparison or needing the real EEPROM peripheral.
  constexpr bool isNewHighWaterMark(uint8_t newTier, uint8_t storedHighest)
  {
    return newTier > storedHighest;
  }
}
