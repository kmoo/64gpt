#include "SaveData.h"
#include <eepromfs.h>

namespace SaveData
{
  SaveFile current{};

  static const char *const SAVE_PATH = "save.dat";

  void init()
  {
    static const eepfs_entry_t entries[] = {
      { SAVE_PATH, sizeof(SaveFile), /*checksum=*/true, /*backup=*/false },
    };

    current = SaveFile{}; // default: nothing persisted yet

    if(eepfs_init(entries, 1) != EEPFS_ESUCCESS)
      return; // no EEPROM present (or too small) -- run with defaults,
              // never persists this session, doesn't crash either

    if(!eepfs_verify_signature())
    {
      // Fresh EEPROM, or one reused from another game/build -- eepfs.h's
      // own documented guidance: wipe and start clean rather than trust
      // data that doesn't match this filesystem's layout.
      eepfs_wipe();
      eepfs_write(SAVE_PATH, &current, sizeof(current));
      return;
    }

    if(eepfs_read(SAVE_PATH, &current, sizeof(current)) != EEPFS_ESUCCESS)
      current = SaveFile{}; // checksum failed even with a valid signature
                             // (corruption) -- fall back to defaults
                             // rather than trust partial/garbled data
  }

  static void save()
  {
    eepfs_write(SAVE_PATH, &current, sizeof(current));
    // Fire-and-forget: writes are async (eepfs/eeprom.h's documented
    // "eventually consistent" model) and a single 8-byte block is fast
    // even by EEPROM standards (contrast eepfs_wipe()'s ~1s for a full
    // 4k EEPROM) -- no "saving..." UI needed for one block.
  }

  void recordShadewrathTier(uint8_t tier)
  {
    if(isNewHighWaterMark(tier, current.shadewrathHighestTier))
    {
      current.shadewrathHighestTier = tier;
      save();
    }
  }

  void recordKorrathTier(uint8_t tier)
  {
    if(isNewHighWaterMark(tier, current.korrathHighestTier))
    {
      current.korrathHighestTier = tier;
      save();
    }
  }
}
