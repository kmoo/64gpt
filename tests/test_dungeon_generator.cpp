/* M10: seeded dungeon-level NPC placement + save/load determinism
 * contract. Same red/green discipline as test_context_builder.cpp/
 * test_npc_service.cpp. */
#include "DungeonGenerator.h"
#include "NPCDatabase.h"
#include "NpcService.h"
#include "test_util.h"
#include <string.h>

static void test_deterministic_same_seed_same_placements()
{
  /* This IS the save/load determinism contract: a "reload" is just
   * calling npcsForLevel() again with the stored level seed. If this
   * holds, reloading a level reproduces the same thin-tier instance
   * dialogue for free -- spawnInstance() is already a proven pure
   * function of (archetype, seed). */
  DungeonGenerator::NpcPlacement a[DungeonGenerator::NPCS_PER_LEVEL];
  DungeonGenerator::NpcPlacement b[DungeonGenerator::NPCS_PER_LEVEL];
  DungeonGenerator::npcsForLevel(0x5eed, a);
  DungeonGenerator::npcsForLevel(0x5eed, b);
  for(int i = 0; i < DungeonGenerator::NPCS_PER_LEVEL; ++i)
  {
    CHECK(a[i].archetype == b[i].archetype);
    CHECK_EQ_INT((int)a[i].instanceSeed, (int)b[i].instanceSeed);
  }
}

static void test_different_seeds_differ()
{
  DungeonGenerator::NpcPlacement a[DungeonGenerator::NPCS_PER_LEVEL];
  DungeonGenerator::NpcPlacement b[DungeonGenerator::NPCS_PER_LEVEL];
  DungeonGenerator::npcsForLevel(0x1234, a);
  DungeonGenerator::npcsForLevel(0x5678, b);
  bool anyDiffer = false;
  for(int i = 0; i < DungeonGenerator::NPCS_PER_LEVEL; ++i)
    if(a[i].archetype != b[i].archetype || a[i].instanceSeed != b[i].instanceSeed)
      anyDiffer = true;
  CHECK(anyDiffer);
}

static void test_seed_zero_remaps_like_spawn_instance()
{
  /* Same xorshift32 fixed-point rule spawnInstance() already follows --
   * 0 must not silently produce all-zero/degenerate output. */
  DungeonGenerator::NpcPlacement placements[DungeonGenerator::NPCS_PER_LEVEL];
  DungeonGenerator::npcsForLevel(0, placements);
  bool anyNonZeroSeed = false;
  for(int i = 0; i < DungeonGenerator::NPCS_PER_LEVEL; ++i)
  {
    CHECK(placements[i].archetype != nullptr);
    if(placements[i].instanceSeed != 0)anyNonZeroSeed = true;
  }
  CHECK(anyNonZeroSeed);
}

static void test_placements_spawn_real_instances_via_existing_mechanism()
{
  /* The whole point: a placement is exactly what spawnInstance() already
   * takes, so a level's NPCs are ordinary archetype instances, no new
   * spawn mechanism needed. */
  DungeonGenerator::NpcPlacement placements[DungeonGenerator::NPCS_PER_LEVEL];
  DungeonGenerator::npcsForLevel(0xC0FFEE, placements);
  for(int i = 0; i < DungeonGenerator::NPCS_PER_LEVEL; ++i)
  {
    NPCDatabase::NPC npc = NPCDatabase::spawnInstance(
        *placements[i].archetype, placements[i].instanceSeed);
    CHECK(npc.id[0] != '\0');
    CHECK(npc.tier == NPCDatabase::Tier::THIN);
  }
}

static void test_placements_drawn_from_thin_archetype_pool()
{
  /* Every placement's archetype must be one of the real, declared thin
   * archetypes -- catches a stale pointer or an uninitialized entry. */
  const NPCDatabase::Archetype *const pool[] = {
    &NPCDatabase::GUARD_ARCHETYPE, &NPCDatabase::PUB_PATRON_ARCHETYPE,
    &NPCDatabase::BLACKSMITH_ARCHETYPE, &NPCDatabase::WIZARD_ARCHETYPE,
    &NPCDatabase::VILLAGER_ARCHETYPE, &NPCDatabase::MERCHANT_ARCHETYPE,
    &NPCDatabase::HEALER_ARCHETYPE,
  };
  DungeonGenerator::NpcPlacement placements[DungeonGenerator::NPCS_PER_LEVEL];
  DungeonGenerator::npcsForLevel(42, placements);
  for(int i = 0; i < DungeonGenerator::NPCS_PER_LEVEL; ++i)
  {
    bool found = false;
    for(const auto *arch : pool)
      if(placements[i].archetype == arch)found = true;
    CHECK(found);
  }
}

static void test_reload_reproduces_identical_dialogue_text()
{
  /* The actual DoD claim (m10.md): "reloading a level reproduces the
   * same thin-tier instance dialogue", not just the same seed. Go all
   * the way to the real conditioning STRING an NpcService-routed
   * archetype would feed the model -- "first play" and "reload" are
   * just two independent derivations from the same stored level seed,
   * with nothing else carried over between them. */
  auto conditioningStringsForLevel = [](uint32_t levelSeed, char out[][96]) {
    DungeonGenerator::NpcPlacement placements[DungeonGenerator::NPCS_PER_LEVEL];
    DungeonGenerator::npcsForLevel(levelSeed, placements);
    for(int i = 0; i < DungeonGenerator::NPCS_PER_LEVEL; ++i)
    {
      NPCDatabase::NPC npc = NPCDatabase::spawnInstance(
          *placements[i].archetype, placements[i].instanceSeed);
      NpcService::Profile profile = NpcService::profileFor(npc);
      NpcService::RelationshipState rel{500, 500, 500, 500, 0};
      NpcService::buildPromptFields(out[i], 96, profile, rel,
                                     "cheerful", "greeting", "");
    }
  };

  char firstPlay[DungeonGenerator::NPCS_PER_LEVEL][96];
  char reload[DungeonGenerator::NPCS_PER_LEVEL][96];
  conditioningStringsForLevel(0xFEED, firstPlay);
  conditioningStringsForLevel(0xFEED, reload);
  for(int i = 0; i < DungeonGenerator::NPCS_PER_LEVEL; ++i)
    CHECK(strcmp(firstPlay[i], reload[i]) == 0);
}

int main()
{
  test_deterministic_same_seed_same_placements();
  test_different_seeds_differ();
  test_seed_zero_remaps_like_spawn_instance();
  test_placements_spawn_real_instances_via_existing_mechanism();
  test_placements_drawn_from_thin_archetype_pool();
  test_reload_reproduces_identical_dialogue_text();
  return test_summary("test_dungeon_generator");
}
