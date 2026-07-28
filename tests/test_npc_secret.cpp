/* NPC secrets (docs/ideas-m7-living-npcs.md "The feature I would add:
 * NPC secrets" -- the SECRET slot specifically). */
#include "NpcSecret.h"
#include "test_util.h"

using namespace NpcSecret;

static void test_reveal_secret_sets_revealed()
{
  Secret secret{101, 5, false};
  revealSecret(secret);
  CHECK(secret.revealed);
}

static void test_reveal_secret_is_idempotent()
{
  Secret secret{101, 5, false};
  revealSecret(secret);
  revealSecret(secret);
  revealSecret(secret);
  CHECK(secret.revealed); /* no crash, no toggling back */
}

static void test_is_discoverable_requires_condition_met()
{
  Secret secret{101, 5, false};
  CHECK(!isDiscoverable(secret, false));
  CHECK(isDiscoverable(secret, true));
}

static void test_is_discoverable_false_once_already_revealed()
{
  Secret secret{101, 5, false};
  revealSecret(secret);
  CHECK(!isDiscoverable(secret, true)); /* already known, nothing to discover */
}

int main()
{
  test_reveal_secret_sets_revealed();
  test_reveal_secret_is_idempotent();
  test_is_discoverable_requires_condition_met();
  test_is_discoverable_false_once_already_revealed();
  return test_summary("test_npc_secret");
}
