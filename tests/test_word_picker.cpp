/* Word-picker selection validator (docs/ideas.md #7), bounds-safety
 * slice only -- see WordPicker.h's own header comment for scope. */
#include "WordPicker.h"
#include "test_util.h"
#include <string.h>

using namespace WordPicker;

static const char *const TOKENS[3] = {"HAPPY", "WORRIED", "TENDER"};

static void test_resolve_valid_selection()
{
  const char *out = nullptr;
  bool ok = resolveSelection(TOKENS, 3, 1, &out);
  CHECK(ok);
  CHECK(out != nullptr);
  CHECK_EQ_INT(strcmp(out, "WORRIED"), 0);
}

static void test_resolve_first_and_last_index()
{
  const char *out = nullptr;
  CHECK(resolveSelection(TOKENS, 3, 0, &out));
  CHECK_EQ_INT(strcmp(out, "HAPPY"), 0);
  CHECK(resolveSelection(TOKENS, 3, 2, &out));
  CHECK_EQ_INT(strcmp(out, "TENDER"), 0);
}

static void test_resolve_rejects_negative_index()
{
  const char *out = (const char *)0x1; /* sentinel: must stay untouched */
  bool ok = resolveSelection(TOKENS, 3, -1, &out);
  CHECK(!ok);
  CHECK(out == (const char *)0x1);
}

static void test_resolve_rejects_index_at_and_past_count()
{
  const char *out = (const char *)0x1;
  CHECK(!resolveSelection(TOKENS, 3, 3, &out)); /* one past the end */
  CHECK(out == (const char *)0x1);
  CHECK(!resolveSelection(TOKENS, 3, 100, &out)); /* garbled input */
  CHECK(out == (const char *)0x1);
}

static void test_resolve_zero_count_always_rejects()
{
  const char *out = (const char *)0x1;
  CHECK(!resolveSelection(TOKENS, 0, 0, &out));
  CHECK(out == (const char *)0x1);
}

int main()
{
  test_resolve_valid_selection();
  test_resolve_first_and_last_index();
  test_resolve_rejects_negative_index();
  test_resolve_rejects_index_at_and_past_count();
  test_resolve_zero_count_always_rejects();
  return test_summary("test_word_picker");
}
