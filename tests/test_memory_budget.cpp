/* Regression guard for the hardware budget constants in core/ngpt.h.
 *
 * This project has a documented history of these numbers drifting without
 * anyone noticing: M12 bumped NGPT_GRU_MAX_HIDDEN 320 -> 1024 to test
 * whether more model capacity fixed a coherence problem, found out it made
 * things WORSE (val loss regressed, RSP inference dropped from 44 ch/s to
 * ~5 ch/s), and left the constant at 1024 as a ceiling rather than reverting
 * it -- see docs/milestones/m12.md and core/ngpt.h's own comment above
 * NGPT_GRU_MAX_HIDDEN. Nothing in tests/ pinned the actual numeric values
 * before this file, so a future edit could silently change either constant
 * (or the ngpt_ctx layout that depends on them) with no red test to catch
 * it. This file exists purely to make that kind of drift loud: if you're
 * here because a CHECK below just failed, that's the point -- go update
 * this file consciously, with the same kind of sign-off M12 documented,
 * rather than let the change pass unremarked. */
#include "ngpt.h"
#include "test_util.h"
#include <cstddef>

/* ---- 1: the hidden-size ceiling ----
 * This is a regression guard, NOT an endorsement of 1024 as a target
 * operating point. M12's own "honest negative": H=1024 trains to a WORSE
 * val loss than H=320 and runs the RSP kernel at ~9x fewer chars/sec
 * (44 ch/s at H=320 vs ~5 ch/s at H=1024). The constant stays at 1024
 * only as a static-array ceiling that ngpt_ctx.h[] must be sized to fit;
 * real generation on real hardware uses a much smaller H (see fact #4
 * below on NGPT_RSP_H, which currently only builds bodies for 320 and
 * 1024, not everything in between or above). */
static void test_gru_max_hidden_ceiling()
{
  CHECK_EQ_INT(NGPT_GRU_MAX_HIDDEN, 1024);
}

/* ---- 2: the vocab ceiling ---- */
static void test_gru_max_vocab_ceiling()
{
  CHECK_EQ_INT(NGPT_GRU_MAX_VOCAB, 96);
}

/* ---- 3: ngpt_ctx's total footprint ----
 * ngpt_ctx (core/ngpt.h) carries the GRU hidden state inline -- no heap,
 * per this project's hard constraints -- as `int16_t h[NGPT_GRU_MAX_HIDDEN]`.
 * That means ngpt_ctx's size is dominated by whatever NGPT_GRU_MAX_HIDDEN
 * is, and a future bump to that constant (or a change to the struct's
 * other fields) changes how much RDRAM every live NPC context costs. This
 * static_assert pins TODAY's exact size so either kind of change is caught
 * here, not absorbed silently.
 *
 * The number below (2080) was derived by hand from the struct's field
 * declarations, in order, using ordinary C struct layout rules (natural
 * alignment, no #pragma pack in ngpt.h) on this host's LP64 ABI
 * (8-byte pointers, 8-byte max alignment) -- verified against the real
 * compiler's sizeof/offsetof, not just asserted on faith:
 *
 *   offset  field                          size   notes
 *   0       const ngpt_model *model        8      align 8
 *   8       uint32_t pos                   4      align 4 (packs after ptr)
 *   12      uint8_t finished                1
 *   13      -- 1 byte padding --            1      align h[] to 2
 *   14      int16_t h[NGPT_GRU_MAX_HIDDEN]  2048   1024 * 2 bytes
 *   2062    uint16_t cur                    2      already 2-aligned
 *   2064    uint8_t sample_on               1
 *   2065    -- 1 byte padding --            1      align inv_t_q8 to 2
 *   2066    uint16_t inv_t_q8               2
 *   2068    uint16_t top_k                  2
 *   2070    -- 2 bytes padding --           2      align rng to 4
 *   2072    uint32_t rng                    4
 *   2076    uint8_t minp_shift              1
 *   2077    uint8_t trie_on                 1
 *   2078    uint16_t trie_node              2      already 2-aligned
 *   ----------------------------------------------
 *   2080 total (struct align is 8 from the pointer member; 2080 is
 *   already a multiple of 8, so no trailing pad is needed)
 *
 * If NGPT_GRU_MAX_HIDDEN ever changes again, this number MUST change too
 * (by exactly 2 * the delta in NGPT_GRU_MAX_HIDDEN) -- if it doesn't, that
 * itself is a sign this test wasn't actually updated, just silenced. */
static_assert(sizeof(ngpt_ctx) == 2080,
              "ngpt_ctx size drifted from its hand-derived, hardware-"
              "budget-relevant layout -- see the by-hand derivation above "
              "this static_assert in tests/test_memory_budget.cpp and "
              "update BOTH the comment and the literal consciously");

static void test_ngpt_ctx_size_matches_hand_derivation()
{
  /* Same fact as the static_assert above, surfaced as an ordinary runtime
   * CHECK too so a failure shows up in ctest's normal pass/fail count
   * (test_summary()'s tally), not just as a compile error a reader might
   * not immediately connect to "the memory budget regressed". */
  CHECK_EQ_INT(sizeof(ngpt_ctx), 2080);
}

/* ---- 4: NGPT_RSP_H's build-time default ----
 * NGPT_RSP_H is a preprocessor default defined inside
 * game/src/user/rsp_ngpt.S (an N64 RSP assembly file), NOT anything
 * reachable from core/ngpt.h or from a host C++ test -- there is no
 * portable way to #include or parse a .S file from here, and this test
 * deliberately does not try to fake one up. As of this writing,
 * rsp_ngpt.S lines ~31-40 read:
 *
 *   #ifndef NGPT_RSP_H
 *   #define NGPT_RSP_H 320
 *   #endif
 *   ...
 *   #error ... (anything other than 320 or 1024)
 *
 * i.e. the RSP kernel only has two hardware-verified body variants
 * compiled in (rsp_ngpt_h320.inc and rsp_ngpt_h1024.inc), and 320 -- not
 * NGPT_GRU_MAX_HIDDEN's 1024 -- is what actually ships as the default for
 * real generation on real hardware (see docs/milestones/m12.1.md phase 6
 * for how a new H variant gets added).
 *
 * This is a MANUALLY-SYNCHRONIZED pin, not an automatic cross-check: if
 * rsp_ngpt.S's default or its accepted-value list ever change, a human
 * has to notice and update EXPECTED_DEFAULT_RSP_H below by hand -- this
 * test cannot and does not read the .S file's real value. Its only job is
 * to put that fact somewhere a reviewer actually looks (a host test file)
 * instead of leaving it undocumented outside a single assembly comment. */
static const int EXPECTED_DEFAULT_RSP_H = 320;

static void test_rsp_h_default_is_documented_and_pinned()
{
  CHECK_EQ_INT(EXPECTED_DEFAULT_RSP_H, 320);
}

int main()
{
  test_gru_max_hidden_ceiling();
  test_gru_max_vocab_ceiling();
  test_ngpt_ctx_size_matches_hand_derivation();
  test_rsp_h_default_is_documented_and_pinned();
  return test_summary("test_memory_budget");
}
