/* M12.1 phase 3: bit-exactness test for the integer min-p gate
 * (ngpt_set_minp / ngpt_sample_pick's minp_shift branch, core/
 * ngpt_sample.cpp). Unlike test_sampled_model.cpp / test_matvec_hook.cpp,
 * this needs no trained blob at all: ngpt_sample_pick(ctx, logits, V)
 * only ever reads ctx->model->gru.k_out from the model, so a
 * hand-built ngpt_model with just k_out set is a complete, valid
 * fixture -- the same style test_overflow_mitigation.cpp uses a matvec
 * hook to drive acc_h to fixed values without a real trained model.
 *
 * Expected outputs (token ids AND the exact post-call ctx->rng state)
 * are cross-checked against ngpt_trainer.ref_impl.sample_from_logits
 * with the same minp_shift, logits, seeds, top_k, inv_t_q8, k_out=6 --
 * the Python reference is the ground truth this file mirrors, per the
 * project's bit-exactness contract (docs/03-blob-format.md's spirit,
 * applied to the sampler instead of the blob). Reference values were
 * generated once via `uv run python -c "..."` against ref_impl.py and
 * pinned here as literals, the same pattern test_sampler.py's own
 * pinned xorshift32 sequence uses. */
#include "ngpt.h"
#include "ngpt_sample.h"
#include "test_util.h"
#include <string.h>

static void makeCtx(ngpt_model *m, ngpt_ctx *ctx, uint16_t top_k,
                    uint16_t inv_t_q8, uint8_t minp_shift, uint32_t rng_seed)
{
  memset(m, 0, sizeof(*m));
  m->model_type = NGPT_MODEL_GRU;
  m->gru.k_out = 6; /* matches test_sampler.py's Q = SimpleNamespace(k_out=6) */

  memset(ctx, 0, sizeof(*ctx));
  ctx->model = m;
  ctx->sample_on = 1;
  ctx->top_k = top_k;
  ctx->inv_t_q8 = inv_t_q8;
  ctx->minp_shift = minp_shift;
  ctx->rng = rng_seed; /* pre-remapped: every seed used below is nonzero,
                        * so this matches ref_impl's `state = seed` exactly
                        * (the 0->1 remap lives in ngpt_set_sampler, not
                        * ngpt_sample_pick, and isn't exercised here) */
}

int main(void)
{
  /* ---- Fixture A: a clear leader (docs/milestones/m12.1.md's own
   * "collapses to near-greedy" case, mirrored from test_sampler.py's
   * test_minp_shift1_collapses_a_clear_leader_to_near_greedy). At
   * minp_shift=1, every seed must return token 0 (the argmax) AND the
   * post-call rng must match ref_impl bit-for-bit. */
  {
    const int32_t logits[] = {1000 << 14, 400 << 14, 390 << 14, 380 << 14, 10 << 14};
    const uint32_t V = 5;
    struct { uint32_t seed, want_rng_after; } cases[] = {
      {1,     270369u},
      {7,     1892583u},
      {12345, 3336926330u},
      {99,    25193669u},
      {777,   205866009u},
      {42,    11355432u},
    };
    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      ngpt_model m; ngpt_ctx ctx;
      makeCtx(&m, &ctx, /*top_k=*/5, /*inv_t_q8=*/256, /*minp_shift=*/1, cases[i].seed);
      uint32_t tok = ngpt_sample_pick(&ctx, logits, V);
      CHECK_EQ_INT((int)tok, 0);
      CHECK_EQ_INT(ctx.rng, cases[i].want_rng_after);
    }
  }

  /* ---- Fixture B: close top-k candidates, minp_shift=2 -- variety
   * must survive (NOT collapse to a single token the way Fixture A
   * does), and the exact per-seed token sequence must match ref_impl
   * (seeds 1..20, independent calls -- each ctx freshly seeded, not
   * chained). Also proves id 4 (the clear outlier) is reachable at
   * this looser shift (floor=25%) even though it's excluded at
   * shift=1 -- the monotonicity test_minp_higher_shift_keeps_more_
   * candidates in test_sampler.py covers as its own property. */
  {
    const int32_t logits[] = {100 << 14, 90 << 14, 70 << 14, 40 << 14, 5 << 14};
    const uint32_t V = 5;
    const int want[20] = {1,3,0,2,0,1,3,1,2,0,2,4,1,3,0,2,0,1,3,0};
    int seenMask = 0;
    for (uint32_t seed = 1; seed <= 20; ++seed) {
      ngpt_model m; ngpt_ctx ctx;
      makeCtx(&m, &ctx, /*top_k=*/5, /*inv_t_q8=*/256, /*minp_shift=*/2, seed);
      uint32_t tok = ngpt_sample_pick(&ctx, logits, V);
      CHECK_EQ_INT((int)tok, want[seed - 1]);
      seenMask |= (1 << tok);
    }
    CHECK(seenMask != (1 << want[0])); /* more than one distinct token appeared */
  }

  /* ---- Fixture C: minp_shift=0 is a true no-op, exercised directly
   * through ngpt_sample_pick (not just via ngpt_reset's default) --
   * matches greedy top-1 (order[0]) whenever total weight concentrates
   * there, and more generally must reproduce the SAME draw as calling
   * with the gate on at a shift loose enough to keep every candidate
   * (shift large enough that floor <= the smallest weight): the two
   * paths must agree bit-for-bit since the "kept" set is identical. */
  {
    const int32_t logits[] = {100 << 14, 99 << 14, 98 << 14, 97 << 14, 96 << 14};
    const uint32_t V = 5;
    for (uint32_t seed = 1; seed <= 10; ++seed) {
      ngpt_model m; ngpt_ctx ctxOff, ctxLoose;
      makeCtx(&m, &ctxOff, 5, 256, /*minp_shift=*/0, seed);
      makeCtx(&m, &ctxLoose, 5, 256, /*minp_shift=*/8, seed); /* floor collapses to 0 */
      uint32_t tokOff = ngpt_sample_pick(&ctxOff, logits, V);
      uint32_t tokLoose = ngpt_sample_pick(&ctxLoose, logits, V);
      CHECK_EQ_INT((int)tokOff, (int)tokLoose);
      CHECK_EQ_INT(ctxOff.rng, ctxLoose.rng);
    }
  }

  return test_summary("test_minp_sampler");
}
