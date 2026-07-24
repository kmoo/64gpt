/**
 * Temperature/top-k sampling, integer-only — mirrors
 * ref_impl.sample_from_logits step for step (same shifts, same rounding,
 * same tie-breaks), so seeded generations are bit-identical between the
 * trainer, the host tests, and the N64.
 *
 * NGPT_LUT_EXP2 comes from the generated header (trainer/make_m4_blob.py
 * emits the identical table into ngpt_trainer/sampler_lut.py — one
 * source, no drift).
 */
#include "ngpt_sample.h"
#include "ngpt_sampler_lut.h"

static int64_t rshift_round(int64_t x, int s)
{
  return (x + ((int64_t)1 << (s - 1))) >> s;
}

static uint32_t xorshift32(uint32_t x)
{
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

/* exp2 LUT over [-16, 0) in Q10, 64-unit buckets; the domain is
 * negative (inputs are logit diffs from the max), so 0 clamps DOWN into
 * the top bucket — mirrors sampler_lut.lut_exp2_lookup. */
static uint32_t lut_exp2(int64_t x_q10)
{
  if (x_q10 < -16384) x_q10 = -16384;
  if (x_q10 > -1) x_q10 = -1;
  return NGPT_LUT_EXP2[(uint32_t)((x_q10 + 16384) >> 6)];
}

uint32_t ngpt_sample_pick(ngpt_ctx *ctx, const int32_t *logits, uint32_t V)
{
  /* scratch is static, not stack: N64 game threads run on small stacks
   * (the M2 lesson) */
  static uint32_t order[NGPT_GRU_MAX_VOCAB];
  static uint32_t weights[NGPT_GRU_MAX_VOCAB];
  static uint8_t used[NGPT_GRU_MAX_VOCAB];

  uint32_t k = ctx->top_k < V ? ctx->top_k : V;

  /* top-k indices, ties toward the lowest id (k selection passes) */
  for (uint32_t v = 0; v < V; ++v) used[v] = 0;
  for (uint32_t n = 0; n < k; ++n) {
    uint32_t best = 0;
    int32_t best_v = 0;
    int have = 0;
    for (uint32_t v = 0; v < V; ++v) {
      if (used[v]) continue;
      if (!have || logits[v] > best_v) { best = v; best_v = logits[v]; have = 1; }
    }
    used[best] = 1;
    order[n] = best;
  }

  /* temperature, then exp2 weights on the Q10 diff from the max — the
   * inv_t product exceeds 32 bits (2^30 x 2^16), so this cold path
   * (k multiplies per emitted char) stays int64 */
  const int s = ctx->model->gru.k_out + 4; /* 2^(k_out+14) -> Q10 */
  const int64_t top = rshift_round((int64_t)logits[order[0]] * ctx->inv_t_q8, 8);
  uint32_t total = 0;
  for (uint32_t n = 0; n < k; ++n) {
    int64_t scaled = rshift_round((int64_t)logits[order[n]] * ctx->inv_t_q8, 8);
    weights[n] = lut_exp2(rshift_round(scaled - top, s));
    total += weights[n];
  }

  /* the RNG advances exactly once per step, even on the greedy path */
  ctx->rng = xorshift32(ctx->rng);
  if (k == 1 || total == 0) return order[0];

  /* M12.1 min-p gate (docs/ideas-coherence-rescue-plan.md fix 3): drop
   * any candidate whose weight falls below weights[0] >> minp_shift.
   * weights[0] is ALWAYS the max — order[0] is the top logit (temperature
   * scaling and the exp2 LUT are both monotonic in the top-k's, all <= 0,
   * diffs from it), so no later candidate's weight can exceed it. This
   * mirrors ref_impl.sample_from_logits's minp_shift branch exactly.
   * minp_shift == 0 (ngpt_reset's default, unless ngpt_set_minp is
   * called) skips this and falls through to the unfiltered draw below —
   * strictly additive, byte-identical to every milestone before M12.1. */
  if (ctx->minp_shift > 0) {
    uint32_t floor = weights[0] >> ctx->minp_shift;
    uint32_t kept_total = 0;
    for (uint32_t n = 0; n < k; ++n) if (weights[n] >= floor) kept_total += weights[n];
    uint32_t draw = ctx->rng % kept_total;
    uint32_t cum = 0;
    for (uint32_t n = 0; n < k; ++n) {
      if (weights[n] < floor) continue;
      cum += weights[n];
      if (cum > draw) return order[n];
    }
    return order[0]; /* unreachable; belt and suspenders */
  }

  uint32_t draw = ctx->rng % total;
  uint32_t cum = 0;
  for (uint32_t n = 0; n < k; ++n) {
    cum += weights[n];
    if (cum > draw) return order[n];
  }
  return order[k - 1]; /* unreachable; belt and suspenders */
}
