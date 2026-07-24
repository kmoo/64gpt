/**
 * Quantized GRU inference — the C twin of trainer/ngpt_trainer/ref_impl.py.
 *
 * Every operation mirrors the reference implementation exactly: int64
 * accumulators, round-half-up arithmetic shifts, 256-entry Q14 LUTs,
 * int16 saturation. No floats, no heap, no libdragon. Weights are read
 * from the blob big-endian byte-by-byte on the fly (int8 needs no byte
 * order; i16/i32 go through the ngpt_read_*be readers), so the same bits
 * flow on the little-endian host and the big-endian N64.
 *
 * C++20 guarantees >> on negative signed integers is arithmetic — the
 * rounding shifts below depend on that (and NumPy matches it).
 */
#include "ngpt_gru.h"
#include "ngpt_sample.h"

#define Q14_ONE 16384

/* M6.1: optional accelerated W_hh matvec (RSP on the N64). NULL keeps
 * the pure-CPU loop below — the path every host test runs. */
static ngpt_matvec_fn ngpt_matvec_hook = 0;

void ngpt_set_matvec(ngpt_matvec_fn fn) { ngpt_matvec_hook = fn; }

static int64_t rshift_round(int64_t x, int s)
{
  return (x + ((int64_t)1 << (s - 1))) >> s;
}

/* M5: the hot loops run int32 — a 64-bit multiply is a different, slower
 * instruction on the R4300i, and ref_impl documents that int32 suffices
 * at these dims (H<=256, V<=96: a row sum stays < 2^30, sum + bias
 * inside int32). Bit-identical to the int64 path for every in-range
 * value; the goldens are the proof. */
static int32_t rshift_round32(int32_t x, int s)
{
  return (x + ((int32_t)1 << (s - 1))) >> s;
}

static int16_t sat16(int64_t x)
{
  if (x > 32767) return 32767;
  if (x < -32768) return -32768;
  return (int16_t)x;
}

static int16_t read_i16be(const uint8_t *p) { return (int16_t)ngpt_read_u16be(p); }
static int32_t read_i32be(const uint8_t *p) { return (int32_t)ngpt_read_u32be(p); }

/* 256-entry LUT over [-8, 8) in Q11: index = (clamped + 16384) >> 7 */
static int32_t lut_lookup(const uint8_t *lut, int32_t x_q11)
{
  if (x_q11 < -16384) x_q11 = -16384;
  if (x_q11 > 16383) x_q11 = 16383;
  uint32_t idx = (uint32_t)((x_q11 + 16384) >> 7);
  return read_i16be(lut + 2 * idx);
}

int ngpt_gru_load(ngpt_model *m)
{
  const uint8_t *p = m->payload;
  if (m->payload_len < 6) return NGPT_ERR_TRUNCATED;

  ngpt_gru_view *g = &m->gru;
  g->H = ngpt_read_u16be(p);
  g->V = ngpt_read_u16be(p + 2);
  g->k_w = p[4];
  g->k_out = p[5];
  if (g->H == 0 || g->H > NGPT_GRU_MAX_HIDDEN) return NGPT_ERR_DIMS;
  if (g->V == 0 || g->V > NGPT_GRU_MAX_VOCAB) return NGPT_ERR_DIMS;

  const uint32_t H = g->H, V = g->V;
  const uint32_t need = 6 + V + 1024              /* dims, charset, LUTs */
                      + 3 * H * V + 3 * H * H     /* W_ih, W_hh          */
                      + 12 * H + 12 * H           /* b_ih, b_hh (i32)    */
                      + V * H + 4 * V;            /* W_out, b_out        */
  if (m->payload_len != need) return NGPT_ERR_TRUNCATED;

  const uint8_t *q = p + 6;
  g->charset = q;            q += V;
  g->lut_sig = q;            q += 512;
  g->lut_tanh = q;           q += 512;
  g->w_ih = q;               q += 3 * H * V;
  g->w_hh = q;               q += 3 * H * H;
  g->b_ih = q;               q += 12 * H;
  g->b_hh = q;               q += 12 * H;
  g->w_out = q;              q += V * H;
  g->b_out = q;
  if (g->charset[0] != 0) return NGPT_ERR_TRUNCATED; /* EOS slot must be 0 */
  return NGPT_OK;
}

/* The gate math only: consume token x_id, update ctx->h in place.
 * Priming uses this without the logits pass. */
static void gru_h_update(ngpt_ctx *ctx, uint32_t x_id)
{
  const ngpt_gru_view *g = &ctx->model->gru;
  const uint32_t H = g->H, V = g->V;
  const int s = g->k_w + 3; /* accumulator 2^(k_w+14) -> Q11 LUT input */

  /* One-hot input in Q14 is a single 16384, so the input-side "matvec"
   * is column ctx->cur of W_ih, shifted left 14 — plus the bias that the
   * trainer already scaled into the accumulator domain.
   * Scratch buffers are static, NOT stack: ~6.5 KB of locals overflows
   * the small stack Pyrite64 runs object-script callbacks on (symptom:
   * memory corruption -> engine bad_function_call at the next frame
   * swap). The engine is single-threaded, so statics are safe. */
  static int32_t acc_i[3 * NGPT_GRU_MAX_HIDDEN];
  static int32_t acc_h[3 * NGPT_GRU_MAX_HIDDEN];
  for (uint32_t i = 0; i < 3 * H; ++i) {
    int32_t w = (int8_t)g->w_ih[i * V + x_id];
    acc_i[i] = (w << 14) + read_i32be(g->b_ih + 4 * i);
  }

  /* W_hh matvec (~80% of a step's MACs): the one place the M6.1 hook
   * replaces. The callback writes raw row.h sums; biases are added here
   * either way, so both paths compute bit-identical acc_h. */
  if (ngpt_matvec_hook) {
    ngpt_matvec_hook(g->w_hh, 3 * H, H, ctx->h, acc_h);
  } else {
    /* M12 (H up to 1024): a row sum's REALISTIC bound (h's Q14-quantized
     * |h|<=16384, not the worst-case 32767) sits at ~99.2% of int32's hard
     * ceiling at H=1024 -- real margin at H<=320, essentially none left
     * here. Accumulating in int64 costs nothing this loop doesn't already
     * pay (this is the CPU reference path only, gated off whenever the RSP
     * hook is installed -- see the boot self-test's BOOT_XCHK_CPU phase,
     * DialogueDemo.cpp) and removes the signed-overflow UB risk outright.
     * Saturating (not wrapping) on narrow-back is deliberate: the RSP
     * kernel's own int32 accumulator (rsp_ngpt.S, unwidened -- fixing
     * THAT costs real cycles on hardware with no native 64-bit path) still
     * carries this project's original, disclosed risk. If a real trained
     * model's weights ever push it into a genuine overflow, wrapping vs.
     * saturating diverge, which is exactly what turns a silent numeric
     * corruption into a loud, on-screen XCHK FAIL -- the same
     * CPU-vs-RSP cross-check this project already trusts as its
     * correctness gate, just made trustworthy for this specific new risk
     * instead of both paths quietly overflowing the same way and still
     * agreeing. */
    for (uint32_t i = 0; i < 3 * H; ++i) {
      int64_t sum = 0;
      const uint8_t *row = g->w_hh + i * H;
      for (uint32_t j = 0; j < H; ++j) sum += (int64_t)(int32_t)((int8_t)row[j]) * ctx->h[j];
      if (sum > INT32_MAX) sum = INT32_MAX;
      else if (sum < INT32_MIN) sum = INT32_MIN;
      acc_h[i] = (int32_t)sum;
    }
  }
  for (uint32_t i = 0; i < 3 * H; ++i) {
    int64_t withBias = (int64_t)acc_h[i] + read_i32be(g->b_hh + 4 * i);
    if (withBias > INT32_MAX) withBias = INT32_MAX;
    else if (withBias < INT32_MIN) withBias = INT32_MIN;
    acc_h[i] = (int32_t)withBias;
  }

  static int16_t h_next[NGPT_GRU_MAX_HIDDEN];
  for (uint32_t j = 0; j < H; ++j) {
    int32_t r = lut_lookup(g->lut_sig, rshift_round32(acc_i[j] + acc_h[j], s));
    int32_t z = lut_lookup(g->lut_sig, rshift_round32(acc_i[H + j] + acc_h[H + j], s));
    /* n-gate: r (Q14) gates the hidden-side accumulator only. The one
     * product that genuinely needs 64 bits: r (2^14) x acc (2^30). */
    int32_t n_acc = acc_i[2 * H + j] +
                    (int32_t)rshift_round((int64_t)r * acc_h[2 * H + j], 14);
    int32_t n = lut_lookup(g->lut_tanh, rshift_round32(n_acc, s));
    h_next[j] = sat16(rshift_round32((Q14_ONE - z) * n, 14) +
                      rshift_round32(z * (int32_t)ctx->h[j], 14));
  }
  for (uint32_t j = 0; j < H; ++j) ctx->h[j] = h_next[j];
}

void ngpt_gru_prime(ngpt_ctx *ctx, const char *prompt)
{
  const ngpt_gru_view *g = &ctx->model->gru;
  /* cur starts as EOS (set by ngpt_reset); for each prompt char: consume
   * cur, then make the char the next input. After the loop cur is the
   * LAST prompt char, so the first ngpt_step consumes it and its argmax
   * is the first generated character. Mirrors ref_impl.prime(). */
  for (const char *p = prompt; p && *p; ++p) {
    uint32_t id = 0;
    for (uint32_t v = 1; v < g->V; ++v) {
      if (g->charset[v] == (uint8_t)*p) { id = v; break; }
    }
    if (id == 0) continue; /* unknown char: skip (docs/milestones/m3.md) */
    gru_h_update(ctx, ctx->cur);
    ctx->cur = (uint16_t)id;
  }
}

int ngpt_gru_step(ngpt_ctx *ctx)
{
  const ngpt_gru_view *g = &ctx->model->gru;
  const uint32_t H = g->H, V = g->V;

  gru_h_update(ctx, ctx->cur);

  /* logits (static scratch — small N64 thread stacks, the M2 lesson) */
  static int32_t logits[NGPT_GRU_MAX_VOCAB];
  for (uint32_t v = 0; v < V; ++v) {
    int32_t sum = 0;
    const uint8_t *row = g->w_out + v * H;
    for (uint32_t j = 0; j < H; ++j) sum += (int32_t)(int8_t)row[j] * ctx->h[j];
    logits[v] = sum + read_i32be(g->b_out + 4 * v);
  }

  /* pick the next token: M4 sampler when enabled, else argmax with ties
   * toward the lowest id (matches np.argmax) */
  uint32_t best = 0;
  if (ctx->sample_on) {
    best = ngpt_sample_pick(ctx, logits, V);
  } else {
    for (uint32_t v = 1; v < V; ++v) {
      if (logits[v] > logits[best]) best = v;
    }
  }

  ctx->cur = (uint16_t)best;
  if (best == 0) { /* EOS */
    ctx->finished = 1;
    return NGPT_EOS;
  }
  return (int)g->charset[best];
}
