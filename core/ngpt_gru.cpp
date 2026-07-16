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

#define Q14_ONE 16384

static int64_t rshift_round(int64_t x, int s)
{
  return (x + ((int64_t)1 << (s - 1))) >> s;
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
static int64_t lut_lookup(const uint8_t *lut, int64_t x_q11)
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

int ngpt_gru_step(ngpt_ctx *ctx)
{
  const ngpt_gru_view *g = &ctx->model->gru;
  const uint32_t H = g->H, V = g->V;
  const int s = g->k_w + 3; /* accumulator 2^(k_w+14) -> Q11 LUT input */

  /* One-hot input in Q14 is a single 16384, so the input-side "matvec"
   * is column ctx->cur of W_ih, shifted left 14 — plus the bias that the
   * trainer already scaled into the accumulator domain. */
  int64_t acc_i[3 * NGPT_GRU_MAX_HIDDEN];
  int64_t acc_h[3 * NGPT_GRU_MAX_HIDDEN];
  for (uint32_t i = 0; i < 3 * H; ++i) {
    int64_t w = (int8_t)g->w_ih[i * V + ctx->cur];
    acc_i[i] = (w << 14) + read_i32be(g->b_ih + 4 * i);

    int64_t sum = 0;
    const uint8_t *row = g->w_hh + i * H;
    for (uint32_t j = 0; j < H; ++j) sum += (int64_t)(int8_t)row[j] * ctx->h[j];
    acc_h[i] = sum + read_i32be(g->b_hh + 4 * i);
  }

  int16_t h_next[NGPT_GRU_MAX_HIDDEN];
  for (uint32_t j = 0; j < H; ++j) {
    int64_t r = lut_lookup(g->lut_sig, rshift_round(acc_i[j] + acc_h[j], s));
    int64_t z = lut_lookup(g->lut_sig, rshift_round(acc_i[H + j] + acc_h[H + j], s));
    /* n-gate: r (Q14) gates the hidden-side accumulator only */
    int64_t n_acc = acc_i[2 * H + j] + rshift_round(r * acc_h[2 * H + j], 14);
    int64_t n = lut_lookup(g->lut_tanh, rshift_round(n_acc, s));
    h_next[j] = sat16(rshift_round((Q14_ONE - z) * n, 14) +
                      rshift_round(z * (int64_t)ctx->h[j], 14));
  }
  for (uint32_t j = 0; j < H; ++j) ctx->h[j] = h_next[j];

  /* logits + argmax; ties break toward the lowest id (matches np.argmax) */
  uint32_t best = 0;
  int64_t best_v = 0;
  for (uint32_t v = 0; v < V; ++v) {
    int64_t sum = 0;
    const uint8_t *row = g->w_out + v * H;
    for (uint32_t j = 0; j < H; ++j) sum += (int64_t)(int8_t)row[j] * ctx->h[j];
    sum += read_i32be(g->b_out + 4 * v);
    if (v == 0 || sum > best_v) { best = v; best_v = sum; }
  }

  ctx->cur = (uint16_t)best;
  if (best == 0) { /* EOS */
    ctx->finished = 1;
    return NGPT_EOS;
  }
  return (int)g->charset[best];
}
