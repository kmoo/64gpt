/**
 * 64GPT inference engine — public streaming API.
 *
 * Portable C-style C++: no exceptions, no RTTI, no heap, no floats.
 * The same source compiles for host tests (gcc/clang, little-endian) and
 * the N64 (mips64 gcc, big-endian). All multi-byte values in the model
 * blob are parsed byte-by-byte, so results are identical on both.
 *
 * Lifecycle:
 *   ngpt_model model;
 *   ngpt_ctx   ctx;
 *   ngpt_load(&model, blob, blob_len);   // validate + index the blob
 *   ngpt_reset(&ctx, &model, prompt);    // start a new generation
 *   int c = ngpt_step(&ctx);             // next byte, or NGPT_EOS when done
 *
 * The caller owns the blob memory; it must outlive the model/ctx.
 * ngpt_step never blocks and does a bounded amount of work, so the game
 * loop can call it a few times per frame to stream text.
 */
#pragma once
#include <stdint.h>

/* ---- blob format constants (see docs/03-blob-format.md) ---- */
#define NGPT_MAGIC          0x4E475054u /* "NGPT" big-endian */
#define NGPT_FORMAT_VERSION 1
#define NGPT_HEADER_SIZE    12

/* model types */
#define NGPT_MODEL_CANNED   0  /* v0: fixed byte string (walking skeleton) */
#define NGPT_MODEL_GRU      1  /* v1: quantized GRU (from milestone M2)    */

/* ngpt_step() sentinel: generation finished */
#define NGPT_EOS (-1)

/* error codes (negative), NGPT_OK on success */
enum {
  NGPT_OK             = 0,
  NGPT_ERR_TRUNCATED  = -2, /* blob shorter than header, or payload cut off */
  NGPT_ERR_MAGIC      = -3, /* first 4 bytes are not "NGPT"                 */
  NGPT_ERR_VERSION    = -4, /* format version not understood                */
  NGPT_ERR_MODEL_TYPE = -5, /* model type not understood                    */
  NGPT_ERR_DIMS       = -6, /* GRU dims exceed the static caps below        */
};

/* Static caps: ngpt_ctx carries the hidden state inline (no heap), so the
 * dims a blob may declare are bounded at load time. M2 shipped H=32, M4's
 * ~100K-param model H=128; M6.1 raised the cap to 256 for the H=256+
 * generative model (M7's "magic zone"); M9 raised it again to 320 for
 * compositional conditioning's extra capacity (docs/milestones/m9.md
 * section 6).
 *
 * M12 bumps this to 1024 — explicit human sign-off (Luke, 2026-07-23),
 * per this file's own frozen-interface status. Grounded in
 * docs/spikes/rsp-matvec-ktile.md's K-chunked RSP kernel, hardware-
 * verified at H=1024 (9/9 bit-exact XCHK passes across the spike's full
 * (chunk, H) sweep, including this exact configuration). That spike's own
 * recommendation was H=768 (real margin on every axis); H=1024 was
 * chosen anyway as "how far can this actually go" — accepted with eyes
 * open, not a default.
 *
 * int32 overflow at this H: a row sum's REALISTIC bound (h's Q14-
 * quantized |h|<=16384, not the worst-case 32767) is 1024*127*16384 ≈
 * 2.13e9 — 99.2% of int32's hard ceiling (2^31-1 ≈ 2.15e9), essentially
 * no margin, a materially different situation from H=320's comfortable
 * 62%. Two things are true about this, both disclosed rather than
 * quietly accepted:
 *   1. The RSP kernel's own int32 accumulator (rsp_ngpt.S) is NOT
 *      widened — real hardware has no native 64-bit path, and widening
 *      it costs real cycles this kernel's whole point is to avoid. This
 *      risk is the same one the spike disclosed and never closed; it
 *      still hasn't been re-validated against a REAL trained model's
 *      bias magnitudes, only a gibberish one.
 *   2. core/ngpt_gru.cpp's CPU reference path (used by the boot self-
 *      test's CPU-vs-RSP cross-check as ground truth) IS widened to
 *      int64 internally, saturating on narrow-back. This doesn't shrink
 *      #1's risk, but it does mean that if the RSP path's int32 path
 *      ever DOES overflow on real weights, it diverges from the now-
 *      correct reference instead of both paths quietly wrapping the same
 *      way and still agreeing — turning a silent numeric-corruption
 *      failure mode into a loud, on-screen XCHK FAIL, the existing
 *      correctness gate this project already trusts, made trustworthy
 *      for this specific new risk. */
#define NGPT_GRU_MAX_HIDDEN 1024
#define NGPT_GRU_MAX_VOCAB  96

/* GRU payload view (model type 1): pointers into the blob, set up once by
 * ngpt_load. Weights are read big-endian byte-by-byte at inference time —
 * nothing is copied or converted, so there is no unpacking buffer and no
 * alignment requirement. Layout: docs/milestones/m2.md. */
typedef struct ngpt_gru_view {
  uint16_t H, V;
  uint8_t k_w, k_out;
  const uint8_t *charset;   /* V bytes, [0] = 0x00 EOS slot       */
  const uint8_t *lut_sig;   /* 256 x i16 BE, Q14 out              */
  const uint8_t *lut_tanh;  /* 256 x i16 BE                       */
  const uint8_t *w_ih;      /* 3H*V x i8, row-major, gates r,z,n  */
  const uint8_t *w_hh;      /* 3H*H x i8                          */
  const uint8_t *b_ih;      /* 3H x i32 BE, scale 2^(k_w+14)      */
  const uint8_t *b_hh;      /* 3H x i32 BE                        */
  const uint8_t *w_out;     /* V*H x i8                           */
  const uint8_t *b_out;     /* V x i32 BE, scale 2^(k_out+14)     */
} ngpt_gru_view;

typedef struct ngpt_model {
  const uint8_t *blob;
  uint32_t blob_len;
  uint16_t format_version;
  uint16_t model_type;
  const uint8_t *payload;  /* points into blob, after the 12-byte header */
  uint32_t payload_len;
  ngpt_gru_view gru;       /* valid only when model_type == NGPT_MODEL_GRU */
} ngpt_model;

typedef struct ngpt_ctx {
  const ngpt_model *model;
  uint32_t pos;      /* canned model: next payload byte to emit */
  uint8_t finished;  /* 1 once EOS has been reached             */
  /* GRU state (model type 1) */
  int16_t h[NGPT_GRU_MAX_HIDDEN];  /* hidden state, Q14 */
  uint16_t cur;                    /* last emitted token id (0 = EOS) */
  /* M4 sampler config — sample_on == 0 (the ngpt_reset default) keeps
   * the M2/M3 greedy-argmax behavior exactly. */
  uint8_t sample_on;
  uint16_t inv_t_q8;               /* round(256 / temperature)          */
  uint16_t top_k;
  uint32_t rng;                    /* xorshift32 state, never 0         */
} ngpt_ctx;

/* Byte-oriented big-endian readers — the reason the blob parses
 * identically on the little-endian host and the big-endian N64. */
uint16_t ngpt_read_u16be(const uint8_t *p);
uint32_t ngpt_read_u32be(const uint8_t *p);

/* Validate the blob and fill in `m`. Returns NGPT_OK or an NGPT_ERR_*.
 * Does not copy: `blob` must stay alive while the model is in use. */
int ngpt_load(ngpt_model *m, const void *blob, uint32_t blob_len);

/* Begin a new generation. `prompt` selects conditioning (NPC/mood/event);
 * the canned model (v0) ignores it — pass "" until M3. */
void ngpt_reset(ngpt_ctx *ctx, const ngpt_model *m, const char *prompt);

/* Produce the next character: returns 0..255, or NGPT_EOS when the
 * generation is complete. After EOS it keeps returning NGPT_EOS until
 * the next ngpt_reset. */
int ngpt_step(ngpt_ctx *ctx);

/* M6.1: optional accelerated matvec backend (additive — the frozen
 * 4-call streaming API is untouched). The engine's hot loop is the
 * W_hh matvec: rows x cols int8 weights (row-major, straight from the
 * blob, no alignment guarantee) dotted with the int16 hidden state.
 * A registered callback must write the raw row.h sum for every row to
 * out[0..rows-1] — exactly sum((int8)row[j] * h[j]), no biases (the
 * engine adds those). NULL (the default) keeps the pure-CPU path, so
 * the host test suite stays the bit-exactness referee. On the N64 the
 * game registers an RSP-backed callback; the engine is single-threaded,
 * so the hook is a plain global. Dims come from the blob — a backend
 * must check rows/cols and fall back to a CPU loop for shapes it
 * cannot handle (that is what keeps this working at H=256+). */
typedef void (*ngpt_matvec_fn)(const uint8_t *w_rows, uint32_t rows,
                               uint32_t cols, const int16_t *h, int32_t *out);
void ngpt_set_matvec(ngpt_matvec_fn fn);

/* M4: enable temperature/top-k sampling for this generation (additive —
 * never calling this keeps greedy argmax). Call AFTER ngpt_reset, which
 * turns sampling off. seed 0 is remapped to 1 (xorshift32 fixed point);
 * design + fixed-point formats: docs/milestones/m4.md. */
void ngpt_set_sampler(ngpt_ctx *ctx, uint32_t seed, uint16_t inv_t_q8,
                      uint16_t top_k);
