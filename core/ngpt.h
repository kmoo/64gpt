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
};

typedef struct ngpt_model {
  const uint8_t *blob;
  uint32_t blob_len;
  uint16_t format_version;
  uint16_t model_type;
  const uint8_t *payload;  /* points into blob, after the 12-byte header */
  uint32_t payload_len;
} ngpt_model;

typedef struct ngpt_ctx {
  const ngpt_model *model;
  uint32_t pos;      /* canned model: next payload byte to emit */
  uint8_t finished;  /* 1 once EOS has been reached             */
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
