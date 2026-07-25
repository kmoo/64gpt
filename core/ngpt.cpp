/**
 * 64GPT inference engine — blob parsing + model dispatch.
 * See ngpt.h for the API contract and docs/03-blob-format.md for the format.
 */
#include "ngpt.h"
#include "ngpt_gru.h"

uint16_t ngpt_read_u16be(const uint8_t *p)
{
  return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

uint32_t ngpt_read_u32be(const uint8_t *p)
{
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
}

int ngpt_load(ngpt_model *m, const void *blob, uint32_t blob_len)
{
  const uint8_t *b = (const uint8_t *)blob;

  m->blob = b;
  m->blob_len = blob_len;
  m->payload = 0;
  m->payload_len = 0;

  if (blob_len < NGPT_HEADER_SIZE) return NGPT_ERR_TRUNCATED;
  if (ngpt_read_u32be(b) != NGPT_MAGIC) return NGPT_ERR_MAGIC;

  m->format_version = ngpt_read_u16be(b + 4);
  m->model_type     = ngpt_read_u16be(b + 6);
  uint32_t payload_len = ngpt_read_u32be(b + 8);

  if (m->format_version != NGPT_FORMAT_VERSION &&
      m->format_version != NGPT_FORMAT_VERSION_TRIE) return NGPT_ERR_VERSION;
  if (m->model_type != NGPT_MODEL_CANNED &&
      m->model_type != NGPT_MODEL_GRU) return NGPT_ERR_MODEL_TYPE;
  if (payload_len > blob_len - NGPT_HEADER_SIZE) return NGPT_ERR_TRUNCATED;

  m->payload = b + NGPT_HEADER_SIZE;
  m->payload_len = payload_len;

  if (m->model_type == NGPT_MODEL_GRU) return ngpt_gru_load(m);
  return NGPT_OK;
}

void ngpt_reset(ngpt_ctx *ctx, const ngpt_model *m, const char *prompt)
{
  ctx->model = m;
  ctx->pos = 0;
  ctx->finished = 0;
  ctx->cur = 0; /* GRU: generation starts from the EOS token... */
  for (uint32_t j = 0; j < NGPT_GRU_MAX_HIDDEN; ++j) ctx->h[j] = 0; /* ...and h = 0 */
  ctx->sample_on = 0; /* greedy unless ngpt_set_sampler is called after */
  ctx->inv_t_q8 = 256;
  ctx->top_k = 1;
  ctx->rng = 1;
  ctx->minp_shift = 0; /* M12.1: gate off unless ngpt_set_minp is called */
  ctx->trie_on = 0;    /* M12.1 phase 4: guard off unless ngpt_set_trie_guard is called */
  ctx->trie_node = 0;  /* root */

  /* M3 conditioning: prime the GRU on the prompt (h-updates only, no
   * emission). The canned model ignores prompts. */
  if (m && m->model_type == NGPT_MODEL_GRU && prompt && prompt[0])
    ngpt_gru_prime(ctx, prompt);
}

void ngpt_set_sampler(ngpt_ctx *ctx, uint32_t seed, uint16_t inv_t_q8,
                      uint16_t top_k)
{
  ctx->sample_on = 1;
  ctx->inv_t_q8 = inv_t_q8;
  ctx->top_k = top_k > 0 ? top_k : 1;
  ctx->rng = seed != 0 ? seed : 1; /* 0 is xorshift32's fixed point */
}

void ngpt_set_minp(ngpt_ctx *ctx, uint8_t shift)
{
  ctx->minp_shift = shift;
}

void ngpt_set_trie_guard(ngpt_ctx *ctx, uint8_t on)
{
  ctx->trie_on = on;
}

int ngpt_step(ngpt_ctx *ctx)
{
  const ngpt_model *m = ctx->model;
  if (ctx->finished || !m || !m->payload) return NGPT_EOS;

  if (m->model_type == NGPT_MODEL_GRU) return ngpt_gru_step(ctx);

  /* NGPT_MODEL_CANNED: emit the payload bytes verbatim, then EOS. */
  if (ctx->pos >= m->payload_len) {
    ctx->finished = 1;
    return NGPT_EOS;
  }
  return (int)m->payload[ctx->pos++];
}
