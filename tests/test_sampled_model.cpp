/* Sampled GRU tests against the trainer-committed M4 goldens: for each
 * (prompt, response) pair in m4_goldens.bin, ngpt_reset with the prompt
 * + ngpt_set_sampler with the pinned seed/params, then step-until-EOS
 * must be byte-identical to the seeded sampled response; EOS sticky and
 * a re-reset+reseed regenerates; pair-0 generation is bit-exact per step
 * vs m4_trace.bin. If this is green, the ROM self-test on the same blob
 * must PASS. */
#include "ngpt.h"
#include "test_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NGPT_VECTOR_DIR
#error "NGPT_VECTOR_DIR must be defined by the build"
#endif

static uint8_t *read_file(const char *name, uint32_t *out_len)
{
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s", NGPT_VECTOR_DIR, name);
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = (uint8_t *)malloc((size_t)len);
  if (fread(buf, 1, (size_t)len, f) != (size_t)len) { fprintf(stderr, "short read %s\n", path); exit(2); }
  fclose(f);
  *out_len = (uint32_t)len;
  return buf;
}

static uint32_t generate(ngpt_ctx *ctx, uint8_t *out, uint32_t cap)
{
  uint32_t n = 0;
  int c;
  while ((c = ngpt_step(ctx)) != NGPT_EOS) {
    CHECK(c >= 0 && c <= 255);
    if (n < cap) out[n] = (uint8_t)c;
    ++n;
    if (n > 100000) { fprintf(stderr, "runaway generation\n"); exit(2); }
  }
  return n;
}

int main(void)
{
  uint32_t blob_len, goldens_len, trace_len;
  uint8_t *blob = read_file("m4_gru.bin", &blob_len);
  uint8_t *gold = read_file("m4_goldens.bin", &goldens_len);
  uint8_t *tr = read_file("m4_trace.bin", &trace_len);

  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, blob_len), NGPT_OK);
  CHECK_EQ_INT(m.model_type, NGPT_MODEL_GRU);

  /* m4_goldens.bin: u32 sample seed, u16 inv_t_q8, u16 top_k, u16 pair
   * count, then per pair u16 len + prompt bytes, u16 len + response
   * bytes — parsed byte-by-byte */
  uint32_t seed = ngpt_read_u32be(gold);
  uint16_t inv_t = ngpt_read_u16be(gold + 4);
  uint16_t top_k = ngpt_read_u16be(gold + 6);
  uint16_t pairs = ngpt_read_u16be(gold + 8);
  const uint8_t *p = gold + 10;
  char pair0_prompt[97] = {0};

  ngpt_ctx ctx;
  for (uint16_t i = 0; i < pairs; ++i) {
    uint16_t plen = ngpt_read_u16be(p); p += 2;
    CHECK(plen <= 96);
    char prompt[97];
    memcpy(prompt, p, plen); prompt[plen] = 0; p += plen;
    uint16_t rlen = ngpt_read_u16be(p); p += 2;
    const uint8_t *want = p; p += rlen;
    CHECK((uint32_t)(p - gold) <= goldens_len);
    if (i == 0) memcpy(pair0_prompt, prompt, plen + 1);

    /* seeded sampled generation is byte-identical to the golden */
    uint8_t got[4096];
    ngpt_reset(&ctx, &m, prompt);
    ngpt_set_sampler(&ctx, seed, inv_t, top_k);
    uint32_t got_len = generate(&ctx, got, sizeof(got));
    CHECK_EQ_INT(got_len, rlen);
    CHECK(memcmp(got, want, rlen) == 0);

    /* EOS sticky; reset + same seed regenerates identically */
    CHECK_EQ_INT(ngpt_step(&ctx), NGPT_EOS);
    uint8_t again[4096];
    ngpt_reset(&ctx, &m, prompt);
    ngpt_set_sampler(&ctx, seed, inv_t, top_k);
    CHECK_EQ_INT(generate(&ctx, again, sizeof(again)), got_len);
    CHECK(memcmp(again, got, got_len) == 0);
  }
  CHECK_EQ_INT((uint32_t)(p - gold), goldens_len);

  /* sampling off (no ngpt_set_sampler after reset) stays pure greedy:
   * two runs agree with each other — the M2/M3 behavior is untouched */
  {
    uint8_t a[4096], b[4096];
    ngpt_reset(&ctx, &m, pair0_prompt);
    uint32_t la = generate(&ctx, a, sizeof(a));
    ngpt_reset(&ctx, &m, pair0_prompt);
    uint32_t lb = generate(&ctx, b, sizeof(b));
    CHECK_EQ_INT(la, lb);
    CHECK(memcmp(a, b, la) == 0);
  }

  /* pair-0 per-step bit-exactness vs the sampled reference trace:
   * u32 count, then per step u16 input_id, u16 chosen_id, H x i16 h */
  const uint32_t H = m.gru.H;
  uint32_t steps = ngpt_read_u32be(tr);
  CHECK_EQ_INT(trace_len, 4 + steps * (4 + 2 * H));
  ngpt_reset(&ctx, &m, pair0_prompt);
  ngpt_set_sampler(&ctx, seed, inv_t, top_k);
  const uint8_t *rec = tr + 4;
  for (uint32_t st = 0; st < steps; ++st, rec += 4 + 2 * H) {
    uint16_t want_in = ngpt_read_u16be(rec);
    uint16_t want_next = ngpt_read_u16be(rec + 2);
    CHECK_EQ_INT(ctx.cur, want_in);
    ngpt_step(&ctx);
    CHECK_EQ_INT(ctx.cur, want_next);
    for (uint32_t j = 0; j < H; ++j) {
      int16_t want_h = (int16_t)ngpt_read_u16be(rec + 4 + 2 * j);
      CHECK_EQ_INT(ctx.h[j], want_h);
    }
  }
  CHECK_EQ_INT(ctx.cur, 0); /* trace ends on EOS */

  free(blob);
  free(gold);
  free(tr);
  return test_summary("test_sampled_model");
}
