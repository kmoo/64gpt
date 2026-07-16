/* GRU model (type 1) tests against the trainer-committed goldens:
 * full generation byte-identical, per-step hidden state bit-exact vs
 * tests/vectors/m2_trace.bin, EOS sticky, reset regenerates. If this is
 * green, the ROM self-test on the same blob must PASS. */
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
  uint32_t blob_len, want_len, trace_len;
  uint8_t *blob = read_file("m2_gru.bin", &blob_len);
  uint8_t *want = read_file("m2_expected.txt", &want_len);
  uint8_t *tr = read_file("m2_trace.bin", &trace_len);

  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, blob_len), NGPT_OK);
  CHECK_EQ_INT(m.model_type, NGPT_MODEL_GRU);

  /* full generation is byte-identical to the golden */
  uint8_t got[4096];
  ngpt_ctx ctx;
  ngpt_reset(&ctx, &m, "");
  uint32_t got_len = generate(&ctx, got, sizeof(got));
  CHECK_EQ_INT(got_len, want_len);
  CHECK(memcmp(got, want, want_len) == 0);

  /* EOS sticky; reset regenerates identically */
  CHECK_EQ_INT(ngpt_step(&ctx), NGPT_EOS);
  uint8_t again[4096];
  ngpt_reset(&ctx, &m, "");
  CHECK_EQ_INT(generate(&ctx, again, sizeof(again)), got_len);
  CHECK(memcmp(again, got, got_len) == 0);

  /* per-step bit-exactness vs the reference-impl trace:
   * u32 count, then per step u16 input_id, u16 argmax_id, H x i16 h */
  const uint32_t H = m.gru.H;
  uint32_t steps = ngpt_read_u32be(tr);
  CHECK_EQ_INT(trace_len, 4 + steps * (4 + 2 * H));
  ngpt_reset(&ctx, &m, "");
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
  free(want);
  free(tr);
  return test_summary("test_gru_model");
}
