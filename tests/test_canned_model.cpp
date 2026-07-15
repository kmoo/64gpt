/* Streaming-API tests against the committed golden vectors.
 * The exact same blob (tests/vectors/m1_canned.bin == game/rawfs/model.bin)
 * and the exact same expected bytes are replayed by the ROM's boot
 * self-test — host-green here is the promise that the N64 shows
 * SELFTEST PASS. */
#include "ngpt.h"
#include "test_util.h"
#include <stdio.h>
#include <stdlib.h>

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

/* run a full generation into out (cap chars), return number of chars */
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
  uint32_t blob_len, want_len;
  uint8_t *blob = read_file("m1_canned.bin", &blob_len);
  uint8_t *want = read_file("m1_expected.txt", &want_len);

  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, blob_len), NGPT_OK);
  CHECK_EQ_INT(m.model_type, NGPT_MODEL_CANNED);

  /* full generation is byte-identical to the golden */
  uint8_t got[4096];
  ngpt_ctx ctx;
  ngpt_reset(&ctx, &m, "");
  uint32_t got_len = generate(&ctx, got, sizeof(got));
  CHECK_EQ_INT(got_len, want_len);
  CHECK(got_len <= sizeof(got));
  CHECK(memcmp(got, want, want_len) == 0);

  /* EOS is sticky: stepping past the end keeps returning EOS */
  CHECK_EQ_INT(ngpt_step(&ctx), NGPT_EOS);
  CHECK_EQ_INT(ngpt_step(&ctx), NGPT_EOS);

  /* reset regenerates the identical text (the A-button path) */
  uint8_t again[4096];
  ngpt_reset(&ctx, &m, "");
  uint32_t again_len = generate(&ctx, again, sizeof(again));
  CHECK_EQ_INT(again_len, got_len);
  CHECK(memcmp(again, got, got_len) == 0);

  /* per-frame budget: stepping in small chunks (as the game loop does)
   * yields the same byte stream as stepping continuously */
  ngpt_reset(&ctx, &m, "");
  uint32_t n = 0;
  int done = 0;
  uint8_t chunked[4096];
  while (!done) {
    for (int i = 0; i < 3; ++i) {          /* "3 chars per frame" */
      int c = ngpt_step(&ctx);
      if (c == NGPT_EOS) { done = 1; break; }
      if (n < sizeof(chunked)) chunked[n] = (uint8_t)c;
      ++n;
    }
  }
  CHECK_EQ_INT(n, got_len);
  CHECK(memcmp(chunked, got, got_len) == 0);

  free(blob);
  free(want);
  return test_summary("test_canned_model");
}
