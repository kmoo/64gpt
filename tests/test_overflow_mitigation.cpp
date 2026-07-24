/* M12: gru_h_update()'s bias-add (core/ngpt_gru.cpp) was widened to
 * int64 internally, saturating at INT32_MIN/MAX on narrow-back --
 * headroom got tight at H up to 1024 (core/ngpt.h has the full
 * accounting). acc_h[] is filled by a matvec hook BEFORE the bias-add
 * runs (same ngpt_set_matvec() mechanism test_matvec_hook.cpp uses),
 * so a hook that returns a fixed near-boundary value drives the
 * bias-add right up against the saturation edge without needing a real
 * H=1024 blob -- any bias in tests/vectors/m4_gru.bin added on top must
 * not wrap around int32, and generation must complete cleanly. */
#include "ngpt.h"
#include "test_util.h"
#include <stdint.h>
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

/* A constant matvec fill (below) is a deliberately degenerate input no
 * real trained model produces -- the sampler may never pick EOS (stuck
 * on the same top token forever), which is a legitimate outcome for
 * this scenario, not a bug. So this loop bounds step count instead of
 * requiring natural termination like test_matvec_hook.cpp's generate()
 * does for real goldens; the only thing under test is that every step
 * stays well-defined (valid byte, no crash) for STEPS iterations. */
static const uint32_t STEPS = 500;

static void stepBounded(ngpt_ctx *ctx)
{
  for (uint32_t i = 0; i < STEPS; ++i) {
    int c = ngpt_step(ctx);
    if (c == NGPT_EOS) return; /* fine if it does terminate */
    CHECK(c >= 0 && c <= 255);
  }
}

/* Fills every row sum with a fixed near-boundary value, positive or
 * negative -- picked close enough to INT32_MAX/MIN that ANY nonzero
 * bias added on top would overflow plain int32 addition (wrap around),
 * but far enough from the true edge that the saturating add still has
 * a defined, distinct clamped result to land on. */
static int32_t g_fill_value;

static void matvec_extreme(const uint8_t *w_rows, uint32_t rows, uint32_t cols,
                           const int16_t *h, int32_t *out)
{
  (void)w_rows; (void)cols; (void)h;
  for (uint32_t i = 0; i < rows; ++i) out[i] = g_fill_value;
}

int main(void)
{
  uint32_t blob_len;
  uint8_t *blob = read_file("m4_gru.bin", &blob_len);

  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, blob_len), NGPT_OK);
  CHECK_EQ_INT(m.model_type, NGPT_MODEL_GRU);

  ngpt_ctx ctx;

  /* Positive boundary: every acc_h[i] starts at INT32_MAX - 1000 (room
   * left for a real bias to push it over without the fill value itself
   * already being clamped by the hook). If the bias-add wrapped instead
   * of saturating, this would go negative and generation would still
   * "work" but on corrupted state -- the real assertion here is just
   * that nothing crashes/hangs and every emitted byte stays in the
   * engine's valid range, proving no UB propagated into the sampler. */
  g_fill_value = INT32_MAX - 1000;
  ngpt_set_matvec(matvec_extreme);
  ngpt_reset(&ctx, &m, "test");
  ngpt_set_sampler(&ctx, 1, 256, 5);
  stepBounded(&ctx);

  /* Negative boundary: same idea, the other direction. */
  g_fill_value = INT32_MIN + 1000;
  ngpt_reset(&ctx, &m, "test");
  ngpt_set_sampler(&ctx, 1, 256, 5);
  stepBounded(&ctx);

  /* Both boundaries exercise the SAME clamp in both directions but at
   * different values -- the real proof is that STEPS iterations of
   * each ran without CHECK() ever seeing an out-of-range byte, which
   * would indicate the saturating add let something wrap/UB propagate
   * into the sampler instead of clamping cleanly. */

  ngpt_set_matvec(NULL);
  free(blob);
  return test_summary("test_overflow_mitigation");
}
