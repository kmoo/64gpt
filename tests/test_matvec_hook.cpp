/* M6.1 matvec hook tests. ngpt_set_matvec lets a platform register an
 * accelerated W_hh matvec (on the N64: the RSP); NULL (the default)
 * keeps the pure-CPU path, so this suite remains the bit-exactness
 * referee. Checks: (1) the hook is really on the hot path — a corrupted
 * callback changes the output; (2) a correct callback replays every M4
 * golden byte-identically; (3) NULL restores the default exactly. */
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

/* Reference callback: the same raw row.h sums the engine computes
 * itself (no biases — the engine owns those). Also records what it was
 * called with, so the test can assert the contract. */
static uint32_t g_calls;
static const uint8_t *g_last_w;
static uint32_t g_last_rows, g_last_cols;

static void matvec_ref(const uint8_t *w_rows, uint32_t rows, uint32_t cols,
                       const int16_t *h, int32_t *out)
{
  ++g_calls;
  g_last_w = w_rows;
  g_last_rows = rows;
  g_last_cols = cols;
  for (uint32_t i = 0; i < rows; ++i) {
    int32_t sum = 0;
    const uint8_t *row = w_rows + i * cols;
    for (uint32_t j = 0; j < cols; ++j) sum += (int32_t)(int8_t)row[j] * h[j];
    out[i] = sum;
  }
}

/* Deliberately wrong sums: if the engine consumes the callback's
 * results, generation must diverge from the reference. */
static void matvec_broken(const uint8_t *w_rows, uint32_t rows, uint32_t cols,
                          const int16_t *h, int32_t *out)
{
  matvec_ref(w_rows, rows, cols, h, out);
  for (uint32_t i = 0; i < rows; ++i) out[i] += 1 << 20;
}

int main(void)
{
  uint32_t blob_len, goldens_len;
  uint8_t *blob = read_file("m4_gru.bin", &blob_len);
  uint8_t *gold = read_file("m4_goldens.bin", &goldens_len);

  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, blob_len), NGPT_OK);
  CHECK_EQ_INT(m.model_type, NGPT_MODEL_GRU);

  uint32_t seed = ngpt_read_u32be(gold);
  uint16_t inv_t = ngpt_read_u16be(gold + 4);
  uint16_t top_k = ngpt_read_u16be(gold + 6);
  uint16_t pairs = ngpt_read_u16be(gold + 8);
  const uint8_t *p = gold + 10;

  ngpt_ctx ctx;
  for (uint16_t i = 0; i < pairs; ++i) {
    uint16_t plen = ngpt_read_u16be(p); p += 2;
    CHECK(plen <= 96);
    char prompt[97];
    memcpy(prompt, p, plen); prompt[plen] = 0; p += plen;
    uint16_t rlen = ngpt_read_u16be(p); p += 2;
    const uint8_t *want = p; p += rlen;
    CHECK((uint32_t)(p - gold) <= goldens_len);

    /* every M4 golden replays byte-identically through the callback */
    g_calls = 0;
    ngpt_set_matvec(matvec_ref);
    uint8_t got[4096];
    ngpt_reset(&ctx, &m, prompt);
    ngpt_set_sampler(&ctx, seed, inv_t, top_k);
    uint32_t got_len = generate(&ctx, got, sizeof(got));
    CHECK_EQ_INT(got_len, rlen);
    CHECK(memcmp(got, want, rlen) == 0);
    CHECK(g_calls > 0);                      /* the hook actually ran   */
    CHECK(g_last_w == m.gru.w_hh);           /* on W_hh, whole matrix   */
    CHECK_EQ_INT(g_last_rows, 3u * m.gru.H);
    CHECK_EQ_INT(g_last_cols, m.gru.H);

    if (i == 0) {
      /* a broken callback must change the output — proves the engine
       * consumes the callback's sums rather than recomputing */
      ngpt_set_matvec(matvec_broken);
      uint8_t bad[4096];
      ngpt_reset(&ctx, &m, prompt);
      ngpt_set_sampler(&ctx, seed, inv_t, top_k);
      uint32_t bad_len = generate(&ctx, bad, sizeof(bad));
      CHECK(bad_len != rlen || memcmp(bad, want, rlen) != 0);

      /* NULL restores the pure-CPU default exactly */
      ngpt_set_matvec(NULL);
      g_calls = 0;
      uint8_t back[4096];
      ngpt_reset(&ctx, &m, prompt);
      ngpt_set_sampler(&ctx, seed, inv_t, top_k);
      uint32_t back_len = generate(&ctx, back, sizeof(back));
      CHECK_EQ_INT(back_len, rlen);
      CHECK(memcmp(back, want, rlen) == 0);
      CHECK_EQ_INT(g_calls, 0u);
      ngpt_set_matvec(matvec_ref); /* back on for the remaining pairs */
    }
  }
  CHECK_EQ_INT((uint32_t)(p - gold), goldens_len);

  ngpt_set_matvec(NULL);
  free(blob);
  free(gold);
  return test_summary("test_matvec_hook");
}
