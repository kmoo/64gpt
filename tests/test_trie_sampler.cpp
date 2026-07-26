/* M12.1 phase 4: bit-exactness test for the lexicon-trie decode guard
 * (ngpt_set_trie_guard / ngpt_sample_pick's trie branch, core/
 * ngpt_sample.cpp). Same style as test_minp_sampler.cpp: no trained
 * blob needed, just a hand-built ngpt_model with charset + a hand-built
 * trie node array, matching ref_impl.py's build_word_trie() layout
 * exactly (char u8, flags u8, first_child u16, next_sibling u16;
 * NGPT_TRIE_NONE = 0xFFFF; node 0 = root).
 *
 * The word list ["CAT", "CAN", "AT", "IT"] over a 6-symbol vocab
 * (id0=EOS, 1='C', 2='A', 3='T', 4='N', 5='I') was hand-traced through
 * build_word_trie()'s algorithm (see docs/milestones/m12.1.md phase 4)
 * to produce the 9-node array below -- the same cross-check discipline
 * test_minp_sampler.cpp uses against ref_impl.sample_from_logits, here
 * against ref_impl.build_word_trie / sample_from_logits_trie. */
#include "ngpt.h"
#include "ngpt_sample.h"
#include "test_util.h"
#include <string.h>

/* charset: id -> byte. id0 = EOS (0x00 by convention, never read as a word byte). */
static const uint8_t CHARSET[] = {0, 'C', 'A', 'T', 'N', 'I'};
enum { ID_EOS = 0, ID_C = 1, ID_A = 2, ID_T = 3, ID_N = 4, ID_I = 5, V = 6 };

/* nodes[i] = {char, flags, first_child, next_sibling}; TRIE_NONE = 0xFFFF.
 * 0: root            -> child 1 ('C')
 * 1: 'C'              -> child 2 ('A'),           sibling 5 ('A', the AT branch)
 * 2: 'A' (under C)     -> child 3 ('T')
 * 3: 'T' (CAT, end)                                sibling 4 ('N', CAN)
 * 4: 'N' (CAN, end)
 * 5: 'A' (root sibling) -> child 6 ('T'),          sibling 7 ('I')
 * 6: 'T' (AT, end)
 * 7: 'I' (root sibling)  -> child 8 ('T')
 * 8: 'T' (IT, end)
 */
static const uint8_t TRIE_NODES[9][6] = {
  /* 0 */ {0,    0, 0x00, 0x01, 0xFF, 0xFF},
  /* 1 */ {'C',  0, 0x00, 0x02, 0x00, 0x05},
  /* 2 */ {'A',  0, 0x00, 0x03, 0xFF, 0xFF},
  /* 3 */ {'T',  1, 0xFF, 0xFF, 0x00, 0x04},
  /* 4 */ {'N',  1, 0xFF, 0xFF, 0xFF, 0xFF},
  /* 5 */ {'A',  0, 0x00, 0x06, 0x00, 0x07},
  /* 6 */ {'T',  1, 0xFF, 0xFF, 0xFF, 0xFF},
  /* 7 */ {'I',  0, 0x00, 0x08, 0xFF, 0xFF},
  /* 8 */ {'T',  1, 0xFF, 0xFF, 0xFF, 0xFF},
};

static void makeModel(ngpt_model *m, const uint8_t *trie_nodes, uint32_t trie_count)
{
  memset(m, 0, sizeof(*m));
  m->model_type = NGPT_MODEL_GRU;
  m->gru.k_out = 6;
  m->gru.charset = CHARSET;
  m->gru.trie_nodes = trie_nodes;
  m->gru.trie_count = trie_count;
}

static void makeCtx(ngpt_ctx *ctx, const ngpt_model *m, uint16_t top_k,
                    uint16_t inv_t_q8, uint8_t minp_shift, uint8_t trie_on,
                    uint16_t trie_node, uint32_t rng_seed)
{
  memset(ctx, 0, sizeof(*ctx));
  ctx->model = m;
  ctx->sample_on = 1;
  ctx->top_k = top_k;
  ctx->inv_t_q8 = inv_t_q8;
  ctx->minp_shift = minp_shift;
  ctx->trie_on = trie_on;
  ctx->trie_node = trie_node;
  ctx->rng = rng_seed;
}

/* Tiny local xorshift for generating deterministic test-data logits --
 * distinct from ngpt_sample.cpp's internal sampler RNG, which stays
 * fully exercised (and advanced) via ngpt_sample_pick itself. */
static uint32_t test_rng(uint32_t *state)
{
  uint32_t x = *state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  *state = x;
  return x;
}

/* logits peaked on `fav` -- with top_k=1 the sampler's order[0] is the
 * argmax, so this forces the deterministic greedy-trie branch (k==1). */
static void peak_logits(int32_t *lg, uint32_t V, uint32_t fav)
{
  for (uint32_t v = 0; v < V; ++v) lg[v] = 0;
  lg[fav] = 1000 << 14;
}

int main(void)
{
  /* ---- Fixture A: trie_nodes == NULL (a version-1 blob) must be a
   * total no-op even with trie_on requested -- ngpt_set_trie_guard on
   * an old blob must never change behavior. Compares against the same
   * call with trie_on=0. */
  {
    const int32_t logits[V] = {0, 100 << 14, 90 << 14, 80 << 14, 5 << 14, 3 << 14};
    ngpt_model m; makeModel(&m, 0, 0);
    for (uint32_t seed = 1; seed <= 8; ++seed) {
      ngpt_ctx ctxOff, ctxOnNoTrie;
      makeCtx(&ctxOff, &m, 6, 256, 0, /*trie_on=*/0, 0, seed);
      makeCtx(&ctxOnNoTrie, &m, 6, 256, 0, /*trie_on=*/1, 0, seed);
      uint32_t tokOff = ngpt_sample_pick(&ctxOff, logits, V);
      uint32_t tokOn = ngpt_sample_pick(&ctxOnNoTrie, logits, V);
      CHECK_EQ_INT((int)tokOff, (int)tokOn);
      CHECK_EQ_INT(ctxOff.rng, ctxOnNoTrie.rng);
    }
  }

  /* ---- Fixture B: at trie_node = 2 (after "CA"), only 'T' and 'N' are
   * legal continuations. Even when the model's logits STRONGLY favor
   * the illegal 'I', the guard must exclude it and fall back to the
   * best legal option ('T', completing CAT) -- mirrors test_sampler.py's
   * test_trie_blocks_a_word_not_in_the_corpus. */
  {
    const int32_t logits[V] = {0, 0, 0, 1000 << 14, 1 << 14, 10000 << 14};
    ngpt_model m; makeModel(&m, &TRIE_NODES[0][0], 9);
    ngpt_ctx ctx;
    makeCtx(&ctx, &m, V, 256, /*minp_shift=*/0, /*trie_on=*/1, /*trie_node=*/2, /*seed=*/1);
    uint32_t tok = ngpt_sample_pick(&ctx, logits, V);
    CHECK_EQ_INT((int)tok, ID_T);
    CHECK_EQ_INT((int)ctx.trie_node, 3); /* advanced to the CAT end-node */
  }

  /* ---- Fixture C: never invents a word. Drive ngpt_sample_pick
   * directly across many steps with adversarial pseudo-random logits,
   * for many seeds, and confirm every completed generation (chars up
   * to EOS) is exactly one of the 4 corpus words or empty --  mirrors
   * test_sampler.py's test_trie_never_invents_a_word_multistep. This
   * reduced vocab has no non-word byte besides EOS, so exactly one
   * "word" is generated per run, terminated by EOS. */
  {
    const char *words[] = {"CAT", "CAN", "AT", "IT"};
    ngpt_model m; makeModel(&m, &TRIE_NODES[0][0], 9);
    for (uint32_t seed = 1; seed <= 40; ++seed) {
      ngpt_ctx ctx;
      makeCtx(&ctx, &m, V, 200, /*minp_shift=*/0, /*trie_on=*/1, /*trie_node=*/0, seed);
      uint32_t rngData = seed * 2654435761u + 1;
      char out[8]; int outLen = 0;
      for (int step = 0; step < 6; ++step) {
        int32_t logits[V];
        for (uint32_t v = 0; v < V; ++v)
          logits[v] = (int32_t)(int16_t)(test_rng(&rngData) & 0xFFFF) << 8;
        uint32_t tok = ngpt_sample_pick(&ctx, logits, V);
        if (tok == ID_EOS) break;
        CHECK(outLen < (int)sizeof(out) - 1);
        out[outLen++] = (char)CHARSET[tok];
      }
      out[outLen] = '\0';

      if (outLen == 0) continue; /* empty generation: legal (EOS at root) */
      int found = 0;
      for (unsigned w = 0; w < sizeof(words) / sizeof(words[0]); ++w) {
        if (strcmp(out, words[w]) == 0) { found = 1; break; }
      }
      CHECK(found); /* else: invented a word not in the corpus */
    }
  }

  /* ---- Fixture D: WORD-BOUNDARY bytes -- the trie guard's core rule
   * that fixtures A-C structurally could not reach (their reduced vocab
   * had no non-word byte besides EOS, per fixture C's own note). A
   * boundary byte (space/punctuation) or EOS is legal ONLY at the root
   * or an end-of-word node, and picking one RESETS the walk to the root
   * (core/ngpt_sample.cpp trie_legal/trie_advance, mirroring ref_impl.py
   * _trie_legal/_trie_advance). Without this, the model could end a word
   * mid-trie ("CA ") -- i.e. emit a fragment that isn't a corpus word,
   * defeating the whole guard. top_k=1 forces the deterministic greedy
   * branch (k==1), so each case has exactly one correct answer.
   *
   * charset adds id6 = ' ' (a non-word byte); TRIE_NODES is UNCHANGED --
   * a boundary byte never lives in the trie, so the same 9-node array
   * from fixtures B/C applies. */
  {
    static const uint8_t CHARSET_SP[] = {0, 'C', 'A', 'T', 'N', 'I', ' '};
    const uint32_t VSP = 7;
    const uint32_t ID_SPACE = 6;
    ngpt_model m; memset(&m, 0, sizeof(m));
    m.model_type = NGPT_MODEL_GRU;
    m.gru.k_out = 7;
    m.gru.charset = CHARSET_SP;
    m.gru.trie_nodes = &TRIE_NODES[0][0];
    m.gru.trie_count = 9;

    /* D1: mid-word ("CA", node 2, NOT end-of-word) -- a SPACE argmax is
     * illegal; the guard must exclude it and fall back to the only legal
     * continuation 'T' (completing CAT), advancing to the CAT end-node. */
    {
      int32_t lg[7]; peak_logits(lg, VSP, ID_SPACE);
      ngpt_ctx ctx; makeCtx(&ctx, &m, /*top_k=*/1, 256, 0, /*trie_on=*/1, /*node=*/2, 1);
      uint32_t tok = ngpt_sample_pick(&ctx, lg, VSP);
      CHECK_EQ_INT((int)tok, ID_T);
      CHECK_EQ_INT((int)ctx.trie_node, 3);
    }
    /* D2: end-of-word ("CAT", node 3, end) -- a SPACE argmax IS legal;
     * the guard keeps it and RESETS the walk to the root for the next
     * word. This is the case fixtures A-C could never exercise. */
    {
      int32_t lg[7]; peak_logits(lg, VSP, ID_SPACE);
      ngpt_ctx ctx; makeCtx(&ctx, &m, 1, 256, 0, 1, /*node=*/3, 1);
      uint32_t tok = ngpt_sample_pick(&ctx, lg, VSP);
      CHECK_EQ_INT((int)tok, (int)ID_SPACE);
      CHECK_EQ_INT((int)ctx.trie_node, 0); /* boundary resets to root */
    }
    /* D3: mid-word ("CA", node 2) -- EOS argmax is illegal too (no ending
     * a word early); the guard falls back to 'T'. */
    {
      int32_t lg[7]; peak_logits(lg, VSP, ID_EOS);
      ngpt_ctx ctx; makeCtx(&ctx, &m, 1, 256, 0, 1, /*node=*/2, 1);
      uint32_t tok = ngpt_sample_pick(&ctx, lg, VSP);
      CHECK_EQ_INT((int)tok, ID_T);
      CHECK_EQ_INT((int)ctx.trie_node, 3);
    }
    /* D4: root (node 0) -- boundary bytes ARE legal; a SPACE argmax is
     * kept and the walk stays at the root. */
    {
      int32_t lg[7]; peak_logits(lg, VSP, ID_SPACE);
      ngpt_ctx ctx; makeCtx(&ctx, &m, 1, 256, 0, 1, /*node=*/0, 1);
      uint32_t tok = ngpt_sample_pick(&ctx, lg, VSP);
      CHECK_EQ_INT((int)tok, (int)ID_SPACE);
      CHECK_EQ_INT((int)ctx.trie_node, 0);
    }
    /* D5: multi-word integrity -- generate across many steps with
     * adversarial pseudo-random logits over a vocab that DOES contain a
     * boundary byte. Every space-separated token must be a real corpus
     * word: the guard must never emit a fragment ("CA", "A") followed by
     * a space. Strictly stronger than fixture C, which (no space in its
     * vocab) could only ever produce a single word per run. */
    {
      const char *words[] = {"CAT", "CAN", "AT", "IT"};
      for (uint32_t seed = 1; seed <= 40; ++seed) {
        ngpt_ctx ctx; makeCtx(&ctx, &m, VSP, 200, 0, 1, 0, seed);
        uint32_t rngData = seed * 2654435761u + 1;
        char out[64]; int outLen = 0;
        for (int step = 0; step < 24; ++step) {
          int32_t lg[7];
          for (uint32_t v = 0; v < VSP; ++v)
            lg[v] = (int32_t)(int16_t)(test_rng(&rngData) & 0xFFFF) << 8;
          uint32_t tok = ngpt_sample_pick(&ctx, lg, VSP);
          if (tok == ID_EOS) break;
          CHECK(outLen < (int)sizeof(out) - 1);
          out[outLen++] = (char)CHARSET_SP[tok];
        }
        out[outLen] = '\0';
        /* split on spaces; each non-empty token must be a corpus word */
        int i = 0;
        while (i < outLen) {
          if (out[i] == ' ') { i++; continue; }
          char tokbuf[8]; int tl = 0;
          while (i < outLen && out[i] != ' ') { CHECK(tl < 7); tokbuf[tl++] = out[i++]; }
          tokbuf[tl] = '\0';
          int found = 0;
          for (unsigned w = 0; w < sizeof(words) / sizeof(words[0]); ++w)
            if (strcmp(tokbuf, words[w]) == 0) { found = 1; break; }
          CHECK(found); /* else: emitted a fragment that isn't a corpus word */
        }
      }
    }
  }

  return test_summary("test_trie_sampler");
}
