/* NGPT blob parser tests on hand-crafted byte arrays.
 * These prove the parser is byte-oriented (endian-independent) and
 * rejects malformed blobs with the right error codes. */
#include "ngpt.h"
#include "test_util.h"
#include <string.h>

/* A valid v1 canned-text blob, spelled out byte by byte:
 * magic "NGPT", version 1, model type 0 (canned), payload "HI" (2 bytes). */
static const uint8_t VALID_BLOB[] = {
  'N', 'G', 'P', 'T',       /* magic                */
  0x00, 0x01,               /* format version = 1   */
  0x00, 0x00,               /* model type = canned  */
  0x00, 0x00, 0x00, 0x02,   /* payload length = 2   */
  'H', 'I',                 /* payload              */
};

static void test_be_readers(void)
{
  const uint8_t b16[] = { 0x12, 0x34 };
  const uint8_t b32[] = { 0xDE, 0xAD, 0xBE, 0xEF };
  CHECK_EQ_INT(ngpt_read_u16be(b16), 0x1234);
  CHECK_EQ_INT(ngpt_read_u32be(b32), 0xDEADBEEFu);

  /* high-bit bytes must not sign-extend */
  const uint8_t hi16[] = { 0xFF, 0x00 };
  CHECK_EQ_INT(ngpt_read_u16be(hi16), 0xFF00);
}

static void test_valid_blob_parses(void)
{
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, VALID_BLOB, sizeof(VALID_BLOB)), NGPT_OK);
  CHECK_EQ_INT(m.format_version, 1);
  CHECK_EQ_INT(m.model_type, NGPT_MODEL_CANNED);
  CHECK_EQ_INT(m.payload_len, 2);
  CHECK(m.payload == VALID_BLOB + NGPT_HEADER_SIZE);
  CHECK_EQ_INT(m.payload[0], 'H');
  CHECK_EQ_INT(m.payload[1], 'I');
}

static void test_bad_magic(void)
{
  uint8_t blob[sizeof(VALID_BLOB)];
  memcpy(blob, VALID_BLOB, sizeof(VALID_BLOB));
  blob[0] = 'X';
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, sizeof(blob)), NGPT_ERR_MAGIC);
}

static void test_truncated_header(void)
{
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, VALID_BLOB, NGPT_HEADER_SIZE - 1), NGPT_ERR_TRUNCATED);
  CHECK_EQ_INT(ngpt_load(&m, VALID_BLOB, 0), NGPT_ERR_TRUNCATED);
}

static void test_truncated_payload(void)
{
  /* header says 2 payload bytes but only 1 is present */
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, VALID_BLOB, sizeof(VALID_BLOB) - 1), NGPT_ERR_TRUNCATED);
}

static void test_wrong_version(void)
{
  uint8_t blob[sizeof(VALID_BLOB)];
  memcpy(blob, VALID_BLOB, sizeof(VALID_BLOB));
  /* version 3 does not exist -- M12.1 phase 4 legitimately introduced
   * version 2 (GRU payload + trailing word-trie section), so this must
   * probe a version that's actually unsupported. */
  blob[5] = 0x03;
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, sizeof(blob)), NGPT_ERR_VERSION);
}

static void test_unknown_model_type(void)
{
  uint8_t blob[sizeof(VALID_BLOB)];
  memcpy(blob, VALID_BLOB, sizeof(VALID_BLOB));
  blob[7] = 0x7F; /* not a known model type */
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, sizeof(blob)), NGPT_ERR_MODEL_TYPE);
}

static void test_payload_len_overflow(void)
{
  /* payload_len huge: must not wrap around and pass the bounds check */
  uint8_t blob[sizeof(VALID_BLOB)];
  memcpy(blob, VALID_BLOB, sizeof(VALID_BLOB));
  blob[8] = 0xFF; blob[9] = 0xFF; blob[10] = 0xFF; blob[11] = 0xFF;
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, blob, sizeof(blob)), NGPT_ERR_TRUNCATED);
}

/* ---- v2 (trie-carrying) GRU blob parsing --------------------------------
 * Format version 2 (M12.1 phase 4) appends, after the GRU payload, a
 * trailing word-trie section: a u32 node count followed by count*6 node
 * bytes. Before these tests the parser's v2 branch (ngpt_gru_load lines
 * 98-106) had NO positive coverage -- test_wrong_version only proved v3
 * is rejected. We build the SMALLEST valid GRU payload (H=1, V=1) so the
 * assertions are about the trie section, not the weights (all zero; the
 * parser reads their bytes but validates only sizes and charset[0]==0). */
enum { V2_H = 1, V2_V = 1 };
/* Mirrors ngpt_gru_load's `need` formula exactly for H=1, V=1. */
static const uint32_t V2_NEED = 6u + V2_V + 1024u
    + 3u*V2_H*V2_V + 3u*V2_H*V2_H + 12u*V2_H + 12u*V2_H + V2_V*V2_H + 4u*V2_V;

/* Writes a v2 GRU blob carrying `nodes` trie nodes into buf. When
 * payload_len_override >= 0, that value is written into the header's
 * payload-length field instead of the true one (to exercise the size
 * checks). Node bytes get a recognizable 0x10+i pattern for readback.
 * Returns the true total byte count. */
static uint32_t make_v2_gru_blob(uint8_t *buf, uint32_t nodes, long payload_len_override)
{
  uint32_t payload = V2_NEED + 4u + nodes * 6u;
  uint32_t total = NGPT_HEADER_SIZE + payload;
  memset(buf, 0, total);
  buf[0] = 'N'; buf[1] = 'G'; buf[2] = 'P'; buf[3] = 'T';
  buf[4] = 0x00; buf[5] = 0x02;                       /* format version 2  */
  buf[6] = 0x00; buf[7] = (uint8_t)NGPT_MODEL_GRU;     /* model type = GRU  */
  uint32_t pl = payload_len_override >= 0 ? (uint32_t)payload_len_override : payload;
  buf[8]  = (uint8_t)(pl >> 24); buf[9]  = (uint8_t)(pl >> 16);
  buf[10] = (uint8_t)(pl >> 8);  buf[11] = (uint8_t)pl;
  uint8_t *pay = buf + NGPT_HEADER_SIZE;
  pay[1] = V2_H;   /* H = 1 (u16be)  */
  pay[3] = V2_V;   /* V = 1 (u16be)  */
  /* k_w, k_out, LUTs, weights, and charset[0] (the EOS slot) stay 0 */
  uint8_t *ts = pay + V2_NEED;                          /* trie section     */
  ts[0] = (uint8_t)(nodes >> 24); ts[1] = (uint8_t)(nodes >> 16);
  ts[2] = (uint8_t)(nodes >> 8);  ts[3] = (uint8_t)nodes;
  for (uint32_t i = 0; i < nodes * 6u; ++i) ts[4 + i] = (uint8_t)(0x10 + i);
  return total;
}

static void test_v2_need_formula_matches(void)
{
  /* Guards against silent drift between this test and ngpt_gru_load. */
  CHECK_EQ_INT((int)V2_NEED, 1066);
}

static void test_v2_trie_blob_parses(void)
{
  static uint8_t buf[NGPT_HEADER_SIZE + 2048];
  uint32_t total = make_v2_gru_blob(buf, /*nodes=*/2, -1);
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, buf, total), NGPT_OK);
  CHECK_EQ_INT(m.format_version, NGPT_FORMAT_VERSION_TRIE);
  CHECK_EQ_INT(m.model_type, NGPT_MODEL_GRU);
  CHECK_EQ_INT((int)m.gru.trie_count, 2);
  CHECK(m.gru.trie_nodes != 0);
  /* trie_nodes points just PAST the u32 count; node bytes are 0x10+i */
  CHECK_EQ_INT(m.gru.trie_nodes[0], 0x10);   /* first node, first byte  */
  CHECK_EQ_INT(m.gru.trie_nodes[6], 0x16);   /* second node, first byte */
}

static void test_v2_empty_trie_ok(void)
{
  /* A v2 blob with zero trie nodes is still valid (trie_count = 0). */
  static uint8_t buf[NGPT_HEADER_SIZE + 2048];
  uint32_t total = make_v2_gru_blob(buf, /*nodes=*/0, -1);
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, buf, total), NGPT_OK);
  CHECK_EQ_INT((int)m.gru.trie_count, 0);
}

static void test_v2_trie_count_size_mismatch_rejected(void)
{
  /* Physical size is for 2 nodes, but the count field claims 3 -- the
   * exact-size check (payload_len != need + 4 + count*6) must reject it. */
  static uint8_t buf[NGPT_HEADER_SIZE + 2048];
  uint32_t total = make_v2_gru_blob(buf, /*nodes=*/2, -1);
  buf[NGPT_HEADER_SIZE + V2_NEED + 3] = 0x03; /* count 2 -> 3 */
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, buf, total), NGPT_ERR_TRUNCATED);
}

static void test_v2_payload_too_short_for_count_rejected(void)
{
  /* A v2 payload too short to even hold the 4-byte count must be rejected
   * by the v2 minimum-size check (payload_len < need + 4), not read OOB. */
  static uint8_t buf[NGPT_HEADER_SIZE + 2048];
  make_v2_gru_blob(buf, /*nodes=*/0, /*payload_len_override=*/(long)(V2_NEED + 2));
  ngpt_model m;
  CHECK_EQ_INT(ngpt_load(&m, buf, NGPT_HEADER_SIZE + V2_NEED + 2), NGPT_ERR_TRUNCATED);
}

int main(void)
{
  test_be_readers();
  test_valid_blob_parses();
  test_bad_magic();
  test_truncated_header();
  test_truncated_payload();
  test_wrong_version();
  test_unknown_model_type();
  test_payload_len_overflow();
  test_v2_need_formula_matches();
  test_v2_trie_blob_parses();
  test_v2_empty_trie_ok();
  test_v2_trie_count_size_mismatch_rejected();
  test_v2_payload_too_short_for_count_rejected();
  return test_summary("test_blob_parser");
}
