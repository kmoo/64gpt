/* NGPT blob parser tests on hand-crafted byte arrays.
 * These prove the parser is byte-oriented (endian-independent) and
 * rejects malformed blobs with the right error codes. */
#include "ngpt.h"
#include "test_util.h"

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
  blob[5] = 0x02; /* version 2 does not exist */
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
  return test_summary("test_blob_parser");
}
