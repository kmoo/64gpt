/* Minimal dependency-free test harness: CHECK macros + failure count.
 * Each test executable returns non-zero on any failure, which CTest
 * reports as a failed test. */
#pragma once
#include <stdio.h>
#include <string.h>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
      ++g_failures; \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
  } while (0)

#define CHECK_EQ_INT(a, b) do { \
    ++g_checks; \
    long long va_ = (long long)(a), vb_ = (long long)(b); \
    if (va_ != vb_) { \
      ++g_failures; \
      fprintf(stderr, "FAIL %s:%d: %s == %s (got %lld, want %lld)\n", \
              __FILE__, __LINE__, #a, #b, va_, vb_); \
    } \
  } while (0)

static int test_summary(const char *name)
{
  if (g_failures) {
    fprintf(stderr, "%s: %d/%d checks FAILED\n", name, g_failures, g_checks);
    return 1;
  }
  printf("%s: %d checks passed\n", name, g_checks);
  return 0;
}
