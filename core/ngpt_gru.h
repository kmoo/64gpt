/**
 * Internal: quantized-GRU payload parsing and single-step inference.
 * Called only from ngpt.cpp's dispatch — not part of the public API.
 * Numerics contract: docs/milestones/m2.md (must match trainer/ref_impl.py
 * bit-for-bit; the trace goldens in tests/vectors/ prove it).
 */
#pragma once
#include "ngpt.h"

/* Validate the type-1 payload and fill m->gru. NGPT_OK / NGPT_ERR_*. */
int ngpt_gru_load(ngpt_model *m);

/* Prime the context on a prompt string (M3 conditioning): h-updates
 * only, no logits, nothing emitted. Unknown chars are skipped. Called
 * by ngpt_reset after zeroing the state. Mirrors ref_impl.prime(). */
void ngpt_gru_prime(ngpt_ctx *ctx, const char *prompt);

/* Advance one GRU step from ctx->h / ctx->cur: updates both, returns the
 * generated character byte, or NGPT_EOS (and sets ctx->finished). */
int ngpt_gru_step(ngpt_ctx *ctx);
