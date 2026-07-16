/**
 * M4 temperature/top-k sampler — the C twin of ref_impl.sample_from_logits.
 * Internal to the engine: ngpt_gru_step calls ngpt_sample_pick when the
 * ctx has sampling enabled (ngpt_set_sampler). Design and fixed-point
 * formats: docs/milestones/m4.md.
 */
#pragma once
#include "ngpt.h"

/* Advance ctx->rng one xorshift32 step and pick a token id from the
 * logits (int32, length V, in scale 2^(k_out+14)). k=1 or an all-zero
 * weight row degrades to argmax with ties toward the lowest id. */
uint32_t ngpt_sample_pick(ngpt_ctx *ctx, const int32_t *logits, uint32_t V);
