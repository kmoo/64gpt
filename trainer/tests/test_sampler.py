"""Integer sampler legs (m4.md design): xorshift32 pinned to known
values, exp2 LUT shape, and sample_from_logits behavior — argmax at k=1,
greedy at tiny temperature, top-k masking, determinism. End-to-end
sampled goldens against the trained blob come with the M4 export step."""
from types import SimpleNamespace

import numpy as np
import pytest

from ngpt_trainer.ref_impl import xorshift32, sample_from_logits
from ngpt_trainer.sampler_lut import LUT_EXP2, lut_exp2_lookup

Q = SimpleNamespace(k_out=6)  # sampler only reads k_out


def test_xorshift32_pinned_sequence():
    s, seq = 1, []
    for _ in range(4):
        s = xorshift32(s)
        seq.append(s)
    assert seq == [270369, 67634689, 2647435461, 307599695]
    assert xorshift32(0xDEADBEEF) == 1199382711
    assert xorshift32(1199382711) == 2384302402


def test_lut_exp2_shape():
    assert len(LUT_EXP2) == 256
    assert LUT_EXP2[0] == 0          # floor semantics: deep tail is zero
    assert LUT_EXP2[15] == 0         # ...through -15.0625
    assert LUT_EXP2[16] == 1         # floor(exp2(-15) * 2^15) = 1
    assert LUT_EXP2[255] == 31378    # floor(exp2(-1/16) * 2^15)
    assert all(0 <= v <= 0xFFFF for v in LUT_EXP2)
    assert all(a <= b for a, b in zip(LUT_EXP2, LUT_EXP2[1:]))
    # clamping: anything at/below -16.0 hits entry 0, top of range 255
    assert lut_exp2_lookup(-99999) == LUT_EXP2[0]
    assert lut_exp2_lookup(0) == LUT_EXP2[255]


def test_k1_is_argmax_with_low_id_ties():
    logits = np.array([5, 9, 9, 3], dtype=np.int64) << 20
    for seed in (1, 7, 12345):
        tok, _ = sample_from_logits(Q, logits, seed, inv_t_q8=256, top_k=1)
        assert tok == int(np.argmax(logits)) == 1


def test_tiny_temperature_is_greedy():
    rng = np.random.default_rng(0)
    state = 1
    for _ in range(50):
        logits = rng.integers(-1000, 1000, size=40).astype(np.int64) << 14
        tok, state = sample_from_logits(Q, logits, state,
                                        inv_t_q8=256 * 128, top_k=8)
        assert tok == int(np.argmax(logits))


def test_top_k_masks_everything_else():
    logits = (np.arange(10, dtype=np.int64) + 1) << 20  # 9 is best
    state, seen = 1, set()
    for _ in range(300):
        tok, state = sample_from_logits(Q, logits, state,
                                        inv_t_q8=64, top_k=2)  # hot: T=4
        seen.add(tok)
    assert seen == {8, 9}  # both reachable, nothing outside top-2


def test_deterministic_and_seed_sensitive():
    logits = (np.arange(20, dtype=np.int64) + 1) << 18

    def run(seed, n=100):
        state, out = seed, []
        for _ in range(n):
            tok, state = sample_from_logits(Q, logits, state,
                                            inv_t_q8=128, top_k=8)
            out.append(tok)
        return out

    assert run(42) == run(42)
    assert run(42) != run(43)
