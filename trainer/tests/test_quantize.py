"""Quantizer gate: int8 weights within half-ULP of float, biases exact in
the accumulator scale, LUTs correct at the bin edges. Spec: m2.md."""
import numpy as np
import pytest

from ngpt_trainer.model import overfit
from ngpt_trainer.quantize import quantize
from ngpt_trainer.vocab import Vocab

CORPUS = "HALT! WHO GOES THERE?"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture(scope="module")
def models():
    vocab = Vocab.from_text(CORPUS)
    model = overfit(CORPUS, vocab)
    return model, quantize(model), vocab


def test_dequant_error_within_half_ulp(models):
    model, q, _ = models
    pairs = [
        (model.gru.weight_ih_l0, q.W_ih, q.k_w),
        (model.gru.weight_hh_l0, q.W_hh, q.k_w),
        (model.head.weight, q.W_out, q.k_out),
    ]
    for float_w, q_w, k in pairs:
        err = np.abs(float_w.detach().numpy() - q_w.astype(np.float64) / 2**k)
        assert err.max() <= 0.5 / 2**k


def test_int8_range_and_dtypes(models):
    _, q, _ = models
    for w in (q.W_ih, q.W_hh, q.W_out):
        assert w.dtype == np.int8
        assert np.abs(w.astype(np.int64)).max() <= 127
    for b in (q.b_ih, q.b_hh, q.b_out):
        assert b.dtype == np.int32
    assert q.lut_sigmoid.dtype == np.int16 and q.lut_tanh.dtype == np.int16
    assert q.k_w >= 0 and q.k_out >= 0


def test_lut_edges_monotonicity_and_error(models):
    _, q, _ = models
    for lut, fn in ((q.lut_sigmoid, sigmoid), (q.lut_tanh, np.tanh)):
        assert lut[0] == round(float(fn(-8.0)) * 16384)
        assert lut[255] == round(float(fn(127 / 16)) * 16384)
        assert np.all(np.diff(lut.astype(np.int64)) >= 0)
        xs = (np.arange(256) - 128) / 16
        assert np.abs(lut / 16384 - fn(xs)).max() < 1e-4


def test_biases_exact_in_accumulator_scale(models):
    model, q, _ = models
    assert np.array_equal(
        q.b_ih, np.round(model.gru.bias_ih_l0.detach().numpy() * 2**(q.k_w + 14)).astype(np.int32))
    assert np.array_equal(
        q.b_hh, np.round(model.gru.bias_hh_l0.detach().numpy() * 2**(q.k_w + 14)).astype(np.int32))
    assert np.array_equal(
        q.b_out, np.round(model.head.bias.detach().numpy() * 2**(q.k_out + 14)).astype(np.int32))


def test_shapes(models):
    _, q, vocab = models
    H, V = q.H, len(vocab)
    assert (q.W_ih.shape, q.W_hh.shape) == ((3 * H, V), (3 * H, H))
    assert q.b_ih.shape == q.b_hh.shape == (3 * H,)
    assert (q.W_out.shape, q.b_out.shape) == ((V, H), (V,))
    assert q.lut_sigmoid.shape == q.lut_tanh.shape == (256,)
