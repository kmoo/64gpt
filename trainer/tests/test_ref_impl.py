"""THE M2 gate: integer-only inference must reproduce the training line
byte-for-byte. If this passes, the C port has a bit-exact target."""
import numpy as np
import pytest

from ngpt_trainer.model import overfit
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate, trace
from ngpt_trainer.vocab import Vocab

CORPUS = "HALT! WHO GOES THERE?"


@pytest.fixture(scope="module")
def setup():
    vocab = Vocab.from_text(CORPUS)
    q = quantize(overfit(CORPUS, vocab))
    return q, vocab


def test_integer_generation_reproduces_corpus(setup):
    q, vocab = setup
    assert generate(q, vocab) == CORPUS


def test_trace_agrees_with_generate(setup):
    q, vocab = setup
    steps = trace(q, vocab)
    ids = [nxt for (_, _, nxt) in steps]
    assert ids[-1] == vocab.eos_id
    assert vocab.decode(ids[:-1]) == CORPUS
    for _, h, _ in steps:
        assert h.dtype == np.int16


def test_pipeline_deterministic(setup):
    q, vocab = setup
    vocab2 = Vocab.from_text(CORPUS)
    q2 = quantize(overfit(CORPUS, vocab2))
    assert generate(q2, vocab2) == generate(q, vocab)
    assert np.array_equal(q2.W_hh, q.W_hh)
