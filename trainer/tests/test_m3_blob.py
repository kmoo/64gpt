"""Regression gate on the COMMITTED M3 blob: the integer reference must
reproduce every corpus response from its prompt using only the bytes that
ship in the ROM. No training here — fast, and exactly what the console does."""
from pathlib import Path

import pytest

from ngpt_trainer import corpus
from ngpt_trainer.export import parse_blob
from ngpt_trainer.ref_impl import generate

BLOB = Path(__file__).resolve().parent.parent.parent / "tests" / "vectors" / "m3_gru.bin"


@pytest.mark.skipif(not BLOB.exists(), reason="m3 blob not yet generated")
def test_committed_blob_reproduces_all_pairs():
    q, vocab = parse_blob(BLOB.read_bytes())
    for prompt, response in corpus.pairs():
        assert generate(q, vocab, prompt) == response
