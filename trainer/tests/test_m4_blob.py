"""Regression gate on the COMMITTED M4 blob + goldens: the integer
reference must reproduce every seeded sampled golden from the bytes that
ship in the ROM. No training here — fast, and exactly what the console
does at boot."""
import struct
from pathlib import Path

import pytest

from ngpt_trainer.export import parse_blob
from ngpt_trainer.ref_impl import generate_sampled

VECTORS = Path(__file__).resolve().parent.parent.parent / "tests" / "vectors"
BLOB = VECTORS / "m4_gru.bin"
GOLDENS = VECTORS / "m4_goldens.bin"


def parse_goldens(data: bytes):
    seed, inv_t_q8, top_k, count = struct.unpack(">IHHH", data[:10])
    pairs, off = [], 10
    for _ in range(count):
        out = []
        for _ in range(2):
            (n,) = struct.unpack(">H", data[off:off + 2])
            off += 2
            out.append(data[off:off + n].decode("ascii"))
            off += n
        pairs.append(tuple(out))
    assert off == len(data)
    return seed, inv_t_q8, top_k, pairs


@pytest.mark.skipif(not BLOB.exists(), reason="m4 blob not yet generated")
def test_committed_blob_reproduces_sampled_goldens():
    q, vocab = parse_blob(BLOB.read_bytes())
    seed, inv_t_q8, top_k, pairs = parse_goldens(GOLDENS.read_bytes())
    for prompt, response in pairs:
        got = generate_sampled(q, vocab, prompt, seed=seed,
                               inv_t_q8=inv_t_q8, top_k=top_k)
        assert got == response, prompt
