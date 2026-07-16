import struct
import unittest
import numpy as np
from ngpt_trainer.vocab import Vocab
from ngpt_trainer.model import overfit
from ngpt_trainer.quantize import quantize
from ngpt_trainer.export import build_blob, parse_blob, trace_bytes
from ngpt_trainer.ref_impl import generate
from pathlib import Path

CORPUS = 'HALT! WHO GOES THERE?'

class TestExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = Vocab.from_text(CORPUS)
        model = overfit(CORPUS, vocab)
        cls.q = quantize(model)
        cls.vocab = vocab

    def test_round_trip(self):
        q2, v2 = parse_blob(build_blob(self.q, self.vocab))
        self.assertTrue(np.array_equal(self.q.W_ih, q2.W_ih))
        self.assertTrue(np.array_equal(self.q.W_hh, q2.W_hh))
        self.assertTrue(np.array_equal(self.q.b_ih, q2.b_ih))
        self.assertTrue(np.array_equal(self.q.b_hh, q2.b_hh))
        self.assertTrue(np.array_equal(self.q.W_out, q2.W_out))
        self.assertTrue(np.array_equal(self.q.b_out, q2.b_out))
        self.assertTrue(np.array_equal(self.q.lut_sigmoid, q2.lut_sigmoid))
        self.assertTrue(np.array_equal(self.q.lut_tanh, q2.lut_tanh))
        self.assertEqual(self.q.k_w, q2.k_w)
        self.assertEqual(self.q.k_out, q2.k_out)
        self.assertEqual(self.vocab.charset, v2.charset)

    def test_ref_impl_generate(self):
        q2, v2 = parse_blob(build_blob(self.q, self.vocab))
        self.assertEqual(CORPUS, generate(q2, v2))

    def test_header(self):
        blob = build_blob(self.q, self.vocab)
        self.assertEqual(blob[:4], b'NGPT')
        version, model_type, payload_length = struct.unpack('>HHI', blob[4:12])
        self.assertEqual(version, 1)
        self.assertEqual(model_type, 1)
        self.assertEqual(payload_length, len(blob) - 12)

    def test_payload_size(self):
        H, V = self.q.H, self.q.V
        expected_size = 6 + V + 1024 + 3*H*V + 3*H*H + 12*H*2 + V*H + 4*V
        self.assertEqual(len(build_blob(self.q, self.vocab)) - 12, expected_size)

    def test_trace_bytes_length(self):
        steps = [(0, np.zeros(self.q.H, dtype=np.int16), 0) for _ in range(10)]
        trace = trace_bytes(steps, self.q.H)
        self.assertEqual(len(trace), 4 + 10 * (4 + 2 * self.q.H))

if __name__ == '__main__':
    unittest.main()
