"""Integer-only reference GRU inference — the bit-exactness contract.

Every operation here maps 1:1 onto a C operation in core/ (same widths,
same shifts, same rounding, same saturation). If this file reproduces the
training corpus and the C port matches it bit-for-bit, then host == Ares
== silicon. NO floats may appear anywhere in this module.

Number formats (see docs/milestones/m2.md):
  h, gate outputs      int16 Q14      (1.0 == 16384)
  weights              int8, scale 2^k_w (shared) / 2^k_out (head)
  biases               int32, in the accumulator scale 2^(k+14)
  accumulators         int64 in numpy (int32 suffices in C at these dims;
                       numpy just avoids surprise wraparound in tests)
  LUT input            int16 Q11, clamped to [-16384, 16383]
"""
import numpy as np

Q14_ONE = 1 << 14


def rshift_round(x, s: int):
    """Arithmetic right shift with round-half-up bias: (x + 2^(s-1)) >> s.
    C++20 defines >> on negative signed integers as arithmetic; numpy
    matches. Works elementwise on arrays and on scalars."""
    return (x + (1 << (s - 1))) >> s


def sat16(x):
    """Saturate to int16 range (the C side clamps, never wraps)."""
    return np.clip(x, -32768, 32767)


def lut_lookup(lut: np.ndarray, x_q11: np.ndarray) -> np.ndarray:
    """256-entry LUT over [-8, 8) in Q11: index = (clamped + 16384) >> 7."""
    clamped = np.clip(x_q11, -16384, 16383)
    return lut[(clamped + 16384) >> 7].astype(np.int64)


def gru_step(q, h: np.ndarray, x_id: int):
    """One GRU step. h is int16-valued Q14 [H] (carried as int64), x_id a
    token id. Returns (h_next, logits) — logits int64 [V], argmax-ready."""
    H = q.H
    # Input-side "matvec" is a column lookup: one-hot in Q14 is a single
    # 16384, so acc = W_ih[:, x] << 14, already in scale 2^(k_w+14).
    acc_i = (q.W_ih[:, x_id].astype(np.int64) << 14) + q.b_ih.astype(np.int64)
    acc_h = q.W_hh.astype(np.int64) @ h + q.b_hh.astype(np.int64)

    s = q.k_w + 3  # rescale 2^(k_w+14) -> Q11 for the LUTs
    r = lut_lookup(q.lut_sigmoid, rshift_round(acc_i[:H] + acc_h[:H], s))
    z = lut_lookup(q.lut_sigmoid, rshift_round(acc_i[H:2*H] + acc_h[H:2*H], s))

    # n-gate: r gates only the hidden-side accumulator (PyTorch convention).
    # r (Q14) x acc (2^(k_w+14)) -> 2^(k_w+28); shift 14 returns to 2^(k_w+14).
    n_acc = acc_i[2*H:] + rshift_round(r * acc_h[2*H:], 14)
    n = lut_lookup(q.lut_tanh, rshift_round(n_acc, s))

    # h' = (1-z)*n + z*h, all Q14: products are Q28, shift 14 back, saturate.
    h_next = sat16(rshift_round((Q14_ONE - z) * n, 14) + rshift_round(z * h, 14))

    acc_o = q.W_out.astype(np.int64) @ h_next + q.b_out.astype(np.int64)
    return h_next, acc_o


def generate(q, vocab, max_len: int = 256) -> str:
    """Greedy decode from h = 0 and EOS as the first input. np.argmax
    breaks ties toward the lowest index — the C loop must do the same."""
    h = np.zeros(q.H, dtype=np.int64)
    x = vocab.eos_id
    out = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        x = int(np.argmax(logits))
        if x == vocab.eos_id:
            break
        out.append(vocab.decode([x]))
    return "".join(out)


def trace(q, vocab, max_len: int = 256):
    """Per-step goldens for the C tests: list of (input_id, h_after int16[H],
    argmax_id). The blob exporter serializes this."""
    h = np.zeros(q.H, dtype=np.int64)
    x = vocab.eos_id
    steps = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        nxt = int(np.argmax(logits))
        steps.append((x, h.astype(np.int16).copy(), nxt))
        x = nxt
        if x == vocab.eos_id:
            break
    return steps
