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


def gru_h_update(q, h: np.ndarray, x_id: int) -> np.ndarray:
    """The gate math only: consume token x_id, return h_next. Priming
    uses this without computing logits (mirrors the C split exactly)."""
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
    return sat16(rshift_round((Q14_ONE - z) * n, 14) + rshift_round(z * h, 14))


def gru_step(q, h: np.ndarray, x_id: int):
    """One full step: h-update then logits. Returns (h_next, logits) —
    logits int64 [V], argmax-ready."""
    h_next = gru_h_update(q, h, x_id)
    acc_o = q.W_out.astype(np.int64) @ h_next + q.b_out.astype(np.int64)
    return h_next, acc_o


def prime(q, vocab, prompt: str):
    """Consume the prompt without emitting: h-updates only, no logits.
    Returns (h, cur) ready for the generation loop — cur is the LAST
    prompt char, so the first gru_step consumes it and its argmax is the
    first generated character. Unknown chars are skipped (m3.md rule).
    The C implementation must mirror this exactly."""
    h = np.zeros(q.H, dtype=np.int64)
    cur = vocab.eos_id
    for ch in prompt:
        try:
            nxt = vocab.encode(ch)[0]
        except KeyError:
            continue
        h = gru_h_update(q, h, cur)
        cur = nxt
    return h, cur


def generate(q, vocab, prompt: str = "", max_len: int = 256) -> str:
    """Greedy decode; empty prompt reproduces M2 behavior (h = 0, EOS as
    first input). np.argmax breaks ties toward the lowest index — the C
    loop must do the same."""
    h, x = prime(q, vocab, prompt)
    out = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        x = int(np.argmax(logits))
        if x == vocab.eos_id:
            break
        out.append(vocab.decode([x]))
    return "".join(out)


def xorshift32(state: int) -> int:
    """Marsaglia xorshift32 (13/17/5), the C side verbatim. State must
    never be 0 (the sequence's only fixed point); callers remap seed 0
    to 1 before the first call — as must the C side."""
    state &= 0xFFFFFFFF
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state


def sample_from_logits(q, logits, rng_state: int, inv_t_q8: int, top_k: int):
    """One temperature/top-k draw, integer-only (m4.md design). Returns
    (token_id, new_rng_state). inv_t_q8 = round(256/T); k=1 reproduces
    argmax (ties toward the lowest id) regardless of the RNG draw.

    Steps, each mapping 1:1 onto C:
      1. top-k logit indices, ties toward lowest id
      2. temperature: rshift_round(logit * inv_t_q8, 8), still in the
         logit scale 2^(k_out+14)
      3. weights: exp2 LUT on (scaled - scaled_max) rescaled to Q10 by
         rshift_round(diff, k_out + 4)
      4. draw = xorshift32() % total_weight; first index whose
         cumulative weight exceeds draw wins
    """
    from ngpt_trainer.sampler_lut import lut_exp2_lookup
    V = len(logits)
    k = min(top_k, V)
    s = q.k_out + 4  # 2^(k_out+14) -> Q10 for the LUT
    assert s >= 1, "sampler assumes k_out >= -3"
    order = sorted(range(V), key=lambda i: (-int(logits[i]), i))[:k]
    scaled = [rshift_round(int(logits[i]) * inv_t_q8, 8) for i in order]
    top = scaled[0]  # order[0] is the max logit, so scaled[0] is max
    weights = [lut_exp2_lookup(rshift_round(v - top, s)) for v in scaled]
    total = sum(weights)
    rng_state = xorshift32(rng_state)
    if k == 1 or total == 0:  # degenerate: greedy (RNG still advances)
        return order[0], rng_state
    draw = rng_state % total
    cum = 0
    for i, w in zip(order, weights):
        cum += w
        if cum > draw:
            return i, rng_state
    return order[-1], rng_state  # unreachable; belt and suspenders


def generate_sampled(q, vocab, prompt: str = "", seed: int = 1,
                     inv_t_q8: int = 256, top_k: int = 8,
                     max_len: int = 256) -> str:
    """Sampled decode: same priming as generate(), but each step draws
    via sample_from_logits. Deterministic given the seed."""
    h, x = prime(q, vocab, prompt)
    state = seed if seed != 0 else 1
    out = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        x, state = sample_from_logits(q, logits, state, inv_t_q8, top_k)
        if x == vocab.eos_id:
            break
        out.append(vocab.decode([x]))
    return "".join(out)


def trace_sampled(q, vocab, prompt: str, seed: int, inv_t_q8: int,
                  top_k: int, max_len: int = 256):
    """Per-step goldens for the C sampler tests: like trace(), but the
    next token comes from sample_from_logits — records (input_id,
    h_after int16[H], chosen_id), generation steps only."""
    h, x = prime(q, vocab, prompt)
    state = seed if seed != 0 else 1
    steps = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        nxt, state = sample_from_logits(q, logits, state, inv_t_q8, top_k)
        steps.append((x, h.astype(np.int16).copy(), nxt))
        x = nxt
        if x == vocab.eos_id:
            break
    return steps


def trace(q, vocab, prompt: str = "", max_len: int = 256):
    """Per-step goldens for the C tests: list of (input_id, h_after int16[H],
    argmax_id), generation steps only (priming is replayed by ngpt_reset
    on the C side). The blob exporter serializes this."""
    h, x = prime(q, vocab, prompt)
    steps = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        nxt = int(np.argmax(logits))
        steps.append((x, h.astype(np.int16).copy(), nxt))
        x = nxt
        if x == vocab.eos_id:
            break
    return steps
