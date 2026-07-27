"""Float GRU → integer artifacts (int8 weights, int32 biases, int16 LUTs).

All scales are powers of two stored as shift amounts, so the console
rescales with arithmetic shifts only. W_ih and W_hh share one shift so
their accumulators add without rescaling. Spec: docs/milestones/m2.md.
"""
from dataclasses import dataclass

import numpy as np

# k is capped so the bias scale 2^(k+14) stays well inside int32 and the
# C side's rescale shift (k+3) stays sane; floored at 0 so a shift is
# always a right-shift on the console.
K_MAX = 16


def pow2_shift(t: np.ndarray) -> int:
    """Largest k in [0, K_MAX] with round(max|t| * 2^k) <= 127."""
    m = float(np.abs(t).max())
    if m == 0.0:
        return 14
    k = 0
    while k < K_MAX and round(m * 2 ** (k + 1)) <= 127:
        k += 1
    return k


def make_lut(fn) -> np.ndarray:
    """256-entry int16 Q14 table of fn over [-8, 8) in 1/16 steps."""
    lut = np.zeros(256, dtype=np.int16)
    for i in range(256):
        lut[i] = round(float(fn((i - 128) / 16)) * 16384)
    return lut


@dataclass
class QuantizedGRU:
    H: int
    V: int
    k_w: int
    W_ih: np.ndarray      # int8  [3H, V]
    W_hh: np.ndarray      # int8  [3H, H]
    b_ih: np.ndarray      # int32 [3H], scale 2^(k_w+14)
    b_hh: np.ndarray      # int32 [3H], scale 2^(k_w+14)
    k_out: int
    W_out: np.ndarray     # int8  [V, H]
    b_out: np.ndarray     # int32 [V], scale 2^(k_out+14)
    lut_sigmoid: np.ndarray  # int16 [256]
    lut_tanh: np.ndarray     # int16 [256]


def quantize(model) -> QuantizedGRU:
    W_ih = model.gru.weight_ih_l0.detach().numpy()
    W_hh = model.gru.weight_hh_l0.detach().numpy()
    b_ih = model.gru.bias_ih_l0.detach().numpy()
    b_hh = model.gru.bias_hh_l0.detach().numpy()
    W_out = model.head.weight.detach().numpy()
    b_out = model.head.bias.detach().numpy()

    k_w = min(pow2_shift(W_ih), pow2_shift(W_hh))
    k_out = pow2_shift(W_out)

    return QuantizedGRU(
        H=W_hh.shape[1],
        V=W_ih.shape[1],
        k_w=k_w,
        W_ih=np.round(W_ih * 2**k_w).astype(np.int8),
        W_hh=np.round(W_hh * 2**k_w).astype(np.int8),
        b_ih=np.round(b_ih * 2 ** (k_w + 14)).astype(np.int32),
        b_hh=np.round(b_hh * 2 ** (k_w + 14)).astype(np.int32),
        k_out=k_out,
        W_out=np.round(W_out * 2**k_out).astype(np.int8),
        b_out=np.round(b_out * 2 ** (k_out + 14)).astype(np.int32),
        lut_sigmoid=make_lut(lambda x: 1.0 / (1.0 + np.exp(-x))),
        lut_tanh=make_lut(np.tanh),
    )


@dataclass
class QuantizedGRUFiLM(QuantizedGRU):
    """M12.5 (option B, FiLM): adds a per-channel scale/shift derived
    from the attribute one-hot on top of the base GRU quantized
    artifacts. gamma/beta share the SAME int8-weight + Q14-accumulator
    convention as W_ih/W_hh/W_out -- a new application site (see
    ref_impl.film_gamma_beta), not a new number format."""
    k_film: int = 0
    W_film: np.ndarray = None   # int8  [2H, n_attr]
    b_film: np.ndarray = None   # int32 [2H], scale 2^(k_film+14)


def quantize_film(model) -> QuantizedGRUFiLM:
    """quantize()'s shape for CharGRUFiLM (model.cell is an nn.GRUCell,
    so its weights are named weight_ih/weight_hh/bias_ih/bias_hh, no
    _l0 suffix -- otherwise identical to quantize()'s base GRU fields),
    plus film.weight/film.bias quantized the same way as W_out/b_out."""
    W_ih = model.cell.weight_ih.detach().numpy()
    W_hh = model.cell.weight_hh.detach().numpy()
    b_ih = model.cell.bias_ih.detach().numpy()
    b_hh = model.cell.bias_hh.detach().numpy()
    W_out = model.head.weight.detach().numpy()
    b_out = model.head.bias.detach().numpy()
    W_film = model.film.weight.detach().numpy()
    b_film = model.film.bias.detach().numpy()

    k_w = min(pow2_shift(W_ih), pow2_shift(W_hh))
    k_out = pow2_shift(W_out)
    k_film = pow2_shift(W_film)

    return QuantizedGRUFiLM(
        H=W_hh.shape[1],
        V=W_ih.shape[1],
        k_w=k_w,
        W_ih=np.round(W_ih * 2**k_w).astype(np.int8),
        W_hh=np.round(W_hh * 2**k_w).astype(np.int8),
        b_ih=np.round(b_ih * 2 ** (k_w + 14)).astype(np.int32),
        b_hh=np.round(b_hh * 2 ** (k_w + 14)).astype(np.int32),
        k_out=k_out,
        W_out=np.round(W_out * 2**k_out).astype(np.int8),
        b_out=np.round(b_out * 2 ** (k_out + 14)).astype(np.int32),
        lut_sigmoid=make_lut(lambda x: 1.0 / (1.0 + np.exp(-x))),
        lut_tanh=make_lut(np.tanh),
        k_film=k_film,
        W_film=np.round(W_film * 2**k_film).astype(np.int8),
        b_film=np.round(b_film * 2 ** (k_film + 14)).astype(np.int32),
    )
