"""The exp2 LUT shared verbatim with core/ngpt_sampler_lut.h.

Fixed math, not learned weights (M4 architecture decision, m4.md): the
same 256 numbers live here and in the generated C header; the make_m4
script emits the header FROM this module, so they cannot drift.

Table: LUT_EXP2[i] = floor(2^(x) * 2^15) with x = i/16 - 16, i.e. the
input domain is [-16, 0) in steps of 1/16 (Q10: index = (clamped_q10 +
16384) >> 7, mirroring the sigmoid/tanh LUT indexing at Q11). Values fit
u16 (max 31379). FLOOR, not round: entries 0..15 are exactly 0, so a
candidate more than ~15 units below the max gets sampling weight zero —
which is what makes tiny-temperature sampling *exactly* greedy instead
of greedy-minus-one-in-31k (found by test_tiny_temperature_is_greedy).

Float math is allowed HERE (table generation runs in the trainer, like
the sigmoid/tanh LUT emission); inference in ref_impl/C touches only the
integer table.
"""

LUT_EXP2 = tuple(
    int(2.0 ** (i / 16.0 - 16.0) * 32768.0) for i in range(256)
)


def lut_exp2_lookup(x_q10: int) -> int:
    """Integer lookup mirroring the C side: clamp Q10 input to
    [-16384, -1] (the domain is negative — inputs are logit diffs from
    the max; 0 belongs to the top bucket), index = (clamped + 16384) >> 6
    (buckets are 1/16 = 64 Q10 units wide)."""
    if x_q10 < -16384:
        x_q10 = -16384
    elif x_q10 > -1:
        x_q10 = -1
    return LUT_EXP2[(x_q10 + 16384) >> 6]
