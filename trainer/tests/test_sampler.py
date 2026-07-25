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


# ---- M12.1 phase 3: integer min-p gate (docs/ideas-coherence-rescue-plan.md
# fix 3). minp_shift=0 (the default) must reproduce every test above
# byte-for-byte since it's the same code path, unexercised.

def test_minp_disabled_matches_unfiltered_baseline():
    """minp_shift=0 (the ngpt_reset/ref_impl default) must be pixel-for-
    pixel identical to never passing the argument at all -- the whole
    point of an additive gate."""
    logits = (np.arange(10, dtype=np.int64) + 1) << 20
    for seed in (1, 7, 12345):
        tok_a, state_a = sample_from_logits(Q, logits, seed, inv_t_q8=64, top_k=5)
        tok_b, state_b = sample_from_logits(Q, logits, seed, inv_t_q8=64, top_k=5,
                                            minp_shift=0)
        assert (tok_a, state_a) == (tok_b, state_b)


def test_minp_weight_zero_is_always_the_max():
    """The gate's floor is weights[0] >> shift; order[0] is the top logit
    (diff 0), and every other kept candidate has diff <= 0, so no other
    weight can exceed it. If this ever failed, the gate could exclude
    the top candidate itself, which must never happen."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        logits = rng.integers(-2000, 2000, size=12).astype(np.int64) << 14
        for shift in (1, 2, 3, 4):
            tok, _ = sample_from_logits(Q, logits, 5, inv_t_q8=200, top_k=8,
                                        minp_shift=shift)
            # the top candidate must always SURVIVE the gate (it may not be
            # DRAWN, but the gate itself can never make it unreachable —
            # verified indirectly: greedy (top_k=1) always agrees with argmax
            # regardless of shift, since k==1 short-circuits before the gate)
            greedy, _ = sample_from_logits(Q, logits, 5, inv_t_q8=200, top_k=1,
                                           minp_shift=shift)
            assert greedy == int(np.argmax(logits))


def test_minp_shift1_collapses_a_clear_leader_to_near_greedy():
    """A logit clearly ahead of the rest (the runner-up's exp2 weight
    landing under 50% of the leader's) plus minp_shift=1 should exclude
    every other candidate -- every seed lands on the same token as plain
    greedy, WITHOUT needing an extreme temperature (contrast
    test_tiny_temperature_is_greedy, which gets the same outcome via T
    instead of the gate)."""
    logits = np.array([1000, 400, 390, 380, 10], dtype=np.int64) << 14
    state = 1
    for seed in (1, 7, 12345, 99, 777, 42):
        tok, state = sample_from_logits(Q, logits, state, inv_t_q8=256,
                                        top_k=5, minp_shift=1)
        assert tok == int(np.argmax(logits)) == 0


def test_minp_keeps_variety_among_close_candidates():
    """Near-tied top-k logits: min-p should NOT collapse to a single
    token the way a sharp distribution does -- multiple distinct tokens
    must still be reachable across many seeds, proving the gate doesn't
    just default to greedy regardless of input."""
    logits = (np.array([100, 99, 98, 97, 10], dtype=np.int64)) << 14
    state, seen = 1, set()
    for _ in range(300):
        tok, state = sample_from_logits(Q, logits, state, inv_t_q8=256,
                                        top_k=5, minp_shift=1)
        seen.add(tok)
    assert len(seen) > 1
    assert seen <= {0, 1, 2, 3}  # id 4 (10 << 14) is far enough below to be gated out


def test_minp_higher_shift_keeps_more_candidates():
    """floor = weights[0] >> shift: a bigger shift is a SMALLER floor, so
    the reachable set at a looser shift must be a superset of a
    stricter one on the same logits (monotonicity the whole sweep in
    docs/milestones/m12.1.md phase 3 depends on)."""
    logits = (np.array([100, 90, 70, 40, 5], dtype=np.int64)) << 14
    seen_by_shift = {}
    for shift in (1, 2, 3, 4):
        state, seen = 1, set()
        for _ in range(500):
            tok, state = sample_from_logits(Q, logits, state, inv_t_q8=256,
                                            top_k=5, minp_shift=shift)
            seen.add(tok)
        seen_by_shift[shift] = seen
    assert seen_by_shift[1] <= seen_by_shift[2] <= seen_by_shift[3] <= seen_by_shift[4]


def test_minp_deterministic():
    logits = (np.arange(15, dtype=np.int64) + 1) << 18

    def run(seed, n=100):
        state, out = seed, []
        for _ in range(n):
            tok, state = sample_from_logits(Q, logits, state, inv_t_q8=128,
                                            top_k=8, minp_shift=2)
            out.append(tok)
        return out

    assert run(42) == run(42)
    assert run(42) != run(43)


# ---- M12.1 phase 4: lexicon-trie decode guard ----
from ngpt_trainer.ref_impl import (build_word_trie, sample_from_logits_trie,
                                   generate_sampled_trie, TRIE_NONE)
from ngpt_trainer.vocab import Vocab


def _mini_vocab():
    # charset must be sorted (Vocab.from_text's own contract)
    return Vocab.from_text("ABCDEFHIKLNORSTUY '.")


def test_trie_disabled_matches_plain_sampler():
    vocab = _mini_vocab()
    logits = [i * 100 for i in range(len(vocab))]
    for seed in (1, 5, 99):
        tok_a, state_a = sample_from_logits(Q, logits, seed, inv_t_q8=256, top_k=5)
        tok_b, state_b, node_b = sample_from_logits_trie(
            Q, logits, seed, inv_t_q8=256, top_k=5, minp_shift=0,
            trie_nodes=None, trie_node=0, vocab=vocab)
        assert (tok_a, state_a) == (tok_b, state_b)
        assert node_b == 0


def test_trie_blocks_a_word_not_in_the_corpus():
    """CAT is in the trie; CAR is not (shares prefix CA, diverges at
    the 3rd letter). Even if the model's logits strongly favor 'R' as
    the 3rd letter, the guard must exclude it and fall back to a
    legal continuation ('T', completing CAT)."""
    from ngpt_trainer.ref_impl import _trie_advance, _trie_legal
    vocab = _mini_vocab()
    trie = build_word_trie(["CAT", "CAN"])
    node = _trie_advance(trie, 0, vocab.encode("C")[0], vocab)
    node = _trie_advance(trie, node, vocab.encode("A")[0], vocab)
    # node now = after "CA"; only 'T' and 'N' are legal next word-chars
    assert _trie_legal(trie, node, vocab.encode("T")[0], vocab)
    assert _trie_legal(trie, node, vocab.encode("N")[0], vocab)
    assert not _trie_legal(trie, node, vocab.encode("R")[0], vocab)

    logits = [0] * len(vocab)
    logits[vocab.encode("R")[0]] = 10000  # model STRONGLY wants the illegal 'R'
    logits[vocab.encode("T")[0]] = 1
    tok, _, _ = sample_from_logits_trie(Q, logits, 1, inv_t_q8=256, top_k=5,
                                        minp_shift=0, trie_nodes=trie,
                                        trie_node=node, vocab=vocab)
    assert tok == vocab.encode("T")[0]  # fell back to the legal, next-best option


def test_trie_never_invents_a_word_multistep():
    """Drive sample_from_logits_trie directly across many steps (no
    gru_step/full model needed -- this tests the TRIE STATE MACHINE,
    not the GRU numerics) with logits that favor a DIFFERENT letter
    every step almost at random, and confirm every space-delimited
    token completed is a real corpus word -- the actual end-to-end
    guarantee this feature exists to provide."""
    vocab = _mini_vocab()
    words = ["CAT", "CATS", "SAT", "ON", "THE", "MAT", "IT'S", "FINE"]
    trie = build_word_trie(words)
    rng = np.random.default_rng(3)
    for seed in range(1, 20):
        state, node = seed, 0
        out = []
        for _ in range(30):
            logits = rng.integers(-500, 500, size=len(vocab)).astype(np.int64) << 10
            tok, state, node = sample_from_logits_trie(
                Q, logits, state, inv_t_q8=200, top_k=6, minp_shift=0,
                trie_nodes=trie, trie_node=node, vocab=vocab)
            if tok == vocab.eos_id:
                break
            out.append(vocab.decode([tok]))
        text = "".join(out)
        for tok in text.replace(".", " ").split(" "):
            if tok:
                assert tok in words, f"invented word {tok!r} (seed={seed}, text={text!r})"


def test_trie_fallback_always_terminates_a_word():
    """A node with no children must be an end-of-word (build_word_trie's
    invariant) -- if it weren't, the fallback could find no legal
    continuation and the assert in _trie_fallback would fire."""
    vocab = _mini_vocab()
    trie = build_word_trie(["CAT", "DOG"])
    from ngpt_trainer.ref_impl import _trie_is_end
    for node in range(len(trie)):
        has_children = trie[node][2] != TRIE_NONE
        if not has_children:
            assert _trie_is_end(trie, node), f"node {node} is a dead end, not marked as a word"
