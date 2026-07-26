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


def gru_h_update(q, h: np.ndarray, x_id: int, attr_cols=()) -> np.ndarray:
    """The gate math only: consume token x_id, return h_next. Priming
    uses this without computing logits (mirrors the C split exactly).

    attr_cols (M12.3, docs/ideas-m12.3-conditioning-strategies.md option
    A): optional extra column indices into q.W_ih to add into acc_i
    alongside x_id -- e.g. (V + desc_id, V + n_desc + mood_id) for the
    D:/M: per-step attribute conditioning. A learned embedding
    concatenated onto the char one-hot at every timestep is
    mathematically identical to appending more one-hot columns that
    share the SAME W_ih matrix (model.one_hot_attr's convention), so
    this is just more of the exact column-lookup trick already used for
    x_id -- no new accumulator shape, no rescale. Empty tuple (the
    default) reproduces every pre-M12.3 call byte-for-byte, same opt-in
    pattern as sample_from_logits's minp_shift."""
    H = q.H
    # Input-side "matvec" is a column lookup: one-hot in Q14 is a single
    # 16384, so acc = W_ih[:, x] << 14, already in scale 2^(k_w+14).
    acc_i = q.W_ih[:, x_id].astype(np.int64) << 14
    for col in attr_cols:
        acc_i = acc_i + (q.W_ih[:, col].astype(np.int64) << 14)
    acc_i = acc_i + q.b_ih.astype(np.int64)
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


def gru_step_attr(q, h: np.ndarray, x_id: int, attr_cols=()):
    """gru_step, threading attr_cols through to gru_h_update (M12.3)."""
    h_next = gru_h_update(q, h, x_id, attr_cols)
    acc_o = q.W_out.astype(np.int64) @ h_next + q.b_out.astype(np.int64)
    return h_next, acc_o


def prime_attr(q, vocab, prompt: str, attr_cols=()):
    """prime, threading attr_cols through to gru_h_update (M12.3). The
    same fixed attr_cols apply for the whole prompt AND the whole
    generation that follows -- an NPC's voice/mood is constant for one
    reply, mirroring model.one_hot_attr's training-time convention."""
    h = np.zeros(q.H, dtype=np.int64)
    cur = vocab.eos_id
    for ch in prompt:
        try:
            nxt = vocab.encode(ch)[0]
        except KeyError:
            continue
        h = gru_h_update(q, h, cur, attr_cols)
        cur = nxt
    return h, cur


def generate_sampled_attr(q, vocab, prompt: str = "", attr_cols=(), seed: int = 1,
                          inv_t_q8: int = 256, top_k: int = 8,
                          max_len: int = 256, minp_shift: int = 0) -> str:
    """generate_sampled, threading attr_cols through priming and every
    generation step (M12.3). Sampling itself (sample_from_logits) is
    unchanged -- attribute conditioning only ever touches the hidden
    state via acc_i, never the decode-time sampler."""
    h, x = prime_attr(q, vocab, prompt, attr_cols)
    state = seed if seed != 0 else 1
    out = []
    for _ in range(max_len):
        h, logits = gru_step_attr(q, h, x, attr_cols)
        x, state = sample_from_logits(q, logits, state, inv_t_q8, top_k, minp_shift)
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


def sample_from_logits(q, logits, rng_state: int, inv_t_q8: int, top_k: int,
                       minp_shift: int = 0):
    """One temperature/top-k draw, integer-only (m4.md design). Returns
    (token_id, new_rng_state). inv_t_q8 = round(256/T); k=1 reproduces
    argmax (ties toward the lowest id) regardless of the RNG draw.

    Steps, each mapping 1:1 onto C:
      1. top-k logit indices, ties toward lowest id
      2. temperature: rshift_round(logit * inv_t_q8, 8), still in the
         logit scale 2^(k_out+14)
      3. weights: exp2 LUT on (scaled - scaled_max) rescaled to Q10 by
         rshift_round(diff, k_out + 4)
      4. M12.1 min-p gate (docs/ideas-coherence-rescue-plan.md fix 3,
         published min-p sampling arXiv 2407.01082 in integer form):
         if minp_shift > 0, drop any candidate whose weight is below
         weights[0] >> minp_shift. weights[0] is ALWAYS the max weight
         -- order[0] is the top logit (diff 0 from itself), and scaling/
         exp2 are both monotonic in the (all <= 0) diffs, so no later
         candidate's weight can exceed it. minp_shift=0 (the default)
         skips this branch entirely, so every milestone before M12.1
         reproduces byte-for-byte -- this is a strictly additive gate.
      5. draw = xorshift32() % total_weight (over the KEPT candidates
         only, when the gate is active); first index whose cumulative
         weight exceeds draw wins
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
    if minp_shift > 0:
        floor = weights[0] >> minp_shift
        kept_total = sum(w for w in weights if w >= floor)
        draw = rng_state % kept_total
        cum = 0
        for i, w in zip(order, weights):
            if w < floor:
                continue
            cum += w
            if cum > draw:
                return i, rng_state
        return order[0], rng_state  # unreachable; belt and suspenders
    draw = rng_state % total
    cum = 0
    for i, w in zip(order, weights):
        cum += w
        if cum > draw:
            return i, rng_state
    return order[-1], rng_state  # unreachable; belt and suspenders


def generate_sampled(q, vocab, prompt: str = "", seed: int = 1,
                     inv_t_q8: int = 256, top_k: int = 8,
                     max_len: int = 256, minp_shift: int = 0) -> str:
    """Sampled decode: same priming as generate(), but each step draws
    via sample_from_logits. Deterministic given the seed."""
    h, x = prime(q, vocab, prompt)
    state = seed if seed != 0 else 1
    out = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        x, state = sample_from_logits(q, logits, state, inv_t_q8, top_k,
                                      minp_shift)
        if x == vocab.eos_id:
            break
        out.append(vocab.decode([x]))
    return "".join(out)


def trace_sampled(q, vocab, prompt: str, seed: int, inv_t_q8: int,
                  top_k: int, max_len: int = 256, minp_shift: int = 0):
    """Per-step goldens for the C sampler tests: like trace(), but the
    next token comes from sample_from_logits — records (input_id,
    h_after int16[H], chosen_id), generation steps only."""
    h, x = prime(q, vocab, prompt)
    state = seed if seed != 0 else 1
    steps = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        nxt, state = sample_from_logits(q, logits, state, inv_t_q8, top_k,
                                        minp_shift)
        steps.append((x, h.astype(np.int16).copy(), nxt))
        x = nxt
        if x == vocab.eos_id:
            break
    return steps


# ---- M12.1 phase 4: lexicon-trie decode guard (docs/ideas-coherence-
# rescue-plan.md fix 4). A compact first-child/next-sibling trie over
# WORD characters only (A-Z and apostrophe); every other vocab byte
# (space, punctuation) plus EOS is a "boundary" that's always legal
# except mid-word, where it's legal only if the current trie node
# completes a real corpus word. This makes the model structurally
# unable to emit a word that isn't one its corpus actually contains --
# it still CHOOSES which word (personality/mood/conditioning still
# come from the logits), just can't misspell it.
TRIE_NONE = 0xFFFF


def _is_word_byte(b) -> bool:
    return b is not None and ((65 <= b <= 90) or b == 39)  # 'A'-'Z' or "'"


def build_word_trie(words) -> list[tuple[int, int, int, int]]:
    """words: iterable of corpus words (uppercase, letters+apostrophe only
    -- the same set make_m12_blob.py's build_corpus_vocab() already
    computes). Returns a flat node list: (char_byte, flags, first_child,
    next_sibling); node 0 is the root. flags bit 0 = end-of-word. This
    exact list is what export.py serializes into the blob and what
    core/ngpt_sample.cpp walks -- one structure, no drift."""
    nodes = [[0, 0, TRIE_NONE, TRIE_NONE]]
    for word in words:
        cur = 0
        for ch in word:
            b = ord(ch)
            if not _is_word_byte(b):
                continue  # corpus_vocab is pre-filtered to [A-Z']+, belt-and-suspenders
            child, prev = nodes[cur][2], TRIE_NONE
            found = TRIE_NONE
            while child != TRIE_NONE:
                if nodes[child][0] == b:
                    found = child
                    break
                prev, child = child, nodes[child][3]
            if found == TRIE_NONE:
                nodes.append([b, 0, TRIE_NONE, TRIE_NONE])
                new_idx = len(nodes) - 1
                if prev == TRIE_NONE:
                    nodes[cur][2] = new_idx
                else:
                    nodes[prev][3] = new_idx
                cur = new_idx
            else:
                cur = found
        nodes[cur][1] |= 1
    return [tuple(n) for n in nodes]


def _trie_child(nodes, node: int, byte: int) -> int:
    c = nodes[node][2]
    while c != TRIE_NONE:
        if nodes[c][0] == byte:
            return c
        c = nodes[c][3]
    return TRIE_NONE


def _trie_is_end(nodes, node: int) -> bool:
    return (nodes[node][1] & 1) != 0


def _trie_legal(nodes, node: int, idx: int, vocab) -> bool:
    byte = None if idx == vocab.eos_id else ord(vocab.decode([idx]))
    if not _is_word_byte(byte):
        return node == 0 or _trie_is_end(nodes, node)
    return _trie_child(nodes, node, byte) != TRIE_NONE


def _trie_advance(nodes, node: int, idx: int, vocab) -> int:
    byte = None if idx == vocab.eos_id else ord(vocab.decode([idx]))
    if not _is_word_byte(byte):
        return 0
    return _trie_child(nodes, node, byte)


def _trie_fallback(nodes, node: int, logits, vocab) -> int:
    """Highest-logit LEGAL id over the FULL vocab (not just top-k), ties
    toward the lowest id (np.argmax's convention). Guaranteed non-empty:
    every trie node reached by inserting a real word has a child, is an
    end, or both (build_word_trie()'s own invariant) -- EOS/boundary
    bytes are always legal at the root, so this can never come up empty
    on a correctly built trie."""
    best, best_v = None, None
    for i in range(len(logits)):
        if not _trie_legal(nodes, node, i, vocab):
            continue
        if best is None or logits[i] > best_v:
            best, best_v = i, logits[i]
    assert best is not None, "trie invariant violated: no legal continuation"
    return best


def sample_from_logits_trie(q, logits, rng_state: int, inv_t_q8: int, top_k: int,
                            minp_shift: int, trie_nodes, trie_node: int, vocab):
    """Trie-guarded draw: identical top-k/temperature/min-p machinery as
    sample_from_logits, PLUS a legality filter over the word trie.
    Returns (token_id, new_rng_state, new_trie_node). trie_nodes=None
    disables the guard, delegating byte-for-byte to sample_from_logits
    (kept as a separate function, not a modification of that one, so
    the 12 already-passing sampler tests have zero exposure to this
    change)."""
    if trie_nodes is None:
        tok, rng_state = sample_from_logits(q, logits, rng_state, inv_t_q8, top_k, minp_shift)
        return tok, rng_state, trie_node

    from ngpt_trainer.sampler_lut import lut_exp2_lookup
    V = len(logits)
    k = min(top_k, V)
    s = q.k_out + 4
    order = sorted(range(V), key=lambda i: (-int(logits[i]), i))[:k]
    scaled = [rshift_round(int(logits[i]) * inv_t_q8, 8) for i in order]
    top = scaled[0]
    weights = [lut_exp2_lookup(rshift_round(v - top, s)) for v in scaled]
    rng_state = xorshift32(rng_state)

    if k == 1 or sum(weights) == 0:
        tok = order[0] if _trie_legal(trie_nodes, trie_node, order[0], vocab) \
            else _trie_fallback(trie_nodes, trie_node, logits, vocab)
        return tok, rng_state, _trie_advance(trie_nodes, trie_node, tok, vocab)

    floor = (weights[0] >> minp_shift) if minp_shift > 0 else 0
    kept = [(i, w) for i, w in zip(order, weights)
            if w >= floor and _trie_legal(trie_nodes, trie_node, i, vocab)]
    if not kept:
        tok = _trie_fallback(trie_nodes, trie_node, logits, vocab)
    else:
        kept_total = sum(w for _, w in kept)
        draw = rng_state % kept_total
        cum, tok = 0, kept[-1][0]
        for i, w in kept:
            cum += w
            if cum > draw:
                tok = i
                break
    return tok, rng_state, _trie_advance(trie_nodes, trie_node, tok, vocab)


def generate_sampled_trie(q, vocab, prompt: str = "", seed: int = 1,
                          inv_t_q8: int = 256, top_k: int = 8, max_len: int = 256,
                          minp_shift: int = 0, trie_nodes=None) -> str:
    """generate_sampled, trie-guarded: trie_node persists across the
    whole generation (resets to 0/root at the start, same as h/EOS).
    trie_nodes=None reproduces generate_sampled byte-for-byte."""
    h, x = prime(q, vocab, prompt)
    state = seed if seed != 0 else 1
    node = 0
    out = []
    for _ in range(max_len):
        h, logits = gru_step(q, h, x)
        x, state, node = sample_from_logits_trie(q, logits, state, inv_t_q8, top_k,
                                                 minp_shift, trie_nodes, node, vocab)
        if x == vocab.eos_id:
            break
        out.append(vocab.decode([x]))
    return "".join(out)


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
