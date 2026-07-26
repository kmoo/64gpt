"""M12.3 option A -- per-step D:/M: attribute conditioning (docs/ideas-
m12.3-conditioning-strategies.md). The core claim under test: widening
the char one-hot with constant per-sequence attribute columns (model.
one_hot_attr) lets a GRU condition its OUTPUT on those columns alone,
even when the visible prompt text is identical -- and that this survives
int8 quantization via the existing quantize()/ref_impl machinery
unchanged. This is the gate to clear BEFORE spending the ~70-minute real
corpus retrain: if a tiny overfit case can't do this, the real one won't
either.

HARD PROJECT RULE (CLAUDE.md): every call below passes device="cpu"
explicitly -- never rely on the default, which auto-selects MPS."""
import numpy as np
import torch

from ngpt_trainer.model import (
    CharGRU,
    one_hot_attr,
    qat_finetune_attr,
    train_corpus_conditioned_attr,
)
from ngpt_trainer.npc_service import parse_prompt_fields
from ngpt_trainer.quantize import quantize
from ngpt_trainer.ref_impl import generate_sampled_attr, gru_h_update, prime_attr
from ngpt_trainer.vocab import Vocab

# Same prompt text for every pair -- the ONLY signal distinguishing the
# two groups is the attribute id, not anything visible in the prompt
# characters. n_desc=2 (which response), n_mood=1 (unused, exercises the
# "more than one attribute" plumbing without adding a second axis to test).
PROMPT = "HI|"
PAIRS = [(PROMPT, "HELLO")] * 20 + [(PROMPT, "GOODBYE")] * 20
ATTRS = [(0, 0)] * 20 + [(1, 0)] * 20
N_DESC, N_MOOD = 2, 1
VOCAB = Vocab.from_text(PROMPT + "HELLOGOODBYE")


def _attr_cols(desc_id: int, mood_id: int) -> tuple[int, int]:
    V = len(VOCAB)
    return (V + desc_id, V + N_DESC + mood_id)


def test_one_hot_attr_sets_char_and_attr_columns():
    V = len(VOCAB)
    ids = VOCAB.encode("HI")
    t = one_hot_attr(ids, V, desc_id=1, n_desc=N_DESC, mood_id=0, n_mood=N_MOOD)
    assert t.shape == (1, 2, V + N_DESC + N_MOOD)
    for pos, i in enumerate(ids):
        assert t[0, pos, i] == 1.0
        assert t[0, pos, V + 1] == 1.0       # desc_id column
        assert t[0, pos, V + N_DESC + 0] == 1.0  # mood_id column
    # exactly 3 bits set per timestep: 1 char + 1 desc + 1 mood
    assert (t[0, 0] != 0).sum().item() == 3
    assert (t[0, 1] != 0).sum().item() == 3


def test_gru_h_update_attr_cols_default_is_additive_opt_in():
    # Empty attr_cols must reproduce the pre-M12.3 call exactly (same
    # convention as sample_from_logits's minp_shift=0).
    rng = np.random.default_rng(0)
    from ngpt_trainer.quantize import QuantizedGRU, make_lut
    H, V = 4, 6
    q = QuantizedGRU(
        H=H, V=V, k_w=6,
        W_ih=rng.integers(-127, 127, size=(3 * H, V), dtype=np.int64).astype(np.int8),
        W_hh=rng.integers(-127, 127, size=(3 * H, H), dtype=np.int64).astype(np.int8),
        b_ih=rng.integers(-1000, 1000, size=3 * H).astype(np.int32),
        b_hh=rng.integers(-1000, 1000, size=3 * H).astype(np.int32),
        k_out=6,
        W_out=rng.integers(-127, 127, size=(V, H), dtype=np.int64).astype(np.int8),
        b_out=rng.integers(-1000, 1000, size=V).astype(np.int32),
        lut_sigmoid=make_lut(lambda x: 1.0 / (1.0 + np.exp(-x))),
        lut_tanh=make_lut(np.tanh),
    )
    h = rng.integers(-1000, 1000, size=H).astype(np.int64)
    assert np.array_equal(gru_h_update(q, h, 2), gru_h_update(q, h, 2, ()))
    # a non-empty attr_cols must actually change the result (the whole
    # point -- conditioning must reach the accumulator).
    assert not np.array_equal(gru_h_update(q, h, 2), gru_h_update(q, h, 2, (3,)))


def test_attr_conditioning_reaches_output_after_overfit():
    model = train_corpus_conditioned_attr(
        PAIRS, PAIRS, ATTRS, ATTRS, VOCAB, n_desc=N_DESC, n_mood=N_MOOD,
        hidden=32, seed=0, batch_size=8, max_epochs=150, patience=30,
        device="cpu")
    assert isinstance(model, CharGRU)
    q = quantize(model)
    got_hello = generate_sampled_attr(q, VOCAB, PROMPT, attr_cols=_attr_cols(0, 0),
                                      top_k=1)
    got_goodbye = generate_sampled_attr(q, VOCAB, PROMPT, attr_cols=_attr_cols(1, 0),
                                        top_k=1)
    assert got_hello == "HELLO"
    assert got_goodbye == "GOODBYE"


def test_qat_finetune_attr_returns_eval_mode_model_and_preserves_conditioning():
    model = train_corpus_conditioned_attr(
        PAIRS, PAIRS, ATTRS, ATTRS, VOCAB, n_desc=N_DESC, n_mood=N_MOOD,
        hidden=32, seed=0, batch_size=8, max_epochs=150, patience=30,
        device="cpu")
    qat_model = qat_finetune_attr(
        model, PAIRS, PAIRS, ATTRS, ATTRS, VOCAB, n_desc=N_DESC, n_mood=N_MOOD,
        seed=0, batch_size=8, max_epochs=20, patience=10, device="cpu")
    assert isinstance(qat_model, CharGRU)
    assert qat_model.training is False
    assert isinstance(qat_model.final_loss, float)
    q = quantize(qat_model)
    got_hello = generate_sampled_attr(q, VOCAB, PROMPT, attr_cols=_attr_cols(0, 0),
                                      top_k=1)
    got_goodbye = generate_sampled_attr(q, VOCAB, PROMPT, attr_cols=_attr_cols(1, 0),
                                        top_k=1)
    assert got_hello == "HELLO"
    assert got_goodbye == "GOODBYE"


def test_parse_prompt_fields_drives_attr_cols_end_to_end():
    # The real usage pattern: pull D:/M: back out of a prompt string
    # (no corpus generator touched) and turn them into attr_cols via a
    # small caller-owned vocab, exactly what m12_3_attr_spike.py will do
    # against the real corpus.
    prompt = "P:girl D:sassy OCC:villager SPECIES:human R:best_friend " \
             "BOND:rival M:cheerful C:greeting AUD:alone EV:none|"
    fields = parse_prompt_fields(prompt)
    desc_vocab = {"sassy": 0, "cold": 1}
    mood_vocab = {"cheerful": 0, "worried": 1}
    desc_id = desc_vocab[fields["D"]]
    mood_id = mood_vocab[fields["M"]]
    assert (desc_id, mood_id) == (0, 0)
