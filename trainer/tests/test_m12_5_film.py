"""M12.5 option B -- FiLM (feature-wise linear modulation, docs/ideas-
m12.3-conditioning-strategies.md). The core claim under test: gamma/beta
derived from the attribute one-hot and applied to the GRU hidden state
INSIDE the recurrence (h_t = gamma*cell(x_t, h_{t-1}) + beta, every
timestep) lets a tiny GRU condition its OUTPUT on an attribute alone, even
when the visible prompt text is identical -- and that this survives int8
quantization via quantize_film()/ref_impl's FiLM path. Same overfit-gate
methodology as M12.3 (test_m12_3_attr.py): if a tiny synthetic case can't
do this, the real corpus retrain won't either.

M12.3 (input-side concat, redundant with the prefix): 2.33 inv/line.
M12.4 (input-side concat, prefix stripped): 1.44 inv/line, still short of
the <=1.0 gate. M12.5 tests whether hidden-state modulation -- a
mechanism that can't decay across a long unroll the way an additive input
column can -- closes the remaining gap.

HARD PROJECT RULE (CLAUDE.md): every call below passes device="cpu"
explicitly -- never rely on the default, which auto-selects MPS."""
import numpy as np
import torch

from ngpt_trainer.model import (
    CharGRUFiLM,
    one_hot_attr_vec,
    qat_finetune_film,
    train_corpus_conditioned_film,
)
from ngpt_trainer.quantize import quantize_film
from ngpt_trainer.ref_impl import apply_film, film_gamma_beta, generate_sampled_film
from ngpt_trainer.vocab import Vocab

# Same prompt text for every pair -- the ONLY signal distinguishing the
# two groups is the attribute id, not anything visible in the prompt
# characters. n_desc=2 (which response), n_mood=1 (unused, exercises the
# "more than one attribute" plumbing without adding a second axis to test)
# -- identical setup to test_m12_3_attr.py, so any difference in outcome
# is attributable to the conditioning MECHANISM, not the synthetic task.
PROMPT = "HI|"
PAIRS = [(PROMPT, "HELLO")] * 20 + [(PROMPT, "GOODBYE")] * 20
ATTRS = [(0, 0)] * 20 + [(1, 0)] * 20
N_DESC, N_MOOD = 2, 1
VOCAB = Vocab.from_text(PROMPT + "HELLOGOODBYE")


def _attr_cols(desc_id: int, mood_id: int) -> tuple[int, int]:
    return (desc_id, N_DESC + mood_id)


def test_one_hot_attr_vec_sets_desc_and_mood_columns_only():
    t = one_hot_attr_vec(desc_id=1, n_desc=N_DESC, mood_id=0, n_mood=N_MOOD)
    assert t.shape == (1, N_DESC + N_MOOD)
    assert t[0, 1] == 1.0            # desc_id column
    assert t[0, N_DESC + 0] == 1.0   # mood_id column
    assert (t[0] != 0).sum().item() == 2


def test_char_gru_film_zero_init_is_identity():
    # film's all-zero init must make gamma=1+tanh(0)=1, beta=tanh(0)=0
    # for EVERY attribute at construction time -- FiLM's stability trick
    # only works if training starts from a no-op.
    torch.manual_seed(0)
    model = CharGRUFiLM(vocab_size=6, hidden=8, n_attr=N_DESC + N_MOOD)
    for desc_id in range(N_DESC):
        attr = one_hot_attr_vec(desc_id, N_DESC, 0, N_MOOD)
        gamma_beta_raw = model.film(attr)
        gamma = 1.0 + torch.tanh(gamma_beta_raw[0, :8])
        beta = torch.tanh(gamma_beta_raw[0, 8:])
        assert torch.allclose(gamma, torch.ones(8))
        assert torch.allclose(beta, torch.zeros(8))


def test_apply_film_default_identity_reproduces_plain_h():
    # gamma=16384 (Q14 1.0), beta=0 must be a no-op, the FiLM analogue of
    # gru_h_update's attr_cols=() default reproducing pre-M12.3 behavior.
    rng = np.random.default_rng(0)
    h = rng.integers(-1000, 1000, size=8).astype(np.int64)
    gamma = np.full(8, 16384, dtype=np.int64)
    beta = np.zeros(8, dtype=np.int64)
    assert np.array_equal(apply_film(h, gamma, beta), h)


def test_apply_film_nonidentity_changes_h():
    rng = np.random.default_rng(0)
    h = rng.integers(-1000, 1000, size=8).astype(np.int64)
    gamma = np.full(8, 16384, dtype=np.int64)
    beta = np.full(8, 500, dtype=np.int64)
    assert not np.array_equal(apply_film(h, gamma, beta), h)


def test_film_gamma_beta_differs_per_attr_after_overfit():
    model = train_corpus_conditioned_film(
        PAIRS, PAIRS, ATTRS, ATTRS, VOCAB, n_desc=N_DESC, n_mood=N_MOOD,
        hidden=32, seed=0, batch_size=8, max_epochs=150, patience=30,
        device="cpu")
    q = quantize_film(model)
    gamma0, beta0 = film_gamma_beta(q, _attr_cols(0, 0))
    gamma1, beta1 = film_gamma_beta(q, _attr_cols(1, 0))
    assert not (np.array_equal(gamma0, gamma1) and np.array_equal(beta0, beta1))


def test_film_conditioning_reaches_output_after_overfit():
    model = train_corpus_conditioned_film(
        PAIRS, PAIRS, ATTRS, ATTRS, VOCAB, n_desc=N_DESC, n_mood=N_MOOD,
        hidden=32, seed=0, batch_size=8, max_epochs=150, patience=30,
        device="cpu")
    assert isinstance(model, CharGRUFiLM)
    q = quantize_film(model)
    got_hello = generate_sampled_film(q, VOCAB, PROMPT, attr_cols=_attr_cols(0, 0),
                                      top_k=1)
    got_goodbye = generate_sampled_film(q, VOCAB, PROMPT, attr_cols=_attr_cols(1, 0),
                                        top_k=1)
    assert got_hello == "HELLO"
    assert got_goodbye == "GOODBYE"


def test_qat_finetune_film_returns_eval_mode_model_and_preserves_conditioning():
    model = train_corpus_conditioned_film(
        PAIRS, PAIRS, ATTRS, ATTRS, VOCAB, n_desc=N_DESC, n_mood=N_MOOD,
        hidden=32, seed=0, batch_size=8, max_epochs=150, patience=30,
        device="cpu")
    qat_model = qat_finetune_film(
        model, PAIRS, PAIRS, ATTRS, ATTRS, VOCAB, n_desc=N_DESC, n_mood=N_MOOD,
        seed=0, batch_size=8, max_epochs=20, patience=10, device="cpu")
    assert isinstance(qat_model, CharGRUFiLM)
    assert qat_model.training is False
    assert isinstance(qat_model.final_loss, float)
    q = quantize_film(qat_model)
    got_hello = generate_sampled_film(q, VOCAB, PROMPT, attr_cols=_attr_cols(0, 0),
                                      top_k=1)
    got_goodbye = generate_sampled_film(q, VOCAB, PROMPT, attr_cols=_attr_cols(1, 0),
                                        top_k=1)
    assert got_hello == "HELLO"
    assert got_goodbye == "GOODBYE"
