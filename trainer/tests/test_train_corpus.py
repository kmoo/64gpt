"""Test coverage for train_corpus / train_corpus_conditioned (model.py) --
the two real training functions make_m4_blob.py / make_m7_blob.py /
make_m8_blob.py all actually call. Only the toy overfit()/overfit_corpus()
variants had coverage before (test_overfit.py, test_conditioning.py).

HARD PROJECT RULE (CLAUDE.md): every call below passes device="cpu"
explicitly -- never rely on the default, which auto-selects MPS."""
import torch

from ngpt_trainer.model import CharGRU, train_corpus, train_corpus_conditioned
from ngpt_trainer.vocab import Vocab

# train_corpus() carves its OWN internal validation split (every 10th
# pair, encoded[9::10] -- see model.py) rather than taking one as an
# argument, so it needs >=10 pairs or that split is empty and val_loss()
# divides by zero. 10 pairs keeps this fast while giving it exactly one.
FLAT_PAIRS = [(f"{chr(ord('A') + i)}|", "HI" if i % 2 == 0 else "BYE")
              for i in range(10)]

TRAIN_PAIRS = [
    ("A|", "HI"),
    ("B|", "BYE"),
    ("C|", "OK"),
]
VAL_PAIRS = [
    ("A|", "HI"),
    ("B|", "BYE"),
]
CORPUS_TEXT = "".join(p + r for p, r in
                      FLAT_PAIRS + TRAIN_PAIRS + VAL_PAIRS)


def _vocab():
    return Vocab.from_text(CORPUS_TEXT)


def test_train_corpus_returns_eval_mode_model_with_final_loss():
    vocab = _vocab()
    model = train_corpus(FLAT_PAIRS, vocab, hidden=8, seed=0,
                         batch_size=4, max_epochs=3, patience=2,
                         device="cpu")
    assert isinstance(model, CharGRU)
    assert model.training is False
    assert isinstance(model.final_loss, float)


def test_train_corpus_deterministic_with_same_seed():
    vocab = _vocab()
    a = train_corpus(FLAT_PAIRS, vocab, hidden=8, seed=0,
                     batch_size=4, max_epochs=3, patience=2, device="cpu")
    b = train_corpus(FLAT_PAIRS, vocab, hidden=8, seed=0,
                     batch_size=4, max_epochs=3, patience=2, device="cpu")
    a_state = a.state_dict()
    b_state = b.state_dict()
    assert a_state.keys() == b_state.keys()
    for key in a_state:
        assert torch.equal(a_state[key], b_state[key])


def test_train_corpus_conditioned_returns_eval_mode_model_with_final_loss():
    vocab = _vocab()
    model = train_corpus_conditioned(TRAIN_PAIRS, VAL_PAIRS, vocab, hidden=8,
                                     seed=0, batch_size=4, max_epochs=3,
                                     patience=2, device="cpu")
    assert isinstance(model, CharGRU)
    assert model.training is False
    assert isinstance(model.final_loss, float)


def test_train_corpus_conditioned_deterministic_with_same_seed():
    vocab = _vocab()
    a = train_corpus_conditioned(TRAIN_PAIRS, VAL_PAIRS, vocab, hidden=8,
                                 seed=0, batch_size=4, max_epochs=3,
                                 patience=2, device="cpu")
    b = train_corpus_conditioned(TRAIN_PAIRS, VAL_PAIRS, vocab, hidden=8,
                                 seed=0, batch_size=4, max_epochs=3,
                                 patience=2, device="cpu")
    a_state = a.state_dict()
    b_state = b.state_dict()
    assert a_state.keys() == b_state.keys()
    for key in a_state:
        assert torch.equal(a_state[key], b_state[key])
