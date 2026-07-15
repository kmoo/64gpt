"""M2 overfit gate: a GRU trained on ONE line must reproduce it exactly
with greedy decoding (float baseline; the integer path must then match)."""
import pytest

from ngpt_trainer.model import generate_greedy, overfit
from ngpt_trainer.vocab import Vocab

CORPUS = "HALT! WHO GOES THERE?"


@pytest.fixture(scope="module")
def trained():
    vocab = Vocab.from_text(CORPUS)
    model = overfit(CORPUS, vocab)
    return model, vocab


def test_greedy_reproduces_corpus_exactly(trained):
    model, vocab = trained
    assert generate_greedy(model, vocab) == CORPUS


def test_loss_reached_target(trained):
    model, _ = trained
    assert model.final_loss < 1e-3


def test_same_seed_same_generation(trained):
    model, vocab = trained
    retrained = overfit(CORPUS, vocab, seed=0)
    assert generate_greedy(retrained, vocab) == generate_greedy(model, vocab)
