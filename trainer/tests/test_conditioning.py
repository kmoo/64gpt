"""M3 gate: one GRU memorizes all 12 prompt->response pairs; the prompt
alone selects which line comes out (greedy, deterministic)."""
import pytest

from ngpt_trainer import corpus
from ngpt_trainer.model import generate_greedy_prompted, overfit_corpus
from ngpt_trainer.vocab import Vocab


pytestmark = pytest.mark.slow

@pytest.fixture(scope="module")
def trained():
    vocab = Vocab.from_text(corpus.corpus_text())
    model = overfit_corpus(corpus.pairs(), vocab)
    return model, vocab


def test_every_prompt_generates_its_exact_response(trained):
    model, vocab = trained
    for prompt, response in corpus.pairs():
        assert generate_greedy_prompted(model, vocab, prompt) == response


def test_max_per_sequence_loss_reached_target(trained):
    model, _ = trained
    assert model.final_loss < 0.05


def test_same_seed_same_generations(trained):
    # Determinism needs equal step counts, not convergence: two short
    # runs must match each other exactly (keeps the suite fast).
    _, vocab = trained
    a = overfit_corpus(corpus.pairs(), vocab, seed=0, max_steps=300)
    b = overfit_corpus(corpus.pairs(), vocab, seed=0, max_steps=300)
    for prompt, _ in corpus.pairs():
        assert (generate_greedy_prompted(a, vocab, prompt)
                == generate_greedy_prompted(b, vocab, prompt))
