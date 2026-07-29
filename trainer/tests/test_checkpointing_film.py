"""Mid-run checkpointing for the FiLM training loop (M12.5) --
docs/plan.md Known Follow-up: this is the variant where the ORIGINAL
M12.5 near-OOM incident happened (a real GPU/Metal command-buffer crash
mid-run), yet it was the one variant left uncovered when the _attr pair
got checkpoint_path in the prior overnight session. Same "fast toy run"
scope as test_checkpointing.py/test_checkpointing_plain.py."""
import torch

from ngpt_trainer.model import (CharGRUFiLM, qat_finetune_film,
                                 train_corpus_conditioned_film)
from ngpt_trainer.vocab import Vocab

TRAIN_PAIRS = [
    ("HELLO ", "WORLD"),
    ("GOOD ", "MORNING"),
    ("BAD ", "NIGHT"),
    ("STAY ", "ALERT"),
]
VAL_PAIRS = [("HELLO ", "WORLD"), ("BAD ", "NIGHT")]
TRAIN_ATTRS = [(0, 0), (1, 1), (0, 1), (1, 0)]
VAL_ATTRS = [(0, 0), (0, 1)]
N_DESC, N_MOOD = 2, 2


def _vocab():
    text = "".join(p + r for p, r in TRAIN_PAIRS + VAL_PAIRS)
    return Vocab.from_text(text)


def test_float_phase_writes_a_loadable_checkpoint(tmp_path):
    vocab = _vocab()
    ckpt = tmp_path / "float.pt"
    model = train_corpus_conditioned_film(
        TRAIN_PAIRS, VAL_PAIRS, TRAIN_ATTRS, VAL_ATTRS, vocab,
        n_desc=N_DESC, n_mood=N_MOOD, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu", checkpoint_path=str(ckpt))

    assert ckpt.exists()
    saved = torch.load(ckpt, weights_only=True)
    assert set(saved) == {"state", "val_loss", "epoch", "hidden", "n_desc", "n_mood"}
    assert saved["hidden"] == 8
    assert saved["n_desc"] == N_DESC
    assert saved["n_mood"] == N_MOOD
    assert isinstance(saved["val_loss"], float)
    assert saved["val_loss"] >= model.final_loss - 1e-6

    reloaded = CharGRUFiLM(vocab_size=len(vocab), hidden=8, n_attr=N_DESC + N_MOOD)
    reloaded.load_state_dict(saved["state"])


def test_qat_phase_writes_a_loadable_checkpoint(tmp_path):
    vocab = _vocab()
    float_model = train_corpus_conditioned_film(
        TRAIN_PAIRS, VAL_PAIRS, TRAIN_ATTRS, VAL_ATTRS, vocab,
        n_desc=N_DESC, n_mood=N_MOOD, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu")

    ckpt = tmp_path / "qat.pt"
    qat_finetune_film(
        float_model, TRAIN_PAIRS, VAL_PAIRS, TRAIN_ATTRS, VAL_ATTRS, vocab,
        n_desc=N_DESC, n_mood=N_MOOD, seed=0, max_epochs=3, patience=2,
        batch_size=4, device="cpu", checkpoint_path=str(ckpt))

    assert ckpt.exists()
    saved = torch.load(ckpt, weights_only=True)
    assert set(saved) == {"state", "val_loss", "epoch", "n_desc", "n_mood"}


def test_checkpoint_path_none_is_backward_compatible(tmp_path):
    """make_m12_5_film_blob.py (and any other existing caller) never
    passes this argument -- default behavior must be unaffected."""
    vocab = _vocab()
    model = train_corpus_conditioned_film(
        TRAIN_PAIRS, VAL_PAIRS, TRAIN_ATTRS, VAL_ATTRS, vocab,
        n_desc=N_DESC, n_mood=N_MOOD, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu")
    assert isinstance(model.final_loss, float)
    assert list(tmp_path.iterdir()) == []
