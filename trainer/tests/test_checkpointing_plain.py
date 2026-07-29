"""Mid-run checkpointing for the PLAIN (non-attr) training loop --
docs/plan.md Known Follow-ups: train_corpus_conditioned_attr/
qat_finetune_attr already got checkpoint_path (test_checkpointing.py);
this closes the same gap for the plain pair, the one
make_m12_1_blob.py's actual SHIPPED build path uses. Same "fast toy
run, not a claim about production quality" scope as test_checkpointing.py."""
import torch

from ngpt_trainer.model import CharGRU, qat_finetune, train_corpus_conditioned
from ngpt_trainer.vocab import Vocab

TRAIN_PAIRS = [
    ("HELLO ", "WORLD"),
    ("GOOD ", "MORNING"),
    ("BAD ", "NIGHT"),
    ("STAY ", "ALERT"),
]
VAL_PAIRS = [("HELLO ", "WORLD"), ("BAD ", "NIGHT")]


def _vocab():
    text = "".join(p + r for p, r in TRAIN_PAIRS + VAL_PAIRS)
    return Vocab.from_text(text)


def test_float_phase_writes_a_loadable_checkpoint(tmp_path):
    vocab = _vocab()
    ckpt = tmp_path / "float.pt"
    model = train_corpus_conditioned(
        TRAIN_PAIRS, VAL_PAIRS, vocab, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu", checkpoint_path=str(ckpt))

    assert ckpt.exists()
    saved = torch.load(ckpt, weights_only=True)
    assert set(saved) == {"state", "val_loss", "epoch", "hidden"}
    assert saved["hidden"] == 8
    assert isinstance(saved["val_loss"], float)
    assert saved["val_loss"] >= model.final_loss - 1e-6

    reloaded = CharGRU(vocab_size=len(vocab), hidden=8)
    reloaded.load_state_dict(saved["state"])


def test_qat_phase_writes_a_loadable_checkpoint(tmp_path):
    vocab = _vocab()
    float_model = train_corpus_conditioned(
        TRAIN_PAIRS, VAL_PAIRS, vocab, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu")

    ckpt = tmp_path / "qat.pt"
    qat_finetune(
        float_model, TRAIN_PAIRS, VAL_PAIRS, vocab, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu", checkpoint_path=str(ckpt))

    assert ckpt.exists()
    saved = torch.load(ckpt, weights_only=True)
    assert set(saved) == {"state", "val_loss", "epoch"}


def test_checkpoint_path_none_is_backward_compatible(tmp_path):
    """Existing callers (make_m12_1_blob.py) never pass this argument --
    default behavior must be completely unaffected."""
    vocab = _vocab()
    model = train_corpus_conditioned(
        TRAIN_PAIRS, VAL_PAIRS, vocab, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu")
    assert isinstance(model.final_loss, float)
    assert list(tmp_path.iterdir()) == []
