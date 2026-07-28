"""Mid-run checkpointing for the shared attr-conditioned training loop
(docs/plan.md Known Follow-ups, raised 2026-07-26 during M12.5 -- a real
GPU/Metal OOM crashed a ~3-hour training run mid-way with no on-disk
recovery path until the final QAT save). Verifies the checkpoint file
actually appears and is loadable, on a fast toy run -- not a claim about
production training time or quality, just that the mechanism works."""
import torch

from ngpt_trainer.model import (CharGRU, qat_finetune_attr,
                                 train_corpus_conditioned_attr)
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
    model = train_corpus_conditioned_attr(
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
    # The checkpoint's val_loss is whatever the LAST improving epoch saw,
    # never worse than the model's own final (best-restored) loss.
    assert saved["val_loss"] >= model.final_loss - 1e-6

    # Loadable back into a real model with matching architecture.
    reloaded = CharGRU(vocab_size=len(vocab), hidden=8,
                        input_size=len(vocab) + N_DESC + N_MOOD)
    reloaded.load_state_dict(saved["state"])


def test_qat_phase_writes_a_loadable_checkpoint(tmp_path):
    vocab = _vocab()
    float_model = train_corpus_conditioned_attr(
        TRAIN_PAIRS, VAL_PAIRS, TRAIN_ATTRS, VAL_ATTRS, vocab,
        n_desc=N_DESC, n_mood=N_MOOD, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu")

    ckpt = tmp_path / "qat.pt"
    qat_finetune_attr(
        float_model, TRAIN_PAIRS, VAL_PAIRS, TRAIN_ATTRS, VAL_ATTRS, vocab,
        n_desc=N_DESC, n_mood=N_MOOD, seed=0, max_epochs=3, patience=2,
        batch_size=4, device="cpu", checkpoint_path=str(ckpt))

    assert ckpt.exists()
    saved = torch.load(ckpt, weights_only=True)
    assert set(saved) == {"state", "val_loss", "epoch", "n_desc", "n_mood"}


def test_checkpoint_path_none_is_backward_compatible(tmp_path):
    """Default behavior (no checkpoint_path) must be completely
    unaffected -- existing callers (make_m12_1_blob.py, the M12.x spike
    scripts) never pass this argument."""
    vocab = _vocab()
    model = train_corpus_conditioned_attr(
        TRAIN_PAIRS, VAL_PAIRS, TRAIN_ATTRS, VAL_ATTRS, vocab,
        n_desc=N_DESC, n_mood=N_MOOD, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu")
    assert isinstance(model.final_loss, float)
    assert list(tmp_path.iterdir()) == []
