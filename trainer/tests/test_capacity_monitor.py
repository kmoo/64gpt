"""M14's capacity-monitoring metric (docs/milestones/m14.md DoD item):
whether adding cast size measurably degrades a NAMED character's own
held-out generalization loss, not just the aggregate val loss across
the whole combined val set (which can look fine while one voice quietly
degrades -- exactly the M8 capacity-dilution failure mode this exists
to catch again if it recurs post-M9's compositional fix).

Fast toy runs on tiny synthetic corpora, like every other model.py test
in this suite -- not a claim about production-scale numbers."""
from ngpt_trainer.capacity_monitor import (
    capacity_degradation_pct,
    exceeds_split_trigger,
    held_out_loss_for_subset,
)
from ngpt_trainer.model import train_corpus_conditioned
from ngpt_trainer.vocab import Vocab

TRAIN_PAIRS = [
    ("N:selena MOOD:happy ", "HELLO THERE FRIEND"),
    ("N:selena MOOD:sad ", "OH NO WHAT HAPPENED"),
    ("N:guard MOOD:happy ", "GOOD DAY CITIZEN"),
    ("N:guard MOOD:sad ", "MOVE ALONG NOW"),
]
VAL_PAIRS = [
    ("N:selena MOOD:happy ", "HELLO THERE FRIEND"),
    ("N:guard MOOD:happy ", "GOOD DAY CITIZEN"),
]


def _vocab():
    text = "".join(p + r for p, r in TRAIN_PAIRS + VAL_PAIRS)
    return Vocab.from_text(text)


def test_held_out_loss_for_subset_restricts_to_matching_pairs():
    vocab = _vocab()
    model = train_corpus_conditioned(
        TRAIN_PAIRS, VAL_PAIRS, vocab, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu")

    selena_loss = held_out_loss_for_subset(
        model, VAL_PAIRS, vocab, lambda prompt: "N:selena" in prompt,
        device="cpu")
    guard_loss = held_out_loss_for_subset(
        model, VAL_PAIRS, vocab, lambda prompt: "N:guard" in prompt,
        device="cpu")

    assert isinstance(selena_loss, float)
    assert isinstance(guard_loss, float)
    # Different subsets of a 2-pair val set are (almost certainly) not
    # numerically identical losses -- confirms the predicate actually
    # filtered rather than silently scoring the whole set both times.
    assert selena_loss != guard_loss


def test_held_out_loss_for_subset_returns_none_when_predicate_matches_nothing():
    vocab = _vocab()
    model = train_corpus_conditioned(
        TRAIN_PAIRS, VAL_PAIRS, vocab, hidden=8, seed=0, max_epochs=3,
        patience=2, batch_size=4, device="cpu")

    assert held_out_loss_for_subset(
        model, VAL_PAIRS, vocab, lambda prompt: "N:nobody_home" in prompt,
        device="cpu") is None


def test_capacity_degradation_pct_positive_means_worse():
    # Baseline 0.10, current 0.11 -- a 10% increase in held-out loss.
    assert abs(capacity_degradation_pct(0.10, 0.11) - 10.0) < 1e-9
    # Current better than baseline -- negative, not clamped to zero,
    # since an improvement is real signal too (the split-trigger check
    # below is what decides what to do with the sign).
    assert capacity_degradation_pct(0.10, 0.09) < 0.0


def test_exceeds_split_trigger_boundary_is_strict_greater_than():
    # Exactly at the threshold does NOT trigger -- matches this
    # project's existing strictness convention for pre-registered bars
    # (docs/milestones/m13.md: gap > noise_floor, not >=).
    assert exceeds_split_trigger(0.10, 0.105, threshold_pct=5.0) is False
    assert exceeds_split_trigger(0.10, 0.106, threshold_pct=5.0) is True


def test_exceeds_split_trigger_ignores_improvement():
    assert exceeds_split_trigger(0.10, 0.05, threshold_pct=5.0) is False
