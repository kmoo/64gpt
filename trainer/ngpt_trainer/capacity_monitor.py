"""M14's capacity-monitoring metric (docs/milestones/m14.md).

M8's own measured evidence (docs/milestones/m8.md, m9.md's Why section)
is why this exists: three retrains at guard density 1.7%->12.6% of the
shared corpus moved Selena's held-out val loss 0.0964->0.0991->0.1026 --
capacity dilution that the project's usual AGGREGATE val loss and
conditioning-ablation divergence table don't isolate, because both are
computed over the whole combined val set, not one named character's
own slice of it. M9's compositional scheme fixed the specific mechanism
(opaque per-id memorization), but nothing currently re-checks whether
that fix keeps holding as the cast keeps growing (M10's bad guy, M11's
full town, M14's ported archetypes) -- this module is that re-check.

Predicate-based, not schema-hardcoded: the caller supplies what
"belongs to this character" means for their own corpus's prompt tags
(an N: name tag, an OCC: value unique to one named character, whatever
the schema in use actually is) rather than this module assuming any one
project's field vocabulary -- the same portability discipline M14's own
manifest work requires elsewhere.
"""
import torch
import torch.nn as nn

from ngpt_trainer.model import _batchify_masked


def held_out_loss_for_subset(model, val_pairs, vocab, predicate,
                             device: str | None = None,
                             batch_size: int = 64) -> float | None:
    """Masked held-out loss (docs/milestones/m7.md's prefix-masking
    scheme, same as train_corpus_conditioned's own val_loss) restricted
    to the val_pairs whose PROMPT satisfies predicate. Returns None if
    predicate matches zero pairs, rather than dividing by zero -- an
    absent character in this val split is a caller error to notice, not
    a silent 0.0."""
    if device is None:
        device = "cpu"
    subset = [(p, r) for p, r in val_pairs if predicate(p)]
    if not subset:
        return None

    ids = [vocab.encode(p) + vocab.encode(r) for p, r in subset]
    plens = [len(vocab.encode(p)) for p, _ in subset]
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            inputs, targets = _batchify_masked(
                ids[i:i + batch_size], plens[i:i + batch_size], len(vocab))
            logits, _ = model(inputs.to(device))
            n = (targets != -100).sum().item()
            total += loss_fn(logits.reshape(-1, len(vocab)),
                             targets.reshape(-1).to(device)).item() * n
            count += n
    return total / count if count else None


def capacity_degradation_pct(baseline_loss: float, current_loss: float) -> float:
    """Percent change in held-out loss, baseline -> current. Positive
    means WORSE (loss went up); negative means better. Sign is kept
    (not clamped) so callers can distinguish "no change", "improved",
    and "degraded" rather than collapsing the first two together."""
    return (current_loss - baseline_loss) / baseline_loss * 100.0


def exceeds_split_trigger(baseline_loss: float, current_loss: float,
                          threshold_pct: float) -> bool:
    """The concrete decision rule docs/milestones/m10.md's Data Science
    Review named but left unquantified, and m14.md's split-trigger DoD
    item asks to pin down: has a named character's own held-out loss
    degraded by MORE than threshold_pct versus its own smaller-cast
    baseline? Strictly-greater, matching this project's other
    pre-registered-bar convention (m13.md: gap > noise_floor, not >=) --
    landing exactly on the line is not treated as a pass for the cast
    that's already grown, since the trigger exists to catch real
    degradation, not to be satisfied by coincidence at the boundary.
    An improvement (negative degradation) never triggers."""
    return capacity_degradation_pct(baseline_loss, current_loss) > threshold_pct
