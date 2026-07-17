"""Character-trigram Jaccard divergence — the conditioning-ablation
metric fixed by the M7 open-questions resolution (docs/milestones/m7.md):
cheaper than KL, no probability distributions needed, and already
validated as discriminating by docs/spikes/identity-conditioning.md.
Shared by the spike and by make_m7_blob.py's acceptance gates so both
use exactly the same metric.
"""


def trigrams(s: str) -> set:
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def jaccard_distance(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    union = ta | tb
    if not union:
        return 0.0
    return 1.0 - len(ta & tb) / len(union)


def cross_set_divergence(samples_a: list[str], samples_b: list[str]) -> float:
    """Mean pairwise divergence between two sample sets — robust to a
    single string landing on either side's mode by chance (the spike's
    own lesson: never judge conditioning on one greedy/sampled string)."""
    divs = [jaccard_distance(a, b) for a in samples_a for b in samples_b]
    return sum(divs) / len(divs)
