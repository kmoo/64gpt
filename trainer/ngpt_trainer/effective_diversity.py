"""Effective-diversity diagnostic for generated dialogue corpora
(docs/milestones/m7.md, "Effective diversity, not raw byte count"):
distinct-trigram coverage and a near-duplicate-collapsed unique-line
ratio, computed per combo and overall. Catches a generator that's
padding (more bytes, same handful of shapes) rather than genuinely
diversifying — templated text carries much lower entropy per character
than natural language, so raw byte count alone is a misleading corpus-
size metric. Fully generic (plain strings in) so it's reusable for
later corpora (archetypes, bosses) without modification.
"""


def trigrams(s: str) -> set:
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def trigram_coverage(lines: list[str]) -> tuple[int, int, float]:
    """(distinct_trigram_count, total_trigram_occurrences, coverage_ratio).
    Every position in every line counts toward the total, even if the
    same 3-character shape recurs — coverage_ratio drops toward 0 as the
    same shapes repeat more, rises toward 1.0 when every occurrence is
    distinct."""
    distinct: set = set()
    total = 0
    for line in lines:
        line_trigrams = trigrams(line)
        count = len(line) - 2 if len(line) >= 3 else 1
        total += count
        distinct |= line_trigrams
    coverage = len(distinct) / total if total else 0.0
    return len(distinct), total, coverage


def unique_line_ratio(lines: list[str]) -> float:
    """(count of distinct normalized lines) / (total line count), 0.0 for
    an empty list. Normalization: strip() then collapse internal
    whitespace runs to single spaces — near-duplicate collapse, exact
    match after normalization only (no fuzzy/edit-distance matching)."""
    if not lines:
        return 0.0
    normalized = {" ".join(line.split()) for line in lines}
    return len(normalized) / len(lines)


def effective_diversity_report(responses_by_combo: dict) -> dict:
    """Per-combo report plus an "overall" key aggregating every combo's
    lines together. Each value: distinct_trigrams, total_trigrams,
    trigram_coverage, unique_line_ratio. No combos in -> {} out (no
    fabricated "overall" from zero data)."""
    if not responses_by_combo:
        return {}
    report = {}
    all_lines: list[str] = []
    for key, lines in responses_by_combo.items():
        distinct, total, coverage = trigram_coverage(lines)
        report[key] = {
            "distinct_trigrams": distinct,
            "total_trigrams": total,
            "trigram_coverage": coverage,
            "unique_line_ratio": unique_line_ratio(lines),
        }
        all_lines.extend(lines)
    distinct, total, coverage = trigram_coverage(all_lines)
    report["overall"] = {
        "distinct_trigrams": distinct,
        "total_trigrams": total,
        "trigram_coverage": coverage,
        "unique_line_ratio": unique_line_ratio(all_lines),
    }
    return report
