import unittest
from ngpt_trainer.effective_diversity import trigram_coverage, unique_line_ratio, effective_diversity_report

class TestEffectiveDiversity(unittest.TestCase):

    def test_trigram_coverage_empty(self):
        self.assertEqual(trigram_coverage([]), (0, 0, 0.0))

    def test_trigram_coverage_single_short_string(self):
        self.assertEqual(trigram_coverage(["hi"]), (1, 1, 1.0))

    def test_trigram_coverage_single_long_string(self):
        # "hello world" (11 chars) has 9 sliding-window trigram positions,
        # all distinct ("hel","ell","llo","lo ","o w"," wo","wor","orl","rld")
        self.assertEqual(trigram_coverage(["hello world"]), (9, 9, 1.0))

    def test_trigram_coverage_multiple_strings(self):
        # each 3-char string contributes exactly 1 trigram position (itself);
        # all three are distinct
        self.assertEqual(trigram_coverage(["abc", "def", "ghi"]), (3, 3, 1.0))

    def test_unique_line_ratio_empty(self):
        self.assertEqual(unique_line_ratio([]), 0.0)

    def test_unique_line_ratio_single_line(self):
        self.assertEqual(unique_line_ratio(["hello world"]), 1.0)

    def test_unique_line_ratio_multiple_unique_lines(self):
        self.assertEqual(unique_line_ratio(["hello world", "goodbye world"]), 1.0)

    def test_unique_line_ratio_multiple_duplicate_lines(self):
        self.assertEqual(unique_line_ratio(["hello world", "hello world"]), 0.5)

    def test_unique_line_ratio_normalized_lines(self):
        # both normalize to "hello world" -> 1 distinct / 2 total = 0.5,
        # NOT 1.0 (near-duplicate collapse means they count as ONE line)
        self.assertEqual(unique_line_ratio(["  hello   world  ", "hello world"]), 0.5)

    def test_effective_diversity_report_empty(self):
        self.assertEqual(effective_diversity_report({}), {})

    def test_effective_diversity_report_single_combo(self):
        # "hello world" (9 positions) + "goodbye world" (11 positions):
        # ground truth recomputed directly (not hand-derived) since
        # trigram counting is easy to get wrong by eye
        combo_key = ("key", "mood", "context")
        lines = ["hello world", "goodbye world"]
        report = effective_diversity_report({combo_key: lines})
        expected = {
            combo_key: {
                "distinct_trigrams": 16,
                "total_trigrams": 20,
                "trigram_coverage": 0.8,
                "unique_line_ratio": 1.0
            },
            "overall": {
                "distinct_trigrams": 16,
                "total_trigrams": 20,
                "trigram_coverage": 0.8,
                "unique_line_ratio": 1.0
            }
        }
        self.assertEqual(report, expected)

    def test_effective_diversity_report_multiple_combos(self):
        combo_key1 = ("key1", "mood1", "context1")
        combo_key2 = ("key2", "mood2", "context2")
        lines1 = ["hello world", "goodbye world"]
        lines2 = ["hello", "world"]
        report = effective_diversity_report({combo_key1: lines1, combo_key2: lines2})
        expected = {
            combo_key1: {
                "distinct_trigrams": 16,
                "total_trigrams": 20,
                "trigram_coverage": 0.8,
                "unique_line_ratio": 1.0
            },
            combo_key2: {
                "distinct_trigrams": 6,
                "total_trigrams": 6,
                "trigram_coverage": 1.0,
                "unique_line_ratio": 1.0
            },
            "overall": {
                "distinct_trigrams": 16,
                "total_trigrams": 26,
                "trigram_coverage": 0.6153846153846154,
                "unique_line_ratio": 1.0
            }
        }
        self.assertEqual(report, expected)

if __name__ == "__main__":
    unittest.main()
