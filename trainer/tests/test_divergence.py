import pytest
from ngpt_trainer.divergence import trigrams, jaccard_distance, cross_set_divergence

def test_trigrams_short_string():
    assert trigrams("") == {""}
    assert trigrams("a") == {"a"}
    assert trigrams("ab") == {"ab"}

def test_trigrams_long_string():
    assert trigrams("abc") == {"abc"}
    assert trigrams("abcd") == {"abc", "bcd"}
    assert trigrams("aaaa") == {"aaa"}

def test_jaccard_distance_identical_strings():
    assert jaccard_distance("", "") == 0.0
    assert jaccard_distance("abc", "abc") == 0.0

def test_jaccard_distance_completely_disjoint():
    assert jaccard_distance("abc", "def") == 1.0

def test_jaccard_distance_partial_overlap():
    # trigrams("abc") == {"abc"}; trigrams("abcd") == {"abc", "bcd"}.
    # intersection = {"abc"} (size 1), union = {"abc", "bcd"} (size 2)
    # -> distance = 1 - 1/2 = 0.5
    assert jaccard_distance("abc", "abcd") == 0.5

def test_cross_set_divergence_single_pair():
    assert cross_set_divergence(["abc"], ["def"]) == jaccard_distance("abc", "def")

def test_cross_set_divergence_multiple_pairs():
    samples_a = ["abc", "def"]
    samples_b = ["ghi", "jkl"]
    expected_divergence = (jaccard_distance("abc", "ghi") + jaccard_distance("abc", "jkl") +
                            jaccard_distance("def", "ghi") + jaccard_distance("def", "jkl")) / 4
    assert cross_set_divergence(samples_a, samples_b) == expected_divergence

def test_cross_set_divergence_empty_samples():
    with pytest.raises(ZeroDivisionError):
        cross_set_divergence([], ["def"])
    with pytest.raises(ZeroDivisionError):
        cross_set_divergence(["abc"], [])
