from ngpt_trainer.corpus_gates import build_fragment_vocab, structural_gate, FragmentUsageTracker, density_gate

def test_build_fragment_vocab():
    vocab = build_fragment_vocab("Hello, world!", "This is a test.")
    assert "HELLO" in vocab
    assert "WORLD" in vocab
    assert "THIS" in vocab
    assert "IS" in vocab
    assert "A" in vocab
    assert "TEST" in vocab

def test_structural_gate():
    corpus_vocab = build_fragment_vocab("Hello, world!", "This is a test.")
    fragment = "Hello, world!"
    result, reason = structural_gate(fragment, corpus_vocab)
    assert result is True
    assert reason == ""

    fragment = "FOOBAR"
    result, reason = structural_gate(fragment, corpus_vocab)
    assert result is False
    assert reason == "invented word: FOOBAR"

    fragment = "Hello, FOOBAR!"
    result, reason = structural_gate(fragment, corpus_vocab, max_invented=1)
    assert result is True
    assert reason == ""

def test_structural_gate_counts_occurrences_not_distinct_words():
    # Regression: an earlier implementation deduped invented words through
    # a set before counting, so one invented word repeated 3x was scored
    # as "1 invented word" and slipped past max_invented=1. Must match
    # invented_word_count's occurrence-counting definition exactly.
    corpus_vocab = build_fragment_vocab("Hello, world!")
    fragment = "FOOBAR FOOBAR FOOBAR"
    result, reason = structural_gate(fragment, corpus_vocab, max_invented=1)
    assert result is False
    assert reason == "invented word: FOOBAR"

def test_fragment_usage_tracker():
    tracker = FragmentUsageTracker()
    tracker.record("Hello, world!", "Combo 1")
    tracker.record("Hello, world!", "Combo 2")
    tracker.record("Hello, world!", "Combo 3")
    assert tracker.usage_count("Hello, world!") == 3

    tracker.record("Hello, world!", "Combo 1")
    assert tracker.usage_count("Hello, world!") == 3

def test_density_gate():
    result, reason = density_gate(3, 3)
    assert result is True
    assert reason == ""

    result, reason = density_gate(2, 3)
    assert result is False
    assert reason == "used 2 times, need >= 3"

def test_build_fragment_vocab_empty():
    vocab = build_fragment_vocab()
    assert len(vocab) == 0

def test_structural_gate_empty_fragment():
    corpus_vocab = build_fragment_vocab("Hello, world!", "This is a test.")
    result, reason = structural_gate("", corpus_vocab)
    assert result is True
    assert reason == ""

def test_structural_gate_case_insensitivity():
    corpus_vocab = build_fragment_vocab("Hello, world!", "This is a test.")
    fragment = "hello"
    result, reason = structural_gate(fragment, corpus_vocab)
    assert result is True
    assert reason == ""

def test_structural_gate_apostrophe_handling():
    corpus_vocab = build_fragment_vocab("DON'T", "This is a test.")
    fragment = "DON'T"
    result, reason = structural_gate(fragment, corpus_vocab)
    assert result is True
    assert reason == ""

    corpus_vocab = build_fragment_vocab("DONT", "This is a test.")
    fragment = "DON'T"
    result, reason = structural_gate(fragment, corpus_vocab)
    assert result is False
    assert reason == "invented word: DON'T"

def test_density_gate_zero_usages():
    result, reason = density_gate(0, 0)
    assert result is True
    assert reason == ""

def test_fragment_usage_tracker_never_recorded():
    tracker = FragmentUsageTracker()
    assert tracker.usage_count("never recorded") == 0

def test_structural_gate_single_character_words():
    # Regression: the first draft built corpus_vocab FROM the fragment
    # itself, so "X" ended up in-vocab trivially -- the test passed even
    # with the single-character exemption deleted. Vocab here deliberately
    # excludes "X" so it's genuinely invented, and only the len>1 filter
    # keeps it from tripping the gate.
    corpus_vocab = build_fragment_vocab("I SEE THIS", "This is a test.")
    assert "X" not in corpus_vocab
    fragment = "I SEE X"
    result, reason = structural_gate(fragment, corpus_vocab)
    assert result is True
    assert reason == ""
