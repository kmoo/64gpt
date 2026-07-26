"""Corpus-wide charset + lexicon-trie invariants (added M12.2).

Guards the exact failure mode the M12.2 voice-edit pass worried about: a
phrase-bank edit sneaking in a character outside the model's vocab, or the
lexicon-trie's allowable-word set drifting out of sync with the corpus.

These are char-level / N64 hard constraints, not cosmetics. Every corpus
generator feeds ONE shared ``Vocab.from_text`` at build time
(make_m12_1_blob.py), so a stray byte resizes the vocab -- and the engine's
per-vocab scratch arrays are a FIXED size (``NGPT_GRU_MAX_VOCAB`` in
core/ngpt.h), so an oversized vocab fails to load on real hardware. The
lexicon-trie guard (M12.1 phase 4) is likewise rebuilt from the corpus
every build; this test locks the "allowable words can never go stale"
guarantee that property depends on.
"""
import re

from ngpt_trainer import (selena_corpus as sc, guard_corpus as gc,
                          cast_corpus as cc, shadewrath_corpus as swc,
                          korrath_corpus as kc, princess_corpus as pc)
from ngpt_trainer.ref_impl import build_word_trie, TRIE_NONE
from ngpt_trainer.vocab import Vocab

# Every character a RESPONSE may contain: uppercase A-Z (the N64 debug font
# has no lowercase glyphs), space, and the punctuation the authored banks
# use. ':' is legitimately present in a handful of pre-existing lines
# (e.g. "I DON'T SAY THIS LIGHTLY: ..."), so it is allowed here.
RESPONSE_ALLOWED = set(" ABCDEFGHIJKLMNOPQRSTUVWXYZ'!,-.:?")

# Mirrors core/ngpt.h's ``#define NGPT_GRU_MAX_VOCAB`` -- the fixed length
# of the engine's per-vocab scratch arrays. A vocab larger than this fails
# ngpt_gru_load's dims check (NGPT_ERR_DIMS) on hardware.
NGPT_GRU_MAX_VOCAB = 96

# per_combo is charset-invariant (more repetitions add no new characters),
# so modest values keep the test fast while still drawing every bank line.
_GENERATORS = [(sc, 4), (gc, 4), (cc, 3), (swc, 4), (kc, 4), (pc, 4)]


def _all_pairs():
    pairs = []
    for mod, per in _GENERATORS:
        pairs.extend(mod.generate_pairs(seed=0, per_combo=per))
    return pairs


def _full_text():
    return "".join(p + r for p, r in _all_pairs())


def test_every_response_char_is_in_the_allowed_set():
    """A voice edit that introduced, say, a ';' or a curly quote would
    silently enlarge the vocab. Catch it at the response level, where the
    edits live."""
    offenders = {}
    for _, r in _all_pairs():
        extra = set(r) - RESPONSE_ALLOWED
        if extra:
            offenders[r] = sorted(extra)
    assert not offenders, f"responses contain out-of-vocab chars: {offenders}"


def test_every_response_is_uppercase():
    for p, r in _all_pairs():
        assert r == r.upper(), f"non-uppercase response: {r!r} (prompt {p!r})"


def test_combined_vocab_stays_within_engine_limit():
    n = len(Vocab.from_text(_full_text()))
    assert n <= NGPT_GRU_MAX_VOCAB, (
        f"combined-corpus vocab is {n} symbols > engine cap "
        f"{NGPT_GRU_MAX_VOCAB} (core/ngpt.h NGPT_GRU_MAX_VOCAB) -- the blob "
        f"would fail ngpt_gru_load's dims check on hardware")


def test_trie_contains_exactly_every_corpus_word():
    """The lexicon-trie's allowable-word set is derived from the corpus on
    every build, so it can never go stale -- this LOCKS that invariant:
    every [A-Z']+ word the corpus contains must walk the trie to an
    end-of-word node (build_word_trie's flags bit 0)."""
    words = set(re.findall(r"[A-Z']+", _full_text().upper()))
    nodes = build_word_trie(words)
    for w in words:
        node = 0
        for ch in w:
            b = ord(ch)
            child = nodes[node][2]
            found = TRIE_NONE
            while child != TRIE_NONE:
                if nodes[child][0] == b:
                    found = child
                    break
                child = nodes[child][3]
            assert found != TRIE_NONE, f"word {w!r} not walkable in trie at {ch!r}"
            node = found
        assert nodes[node][1] & 1, f"word {w!r} present but not end-of-word marked"
