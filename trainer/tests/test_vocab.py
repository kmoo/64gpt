"""Vocab spec: id 0 is reserved for EOS; ids 1..V-1 are the sorted unique
corpus chars. See docs/milestones/m2.md."""
import pytest

from ngpt_trainer.vocab import Vocab

CORPUS = "HALT! WHO GOES THERE?"


def test_eos_is_id_zero_and_len_counts_it():
    vocab = Vocab.from_text(CORPUS)
    assert vocab.eos_id == 0
    assert len(vocab) == len(set(CORPUS)) + 1


def test_charset_is_sorted_unique_corpus_chars():
    vocab = Vocab.from_text(CORPUS)
    assert vocab.charset == "".join(sorted(set(CORPUS)))


def test_encode_decode_round_trip():
    vocab = Vocab.from_text(CORPUS)
    assert vocab.decode(vocab.encode(CORPUS)) == CORPUS


def test_encode_unknown_char_raises_keyerror():
    vocab = Vocab.from_text(CORPUS)
    with pytest.raises(KeyError):
        vocab.encode("halt")  # lowercase never in corpus


def test_decode_eos_id_raises():
    vocab = Vocab.from_text(CORPUS)
    with pytest.raises(ValueError):
        vocab.decode([vocab.eos_id])


def test_decode_out_of_range_raises():
    vocab = Vocab.from_text(CORPUS)
    with pytest.raises(ValueError):
        vocab.decode([len(vocab)])


def test_to_bytes_layout():
    vocab = Vocab.from_text(CORPUS)
    b = vocab.to_bytes()
    assert b[0] == 0x00
    assert b[1:] == vocab.charset.encode("ascii")


def test_bytes_round_trip():
    vocab = Vocab.from_text(CORPUS)
    again = Vocab.from_bytes(vocab.to_bytes())
    assert again.charset == vocab.charset
    assert again.decode(again.encode(CORPUS)) == CORPUS


def test_from_text_deterministic():
    assert Vocab.from_text(CORPUS).to_bytes() == Vocab.from_text(CORPUS).to_bytes()


def test_non_printable_ascii_rejected():
    with pytest.raises(ValueError):
        Vocab.from_text("HALT!\x01WHO GOES THERE?")
