import re

_WORD_RE = re.compile(r"[A-Z']+")

def build_fragment_vocab(*texts: str) -> set[str]:
    vocab = set()
    for text in texts:
        vocab.update(_WORD_RE.findall(text.upper()))
    return vocab

class FragmentUsageTracker:
    def __init__(self):
        self.fragment_combos = {}

    def record(self, fragment: str, combo_text: str) -> None:
        if fragment not in self.fragment_combos:
            self.fragment_combos[fragment] = set()
        self.fragment_combos[fragment].add(combo_text)

    def usage_count(self, fragment: str) -> int:
        return len(self.fragment_combos.get(fragment, set()))

def structural_gate(fragment: str, corpus_vocab: set[str], max_invented: int = 0) -> tuple[bool, str]:
    # Occurrence count, not distinct-word count: must match
    # invented_word_count's definition exactly (make_m12_1_blob.py) --
    # "FOOBAR FOOBAR FOOBAR" is 3 invented occurrences, not 1 invented
    # word, so it still trips max_invented=1.
    words = _WORD_RE.findall(fragment.upper())
    invented_words = [w for w in words if w not in corpus_vocab and len(w) > 1]
    if len(invented_words) > max_invented:
        return False, f"invented word: {invented_words[0]}"
    return True, ""

def density_gate(usage_count: int, min_reuse: int) -> tuple[bool, str]:
    if usage_count >= min_reuse:
        return True, ""
    return False, f"used {usage_count} times, need >= {min_reuse}"
