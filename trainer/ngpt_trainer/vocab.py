"""Character vocabulary for the 64GPT models.

Id 0 is RESERVED for EOS — the stop token the GRU emits to end generation.
It never appears in text (so decode refuses it), which is what lets the
C engine use a single sentinel check to terminate the stream. Ids 1..V-1
are the corpus's unique characters in sorted (byte-value) order, so the
mapping is deterministic and the blob's charset table is reproducible.
"""


class Vocab:
    def __init__(self, charset: str):
        self.charset = charset
        self.eos_id = 0
        self._char_to_id = {c: i + 1 for i, c in enumerate(charset)}

    @classmethod
    def from_text(cls, corpus: str) -> "Vocab":
        if any(not (32 <= ord(c) <= 126) for c in corpus):
            raise ValueError("corpus must be printable ASCII (32..126)")
        return cls("".join(sorted(set(corpus))))

    def __len__(self) -> int:
        return len(self.charset) + 1  # +1 for the EOS slot at id 0

    def encode(self, s: str) -> list[int]:
        return [self._char_to_id[c] for c in s]  # KeyError on unknown char

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if not (1 <= i <= len(self.charset)):
                raise ValueError(f"id {i} not decodable (0=EOS, max {len(self.charset)})")
            out.append(self.charset[i - 1])
        return "".join(out)

    def to_bytes(self) -> bytes:
        return b"\x00" + self.charset.encode("ascii")

    @classmethod
    def from_bytes(cls, b: bytes) -> "Vocab":
        if not b or b[0] != 0:
            raise ValueError("charset table must start with the 0x00 EOS slot")
        return cls(b[1:].decode("ascii"))
