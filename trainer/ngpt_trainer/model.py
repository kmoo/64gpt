"""Float GRU baseline for M2.

One-hot input, no embedding: on the N64 side the input matvec collapses
to a column lookup, so the float model mirrors that shape (input_size =
vocab_size). Training here is throwaway — what ships is the quantized
integer replay; this model only has to overfit the corpus and hand its
weights to the quantizer.
"""
import torch
import torch.nn as nn


class CharGRU(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(input_size=vocab_size, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.head(out), h


def one_hot(ids: list[int], vocab_size: int) -> torch.Tensor:
    """[1, T, V] float32 — one sequence, batch_first."""
    t = torch.zeros(1, len(ids), vocab_size, dtype=torch.float32)
    for pos, i in enumerate(ids):
        t[0, pos, i] = 1.0
    return t


def overfit(corpus: str, vocab, hidden: int = 32, seed: int = 0, lr: float = 1e-2,
            max_steps: int = 2000, target_loss: float = 1e-3) -> CharGRU:
    torch.manual_seed(seed)
    model = CharGRU(vocab_size=len(vocab), hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Teacher forcing on the single sequence: EOS starts it, EOS ends it.
    inputs = one_hot([vocab.eos_id] + vocab.encode(corpus), len(vocab))
    targets = torch.tensor(vocab.encode(corpus) + [vocab.eos_id], dtype=torch.long)

    loss = None
    for _ in range(max_steps):
        opt.zero_grad()
        logits, _ = model(inputs)
        loss = loss_fn(logits.squeeze(0), targets)
        loss.backward()
        opt.step()
        if loss.item() < target_loss:
            break

    model.eval()
    model.final_loss = loss.item()
    return model


def generate_greedy(model: CharGRU, vocab, max_len: int = 256) -> str:
    """Greedy decode from h=0 / EOS input. torch.argmax breaks ties toward
    the lowest id, matching the integer reference implementation."""
    out = []
    h = None
    current = vocab.eos_id
    with torch.no_grad():
        for _ in range(max_len):
            logits, h = model(one_hot([current], len(vocab)), h)
            current = int(torch.argmax(logits[0, -1]).item())
            if current == vocab.eos_id:
                break
            out.append(vocab.decode([current]))
    return "".join(out)
