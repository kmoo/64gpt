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


def overfit_corpus(pairs: list[tuple[str, str]], vocab, hidden: int = 64,
                   seed: int = 0, lr: float = 5e-3, max_steps: int = 8000,
                   target_loss: float = 1e-2) -> CharGRU:
    """Memorize all prompt->response pairs. One optimizer step per epoch
    on the SUM of per-sequence losses; stops once the max per-sequence
    loss is under target AND every pair reproduces exactly (greedy)."""
    torch.manual_seed(seed)
    model = CharGRU(vocab_size=len(vocab), hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    seqs = []
    for prompt, response in pairs:
        ids = vocab.encode(prompt) + vocab.encode(response)
        seqs.append((one_hot([vocab.eos_id] + ids, len(vocab)),
                     torch.tensor(ids + [vocab.eos_id], dtype=torch.long)))

    worst = None
    for step in range(max_steps):
        opt.zero_grad()
        losses = []
        for inputs, targets in seqs:
            logits, _ = model(inputs)
            losses.append(loss_fn(logits.squeeze(0), targets))
        sum(losses).backward()
        opt.step()
        worst = max(l.item() for l in losses)
        # The real goal is behavioral: every pair reproduced exactly.
        # The loss threshold on top keeps enough margin that int8
        # quantization doesn't flip an argmax (the integer ref-impl
        # tests are the final judge of that).
        if worst < target_loss and step % 25 == 0:
            model.eval()
            ok = all(generate_greedy_prompted(model, vocab, p) == r
                     for p, r in pairs)
            model.train()
            if ok:
                break

    model.eval()
    model.final_loss = worst
    return model


def _batchify(seqs: list[list[int]], vocab_size: int, pad_target: int = -100):
    """Pad a list of id-sequences into (inputs [B,T,V] one-hot, targets
    [B,T] with pad_target on padding). Sequence k is EOS+ids -> ids+EOS
    teacher forcing, like everywhere else in this project."""
    B = len(seqs)
    T = max(len(s) + 1 for s in seqs)
    inputs = torch.zeros(B, T, vocab_size, dtype=torch.float32)
    targets = torch.full((B, T), pad_target, dtype=torch.long)
    for b, ids in enumerate(seqs):
        eos_first = [0] + ids  # vocab.eos_id is 0 by construction
        for pos, i in enumerate(eos_first):
            inputs[b, pos, i] = 1.0
        targets[b, : len(ids) + 1] = torch.tensor(ids + [0], dtype=torch.long)
    return inputs, targets


def train_corpus(pairs: list[tuple[str, str]], vocab, hidden: int = 128,
                 seed: int = 0, lr: float = 3e-3, batch_size: int = 64,
                 max_epochs: int = 60, patience: int = 5,
                 device: str | None = None) -> CharGRU:
    """M4 training: mini-batches over a generated corpus with a held-out
    validation split (every 10th pair — the corpus is combo-interleaved,
    so the split covers every condition). Unlike overfit_corpus the goal
    is GENERALIZATION: early-stop on best val loss, restore that model.

    Float training is throwaway scaffolding (device may be MPS; numerics
    need not be reproducible) — what ships is the quantized integer
    replay, and the acceptance gate (val loss + int-vs-float top-1
    agreement) runs downstream in make_m4_blob.py."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    model = CharGRU(vocab_size=len(vocab), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    encoded = [vocab.encode(p) + vocab.encode(r) for p, r in pairs]
    assert vocab.eos_id == 0
    val = encoded[9::10]
    train = [s for i, s in enumerate(encoded) if i % 10 != 9]

    def val_loss() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), batch_size):
                inputs, targets = _batchify(val[i:i + batch_size], len(vocab))
                logits, _ = model(inputs.to(device))
                n = (targets != -100).sum().item()
                total += loss_fn(logits.reshape(-1, len(vocab)),
                                 targets.reshape(-1).to(device)).item() * n
                count += n
        model.train()
        return total / count

    rng = torch.Generator().manual_seed(seed)
    best, best_state, since_best = float("inf"), None, 0
    for epoch in range(max_epochs):
        order = torch.randperm(len(train), generator=rng).tolist()
        for i in range(0, len(order), batch_size):
            batch = [train[j] for j in order[i:i + batch_size]]
            inputs, targets = _batchify(batch, len(vocab))
            opt.zero_grad()
            logits, _ = model(inputs.to(device))
            loss = loss_fn(logits.reshape(-1, len(vocab)),
                           targets.reshape(-1).to(device))
            loss.backward()
            opt.step()
        v = val_loss()
        print(f"epoch {epoch}: val loss {v:.4f}", flush=True)
        if v < best:
            best, since_best = v, 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    model.final_loss = best
    return model


def generate_greedy_prompted(model: CharGRU, vocab, prompt: str,
                             max_len: int = 256) -> str:
    """Prime ONCE on EOS+prompt (the last position's logits are the first
    prediction), then greedy-decode exactly like generate_greedy."""
    out = []
    with torch.no_grad():
        logits, h = model(one_hot([vocab.eos_id] + vocab.encode(prompt), len(vocab)))
        current = int(torch.argmax(logits[0, -1]).item())
        for _ in range(max_len):
            if current == vocab.eos_id:
                break
            out.append(vocab.decode([current]))
            logits, h = model(one_hot([current], len(vocab)), h)
            current = int(torch.argmax(logits[0, -1]).item())
    return "".join(out)
