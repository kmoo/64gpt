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


def _batchify_masked(seqs: list[list[int]], prompt_lens: list[int], vocab_size: int,
                     pad_target: int = -100):
    """Like _batchify, but ALSO masks every target position covering the
    primed prompt (docs/milestones/m7.md: "mask the loss over the primed
    prefix... otherwise the model spends capacity learning to echo the
    schema tags instead of learning voice"). Position i's target is
    ids[i], predicted from input up to position i; the prompt occupies
    ids[0:prompt_len), so only positions >= prompt_len (the response body
    + trailing EOS) are scored. M4 never needed this (flat 3-field
    prompts, no schema to accidentally memorize) — kept as a separate
    function so M4's own training path stays byte-for-byte unchanged."""
    B = len(seqs)
    T = max(len(s) + 1 for s in seqs)
    inputs = torch.zeros(B, T, vocab_size, dtype=torch.float32)
    targets = torch.full((B, T), pad_target, dtype=torch.long)
    for b, (ids, plen) in enumerate(zip(seqs, prompt_lens)):
        eos_first = [0] + ids
        for pos, i in enumerate(eos_first):
            inputs[b, pos, i] = 1.0
        full_targets = ids + [0]
        for pos in range(plen, len(full_targets)):
            targets[b, pos] = full_targets[pos]
    return inputs, targets


def train_corpus_conditioned(train_pairs: list[tuple[str, str]],
                             val_pairs: list[tuple[str, str]], vocab,
                             hidden: int = 256, seed: int = 0, lr: float = 3e-3,
                             batch_size: int = 64, max_epochs: int = 60,
                             patience: int = 5, device: str | None = None) -> CharGRU:
    """M7 training: prefix-loss masking (above) + an explicit combo-level
    train/val split supplied by the caller (m7.md: hold out whole
    conditioning combos, not just lines within seen combos — the actual
    test of generalizing, not memorizing). Otherwise identical to M4's
    train_corpus (early-stop on best val loss, restore that model)."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    model = CharGRU(vocab_size=len(vocab), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def encode_split(pairs):
        ids = [vocab.encode(p) + vocab.encode(r) for p, r in pairs]
        plens = [len(vocab.encode(p)) for p, _ in pairs]
        return ids, plens

    train_ids, train_plens = encode_split(train_pairs)
    val_ids, val_plens = encode_split(val_pairs)
    assert vocab.eos_id == 0

    def val_loss() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                inputs, targets = _batchify_masked(
                    val_ids[i:i + batch_size], val_plens[i:i + batch_size], len(vocab))
                logits, _ = model(inputs.to(device))
                n = (targets != -100).sum().item()
                total += loss_fn(logits.reshape(-1, len(vocab)),
                                 targets.reshape(-1).to(device)).item() * n
                count += n
        model.train()
        return total / count if count else float("inf")

    rng = torch.Generator().manual_seed(seed)
    best, best_state, since_best = float("inf"), None, 0
    for epoch in range(max_epochs):
        order = torch.randperm(len(train_ids), generator=rng).tolist()
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch_ids = [train_ids[j] for j in idx]
            batch_plens = [train_plens[j] for j in idx]
            inputs, targets = _batchify_masked(batch_ids, batch_plens, len(vocab))
            opt.zero_grad()
            logits, _ = model(inputs.to(device))
            loss = loss_fn(logits.reshape(-1, len(vocab)),
                           targets.reshape(-1).to(device))
            loss.backward()
            # RNN gradients can explode on a single bad batch (M9's first
            # H=320 run: val loss jumped 18x uniformly across the whole
            # held-out set in one epoch -- the signature of a corrupting
            # Adam step, not a data artifact, since a bad batch would only
            # perturb the specific conditions it touched, not everything).
            # Clipping is a no-op when gradients are already small (M7/M8's
            # smaller, cleaner corpora never needed it), so this doesn't
            # change their existing determinism.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def _pow2_shift_t(t: torch.Tensor) -> int:
    """Torch twin of quantize.pow2_shift: largest k in [0, 16] with
    round(max|t| * 2^k) <= 127. Must stay in lockstep with quantize.py —
    the QAT grid IS the export grid, or QAT trains against the wrong
    rounding."""
    m = float(t.detach().abs().max())
    if m == 0.0:
        return 14
    k = 0
    while k < 16 and round(m * 2 ** (k + 1)) <= 127:
        k += 1
    return k


def _fake_quant(w: torch.Tensor, k: int) -> torch.Tensor:
    """Round w onto the int8 grid at shift k and back to float, with the
    straight-through estimator: forward sees the rounded weights, backward
    updates the underlying floats. torch.round is half-to-even, matching
    np.round in quantize()."""
    scale = float(2 ** k)
    return w + ((w * scale).round() / scale - w).detach()


def qat_finetune(model: CharGRU, train_pairs: list[tuple[str, str]],
                 val_pairs: list[tuple[str, str]], vocab,
                 seed: int = 0, lr: float = 3e-4, batch_size: int = 64,
                 max_epochs: int = 30, patience: int = 6,
                 device: str | None = None) -> CharGRU:
    """M12.1: quantization-aware fine-tuning after float convergence.

    Every forward pass runs with the weights FAKE-QUANTIZED onto the
    exact grid quantize() will export to (same pow2_shift k, shared k_w
    across W_ih/W_hh, per-tensor k_out on the head), so the optimizer
    converges to floats whose ROUNDED version behaves — instead of
    hoping the rounded version of a float optimum behaves. Rationale and
    the measurement that motivated this (int8 rounding tripled the
    invented-word rate): docs/ideas-coherence-rescue-plan.md.

    Mechanically this uses the weight_norm pattern, not parametrize:
    nn.GRU's fused kernels read _flat_weights, and RNNBase.__setattr__
    refreshes that list when the weight attributes are assigned — so a
    forward_pre_hook that re-derives the quantized weights from raw
    float parameters is the one approach that provably reaches the
    fused path. Biases and activations stay unquantized: their grids
    (int32 / int16 Q14) are ~64x finer than the weights' int8 and are
    not the dominant error term (measure first — extend only if the
    coherence probe says otherwise).

    Same loop shape as train_corpus_conditioned (prefix-masked loss,
    combo-level val split, early-stop on best val, restore best); the
    val loss driving early-stop is the QUANTIZED forward's, which is
    the number that actually predicts shipped behavior. Returns the
    model with plain float weights restored (best raw state), ready for
    quantize()."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    model = model.to(device)
    model.train()

    gru, head = model.gru, model.head
    raw = {}
    for name in ("weight_ih_l0", "weight_hh_l0"):
        raw[name] = nn.Parameter(getattr(gru, name).detach().clone())
        delattr(gru, name)
        gru.register_parameter(name + "_raw", raw[name])
    raw["head"] = nn.Parameter(head.weight.detach().clone())
    del head.weight
    head.register_parameter("weight_raw", raw["head"])

    def quantize_gru_weights(module, inputs):
        k_w = min(_pow2_shift_t(raw["weight_ih_l0"]),
                  _pow2_shift_t(raw["weight_hh_l0"]))
        module.weight_ih_l0 = _fake_quant(raw["weight_ih_l0"], k_w)
        module.weight_hh_l0 = _fake_quant(raw["weight_hh_l0"], k_w)

    def quantize_head_weight(module, inputs):
        module.weight = _fake_quant(raw["head"], _pow2_shift_t(raw["head"]))

    hooks = [gru.register_forward_pre_hook(quantize_gru_weights),
             head.register_forward_pre_hook(quantize_head_weight)]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def encode_split(pairs):
        ids = [vocab.encode(p) + vocab.encode(r) for p, r in pairs]
        plens = [len(vocab.encode(p)) for p, _ in pairs]
        return ids, plens

    train_ids, train_plens = encode_split(train_pairs)
    val_ids, val_plens = encode_split(val_pairs)
    assert vocab.eos_id == 0

    def val_loss() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                inputs, targets = _batchify_masked(
                    val_ids[i:i + batch_size], val_plens[i:i + batch_size], len(vocab))
                logits, _ = model(inputs.to(device))
                n = (targets != -100).sum().item()
                total += loss_fn(logits.reshape(-1, len(vocab)),
                                 targets.reshape(-1).to(device)).item() * n
                count += n
        model.train()
        return total / count if count else float("inf")

    rng = torch.Generator().manual_seed(seed)
    best, best_state, since_best = float("inf"), None, 0
    for epoch in range(max_epochs):
        order = torch.randperm(len(train_ids), generator=rng).tolist()
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            inputs, targets = _batchify_masked([train_ids[j] for j in idx],
                                               [train_plens[j] for j in idx],
                                               len(vocab))
            opt.zero_grad()
            logits, _ = model(inputs.to(device))
            loss = loss_fn(logits.reshape(-1, len(vocab)),
                           targets.reshape(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        v = val_loss()
        print(f"qat epoch {epoch}: val loss (quantized forward) {v:.4f}", flush=True)
        if v < best:
            best, since_best = v, 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    for h in hooks:
        h.remove()
    # Restore plain float weights (best raw state) so quantize() sees the
    # standard CharGRU attribute layout.
    for name in ("weight_ih_l0", "weight_hh_l0"):
        w = getattr(gru, name + "_raw").detach().clone()
        delattr(gru, name + "_raw")
        if hasattr(gru, name):
            delattr(gru, name)
        gru.register_parameter(name, nn.Parameter(w))
    w = head.weight_raw.detach().clone()
    delattr(head, "weight_raw")
    if hasattr(head, "weight"):
        del head.weight
    head.register_parameter("weight", nn.Parameter(w))
    gru._init_flat_weights()

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
