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
    def __init__(self, vocab_size: int, hidden: int, input_size: int | None = None):
        """input_size defaults to vocab_size (every milestone before
        M12.3). M12.3's per-step D:/M: attribute conditioning passes a
        wider input_size (vocab_size + n_desc + n_mood): a learned
        embedding concatenated onto the char one-hot at every timestep
        is mathematically identical to appending MORE one-hot columns
        that feed the SAME weight_ih_l0 matrix (nn.Embedding(n, d)
        applied to id i selects row i, exactly what a linear layer does
        to a one-hot vector) -- so no separate embedding table or GRU
        subclass is needed, only a wider input and an unchanged head
        (predictions stay over the vocab_size char set only)."""
        super().__init__()
        self.gru = nn.GRU(input_size=input_size or vocab_size, hidden_size=hidden,
                          batch_first=True)
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


def one_hot_attr(ids: list[int], vocab_size: int, desc_id: int, n_desc: int,
                 mood_id: int, n_mood: int) -> torch.Tensor:
    """M12.3: one_hot() widened with two constant attribute columns, set
    at EVERY timestep (an NPC's voice/mood doesn't change mid-reply).
    Columns [0,V) = char one-hot, [V,V+n_desc) = D: one-hot,
    [V+n_desc, V+n_desc+n_mood) = M: one-hot. A learned embedding
    concatenated onto the input is mathematically identical to this: an
    nn.Embedding lookup IS a one-hot-times-weight-matrix select, so
    widening the one-hot and letting the existing weight_ih_l0 matrix
    learn the projection (via a plain CharGRU(..., input_size=vocab_size
    + n_desc + n_mood)) gives the same model with no new module, no
    rank-bottleneck attr_dim, and no new quantize()/qat_finetune() code
    path -- both already operate generically on whatever width
    weight_ih_l0 has."""
    width = vocab_size + n_desc + n_mood
    t = torch.zeros(1, len(ids), width, dtype=torch.float32)
    for pos, i in enumerate(ids):
        t[0, pos, i] = 1.0
        t[0, pos, vocab_size + desc_id] = 1.0
        t[0, pos, vocab_size + n_desc + mood_id] = 1.0
    return t


def generate_greedy_prompted_attr(model: CharGRU, vocab, prompt: str,
                                  desc_id: int, n_desc: int, mood_id: int, n_mood: int,
                                  max_len: int = 256) -> str:
    """generate_greedy_prompted's shape, widened for attribute columns --
    torch-side sanity check during training (NOT the shipped gate; the
    coherence probe measures the quantized model via ref_impl, same as
    every other milestone)."""
    V = len(vocab)
    out = []
    with torch.no_grad():
        x = one_hot_attr([vocab.eos_id] + vocab.encode(prompt), V,
                         desc_id, n_desc, mood_id, n_mood)
        logits, h = model(x)
        current = int(torch.argmax(logits[0, -1]).item())
        for _ in range(max_len):
            if current == vocab.eos_id:
                break
            out.append(vocab.decode([current]))
            x = one_hot_attr([current], V, desc_id, n_desc, mood_id, n_mood)
            logits, h = model(x, h)
            current = int(torch.argmax(logits[0, -1]).item())
    return "".join(out)


def _batchify_masked_attr(seqs: list[list[int]], prompt_lens: list[int],
                          desc_ids: list[int], n_desc: int,
                          mood_ids: list[int], n_mood: int, vocab_size: int,
                          pad_target: int = -100):
    """_batchify_masked widened with constant per-sequence attribute
    columns (one_hot_attr's convention) -- targets are unaffected (still
    over vocab_size only; attributes condition generation, they are
    never predicted)."""
    B = len(seqs)
    T = max(len(s) + 1 for s in seqs)
    width = vocab_size + n_desc + n_mood
    inputs = torch.zeros(B, T, width, dtype=torch.float32)
    targets = torch.full((B, T), pad_target, dtype=torch.long)
    for b, (ids, plen, desc_id, mood_id) in enumerate(
            zip(seqs, prompt_lens, desc_ids, mood_ids)):
        eos_first = [0] + ids
        for pos, i in enumerate(eos_first):
            inputs[b, pos, i] = 1.0
            inputs[b, pos, vocab_size + desc_id] = 1.0
            inputs[b, pos, vocab_size + n_desc + mood_id] = 1.0
        full_targets = ids + [0]
        for pos in range(plen, len(full_targets)):
            targets[b, pos] = full_targets[pos]
    return inputs, targets


def train_corpus_conditioned_attr(train_pairs: list[tuple[str, str]],
                                  val_pairs: list[tuple[str, str]],
                                  train_attrs: list[tuple[int, int]],
                                  val_attrs: list[tuple[int, int]],
                                  vocab, n_desc: int, n_mood: int,
                                  hidden: int = 256, seed: int = 0, lr: float = 3e-3,
                                  batch_size: int = 64, max_epochs: int = 60,
                                  patience: int = 5,
                                  device: str | None = None,
                                  checkpoint_path: str | None = None) -> CharGRU:
    """train_corpus_conditioned's shape (prefix-loss masking, combo-level
    val split, early-stop on best val, restore best), widened input via
    one_hot_attr/_batchify_masked_attr instead of one_hot/_batchify_masked.
    train_attrs/val_attrs are parallel (desc_id, mood_id) lists, one pair
    per train_pairs/val_pairs entry -- the caller resolves prompt ->
    (desc_id, mood_id) via npc_service.parse_prompt_fields() + a
    corpus-built vocab, once, outside this function (keeps this function
    corpus-agnostic, like train_corpus_conditioned itself).

    checkpoint_path: if set, best_state is ALSO written to disk (not just
    kept in memory) every time a new best is found -- protects a long run
    against the process itself dying (a real GPU/Metal command-buffer OOM
    killed M12.5's ~3-hour float phase mid-run; it happened to recover
    only because the corruption manifested as NaN rather than a hard
    kill). The in-memory best_state/restore-at-the-end behavior is
    unchanged either way; this only adds a recovery path for a crash
    mid-run. Caller's job to load it back (torch.load) if resuming."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    V = len(vocab)
    model = CharGRU(vocab_size=V, hidden=hidden,
                    input_size=V + n_desc + n_mood).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def encode_split(pairs):
        ids = [vocab.encode(p) + vocab.encode(r) for p, r in pairs]
        plens = [len(vocab.encode(p)) for p, _ in pairs]
        return ids, plens

    train_ids, train_plens = encode_split(train_pairs)
    val_ids, val_plens = encode_split(val_pairs)
    train_desc = [d for d, _ in train_attrs]
    train_mood = [m for _, m in train_attrs]
    val_desc = [d for d, _ in val_attrs]
    val_mood = [m for _, m in val_attrs]
    assert vocab.eos_id == 0

    def val_loss() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                inputs, targets = _batchify_masked_attr(
                    val_ids[i:i + batch_size], val_plens[i:i + batch_size],
                    val_desc[i:i + batch_size], n_desc,
                    val_mood[i:i + batch_size], n_mood, V)
                logits, _ = model(inputs.to(device))
                n = (targets != -100).sum().item()
                total += loss_fn(logits.reshape(-1, V),
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
            inputs, targets = _batchify_masked_attr(
                [train_ids[j] for j in idx], [train_plens[j] for j in idx],
                [train_desc[j] for j in idx], n_desc,
                [train_mood[j] for j in idx], n_mood, V)
            opt.zero_grad()
            logits, _ = model(inputs.to(device))
            loss = loss_fn(logits.reshape(-1, V),
                           targets.reshape(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        v = val_loss()
        print(f"epoch {epoch}: val loss {v:.4f}", flush=True)
        if v < best:
            best, since_best = v, 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save({"state": best_state, "val_loss": best,
                            "epoch": epoch, "hidden": hidden,
                            "n_desc": n_desc, "n_mood": n_mood},
                           checkpoint_path)
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    model.final_loss = best
    return model


def qat_finetune_attr(model: CharGRU, train_pairs: list[tuple[str, str]],
                      val_pairs: list[tuple[str, str]],
                      train_attrs: list[tuple[int, int]],
                      val_attrs: list[tuple[int, int]], vocab,
                      n_desc: int, n_mood: int,
                      seed: int = 0, lr: float = 3e-4, batch_size: int = 64,
                      max_epochs: int = 30, patience: int = 6,
                      device: str | None = None,
                      checkpoint_path: str | None = None) -> CharGRU:
    """qat_finetune, widened input via _batchify_masked_attr. The hook
    logic is UNCHANGED from qat_finetune: weight_ih_l0 already covers the
    D:/M: one-hot columns (they're just more columns of the same matrix,
    see one_hot_attr), so fake-quantizing it at its own pow2_shift
    automatically keeps the attribute columns on the same int8 grid as
    the char columns and W_hh -- no separate embedding table, no second
    k, nothing extra to keep in lockstep. Kept as its own function (not a
    parameter added to qat_finetune) only because the batching differs,
    matching this file's established convention of a new function per
    input-shape change rather than widening an already-tested one.

    checkpoint_path: see train_corpus_conditioned_attr's docstring --
    same on-disk recovery path for the QAT phase, not just the float
    phase (M12.5's real crash was in the float phase, but QAT runs are
    not immune to the same class of process-level failure)."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    V = len(vocab)
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
        k_w = min(_pow2_shift_t(raw["weight_ih_l0"]), _pow2_shift_t(raw["weight_hh_l0"]))
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
    train_desc = [d for d, _ in train_attrs]
    train_mood = [m for _, m in train_attrs]
    val_desc = [d for d, _ in val_attrs]
    val_mood = [m for _, m in val_attrs]
    assert vocab.eos_id == 0

    def val_loss() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                inputs, targets = _batchify_masked_attr(
                    val_ids[i:i + batch_size], val_plens[i:i + batch_size],
                    val_desc[i:i + batch_size], n_desc,
                    val_mood[i:i + batch_size], n_mood, V)
                logits, _ = model(inputs.to(device))
                n = (targets != -100).sum().item()
                total += loss_fn(logits.reshape(-1, V),
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
            inputs, targets = _batchify_masked_attr(
                [train_ids[j] for j in idx], [train_plens[j] for j in idx],
                [train_desc[j] for j in idx], n_desc,
                [train_mood[j] for j in idx], n_mood, V)
            opt.zero_grad()
            logits, _ = model(inputs.to(device))
            loss = loss_fn(logits.reshape(-1, V),
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
            if checkpoint_path is not None:
                torch.save({"state": best_state, "val_loss": best,
                            "epoch": epoch, "n_desc": n_desc,
                            "n_mood": n_mood}, checkpoint_path)
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    for h in hooks:
        h.remove()
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


FILM_SCALE = 0.2
"""Bounds CharGRUFiLM's gamma to (1-FILM_SCALE, 1+FILM_SCALE) and beta to
(-FILM_SCALE, FILM_SCALE). Must stay in lockstep with ref_impl.FILM_SCALE_Q14
-- the ref_impl integer path re-derives its own Q14 constant from this
same value, they are not independently tunable."""


class CharGRUFiLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, n_attr: int):
        """M12.5 (option B, docs/ideas-m12.3-conditioning-strategies.md):
        feature-wise linear modulation applied to the hidden state INSIDE
        the recurrence, every timestep -- h_t = gamma*cell(x_t, h_{t-1}) +
        beta -- so the attribute signal can't decay across a long unroll
        the way M12.3/M12.4's one-shot input columns could (M12.3's
        redundant-columns result: 2.33 inv/line; M12.4's ablation: 1.44,
        still short of the <=1.0 gate). Needs nn.GRUCell + a manual
        per-step loop instead of nn.GRU, since gamma/beta must feed back
        into the NEXT step's hidden state, not just reweight the final
        output.

        gamma/beta are SQUASHED through tanh (gamma = 1+tanh(raw), beta =
        tanh(raw)) rather than left as a raw linear output. An early
        version without this bound trained fine in float (unconstrained
        gamma stayed near 1 for this toy corpus) but diverged catastrophically
        once quantized to int16 Q14: gamma feeds back into h every step, so
        even a mildly-off-1.0 gamma compounds multiplicatively across a long
        unroll and saturates the fixed-point range within a handful of
        steps (traced: h hit int16's floor after just 2 primed characters).
        tanh bounds gamma to (0, 2) unconditionally, which the raw-linear
        version could not guarantee. film is zero-initialized so both raw
        pre-tanh values are 0 -> gamma=1, beta=0 at the start of training
        (identity -- standard FiLM stability trick, Perez et al. 2017),
        letting training discover a useful (bounded) deviation instead of
        starting from a random reshaping of every hidden unit.

        Bounding gamma to (0,2) alone is NOT enough: gamma feeds back into
        h every step, so even a gamma of 1.4-1.6 (well inside that bound,
        and what an early version actually trained to) compounds
        multiplicatively across a long unroll -- 1.46 squared is already
        past 2x, enough to blow through int16's range in 2 steps (traced:
        saturated after priming just 2 characters, before FILM_SCALE was
        added). FILM_SCALE further squashes the perturbation to (1-scale,
        1+scale) / (-scale, scale), trading modulation strength for
        stability -- the amount of per-step drift the fixed-point hidden
        state can absorb over a realistic reply length without saturating."""
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(vocab_size, hidden)
        self.film = nn.Linear(n_attr, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x, attr, h=None):
        """x: [B,T,V] one-hot char input, batch_first (same convention as
        CharGRU). attr: [B, n_attr] one-hot attribute vector, CONSTANT
        across the sequence (an NPC's voice/mood doesn't change mid-reply,
        same assumption as one_hot_attr). Returns (logits [B,T,V], h)."""
        B, T, _ = x.shape
        if h is None:
            h = x.new_zeros(B, self.hidden)
        gamma_beta_raw = self.film(attr)
        gamma_raw = gamma_beta_raw[:, :self.hidden]
        beta_raw = gamma_beta_raw[:, self.hidden:]
        gamma = 1.0 + FILM_SCALE * torch.tanh(gamma_raw)
        beta = FILM_SCALE * torch.tanh(beta_raw)
        outs = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            h = gamma * h + beta
            outs.append(h)
        out = torch.stack(outs, dim=1)
        return self.head(out), h


def one_hot_attr_vec(desc_id: int, n_desc: int, mood_id: int, n_mood: int) -> torch.Tensor:
    """[1, n_desc+n_mood] one-hot attribute vector for CharGRUFiLM's film
    input. Unlike one_hot_attr (M12.3), this is NOT concatenated onto the
    char input at every timestep -- it's consumed once per generation by
    film to produce gamma/beta, and THOSE are what gets reapplied every
    step (see CharGRUFiLM.forward)."""
    t = torch.zeros(1, n_desc + n_mood, dtype=torch.float32)
    t[0, desc_id] = 1.0
    t[0, n_desc + mood_id] = 1.0
    return t


def generate_greedy_prompted_film(model: CharGRUFiLM, vocab, prompt: str,
                                  desc_id: int, n_desc: int, mood_id: int, n_mood: int,
                                  max_len: int = 256) -> str:
    """generate_greedy_prompted's shape for CharGRUFiLM -- torch-side
    sanity check during training (NOT the shipped gate; the coherence
    probe measures the quantized model via ref_impl, same as every other
    milestone)."""
    V = len(vocab)
    attr = one_hot_attr_vec(desc_id, n_desc, mood_id, n_mood)
    out = []
    with torch.no_grad():
        x = one_hot([vocab.eos_id] + vocab.encode(prompt), V)
        logits, h = model(x, attr)
        current = int(torch.argmax(logits[0, -1]).item())
        for _ in range(max_len):
            if current == vocab.eos_id:
                break
            out.append(vocab.decode([current]))
            x = one_hot([current], V)
            logits, h = model(x, attr, h)
            current = int(torch.argmax(logits[0, -1]).item())
    return "".join(out)


def _batchify_masked_film(seqs: list[list[int]], prompt_lens: list[int],
                          desc_ids: list[int], n_desc: int,
                          mood_ids: list[int], n_mood: int, vocab_size: int,
                          pad_target: int = -100):
    """_batchify_masked's char-input/target shape, UNCHANGED (FiLM's char
    input is plain one_hot, not widened -- unlike M12.3/M12.4's option A),
    plus a separate constant per-sequence attribute one-hot vector for
    film's input."""
    B = len(seqs)
    T = max(len(s) + 1 for s in seqs)
    inputs = torch.zeros(B, T, vocab_size, dtype=torch.float32)
    attrs = torch.zeros(B, n_desc + n_mood, dtype=torch.float32)
    targets = torch.full((B, T), pad_target, dtype=torch.long)
    for b, (ids, plen, desc_id, mood_id) in enumerate(
            zip(seqs, prompt_lens, desc_ids, mood_ids)):
        eos_first = [0] + ids
        for pos, i in enumerate(eos_first):
            inputs[b, pos, i] = 1.0
        attrs[b, desc_id] = 1.0
        attrs[b, n_desc + mood_id] = 1.0
        full_targets = ids + [0]
        for pos in range(plen, len(full_targets)):
            targets[b, pos] = full_targets[pos]
    return inputs, attrs, targets


def train_corpus_conditioned_film(train_pairs: list[tuple[str, str]],
                                  val_pairs: list[tuple[str, str]],
                                  train_attrs: list[tuple[int, int]],
                                  val_attrs: list[tuple[int, int]],
                                  vocab, n_desc: int, n_mood: int,
                                  hidden: int = 256, seed: int = 0, lr: float = 3e-3,
                                  batch_size: int = 64, max_epochs: int = 60,
                                  patience: int = 5,
                                  device: str | None = None) -> CharGRUFiLM:
    """train_corpus_conditioned_attr's shape (prefix-loss masking,
    combo-level val split, early-stop/restore-best), for CharGRUFiLM
    (M12.5, option B): gamma/beta modulate the hidden state every step
    instead of widening the input (option A, M12.3/M12.4)."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    V = len(vocab)
    model = CharGRUFiLM(vocab_size=V, hidden=hidden, n_attr=n_desc + n_mood).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def encode_split(pairs):
        ids = [vocab.encode(p) + vocab.encode(r) for p, r in pairs]
        plens = [len(vocab.encode(p)) for p, _ in pairs]
        return ids, plens

    train_ids, train_plens = encode_split(train_pairs)
    val_ids, val_plens = encode_split(val_pairs)
    train_desc = [d for d, _ in train_attrs]
    train_mood = [m for _, m in train_attrs]
    val_desc = [d for d, _ in val_attrs]
    val_mood = [m for _, m in val_attrs]
    assert vocab.eos_id == 0

    def val_loss() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                inputs, attrs, targets = _batchify_masked_film(
                    val_ids[i:i + batch_size], val_plens[i:i + batch_size],
                    val_desc[i:i + batch_size], n_desc,
                    val_mood[i:i + batch_size], n_mood, V)
                logits, _ = model(inputs.to(device), attrs.to(device))
                n = (targets != -100).sum().item()
                total += loss_fn(logits.reshape(-1, V),
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
            inputs, attrs, targets = _batchify_masked_film(
                [train_ids[j] for j in idx], [train_plens[j] for j in idx],
                [train_desc[j] for j in idx], n_desc,
                [train_mood[j] for j in idx], n_mood, V)
            opt.zero_grad()
            logits, _ = model(inputs.to(device), attrs.to(device))
            loss = loss_fn(logits.reshape(-1, V),
                           targets.reshape(-1).to(device))
            loss.backward()
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


def qat_finetune_film(model: CharGRUFiLM, train_pairs: list[tuple[str, str]],
                      val_pairs: list[tuple[str, str]],
                      train_attrs: list[tuple[int, int]],
                      val_attrs: list[tuple[int, int]], vocab,
                      n_desc: int, n_mood: int,
                      seed: int = 0, lr: float = 3e-4, batch_size: int = 64,
                      max_epochs: int = 30, patience: int = 6,
                      device: str | None = None) -> CharGRUFiLM:
    """qat_finetune's shape for CharGRUFiLM: fake-quantizes cell.weight_ih/
    weight_hh and head.weight exactly like qat_finetune/qat_finetune_attr,
    PLUS a new hook for film.weight -- M12.5's only new quantized tensor.
    gamma/beta ride the SAME int8-weight + Q14-fixed-point machinery as
    everything else (see ref_impl.film_gamma_beta), just a new
    application site rather than a new number format."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    V = len(vocab)
    model = model.to(device)
    model.train()

    cell, head, film = model.cell, model.head, model.film
    raw = {}
    for name in ("weight_ih", "weight_hh"):
        raw[name] = nn.Parameter(getattr(cell, name).detach().clone())
        delattr(cell, name)
        cell.register_parameter(name + "_raw", raw[name])
    raw["head"] = nn.Parameter(head.weight.detach().clone())
    del head.weight
    head.register_parameter("weight_raw", raw["head"])
    raw["film"] = nn.Parameter(film.weight.detach().clone())
    del film.weight
    film.register_parameter("weight_raw", raw["film"])

    def quantize_cell_weights(module, inputs):
        k_w = min(_pow2_shift_t(raw["weight_ih"]), _pow2_shift_t(raw["weight_hh"]))
        module.weight_ih = _fake_quant(raw["weight_ih"], k_w)
        module.weight_hh = _fake_quant(raw["weight_hh"], k_w)

    def quantize_head_weight(module, inputs):
        module.weight = _fake_quant(raw["head"], _pow2_shift_t(raw["head"]))

    def quantize_film_weight(module, inputs):
        module.weight = _fake_quant(raw["film"], _pow2_shift_t(raw["film"]))

    hooks = [cell.register_forward_pre_hook(quantize_cell_weights),
             head.register_forward_pre_hook(quantize_head_weight),
             film.register_forward_pre_hook(quantize_film_weight)]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def encode_split(pairs):
        ids = [vocab.encode(p) + vocab.encode(r) for p, r in pairs]
        plens = [len(vocab.encode(p)) for p, _ in pairs]
        return ids, plens

    train_ids, train_plens = encode_split(train_pairs)
    val_ids, val_plens = encode_split(val_pairs)
    train_desc = [d for d, _ in train_attrs]
    train_mood = [m for _, m in train_attrs]
    val_desc = [d for d, _ in val_attrs]
    val_mood = [m for _, m in val_attrs]
    assert vocab.eos_id == 0

    def val_loss() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_ids), batch_size):
                inputs, attrs, targets = _batchify_masked_film(
                    val_ids[i:i + batch_size], val_plens[i:i + batch_size],
                    val_desc[i:i + batch_size], n_desc,
                    val_mood[i:i + batch_size], n_mood, V)
                logits, _ = model(inputs.to(device), attrs.to(device))
                n = (targets != -100).sum().item()
                total += loss_fn(logits.reshape(-1, V),
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
            inputs, attrs, targets = _batchify_masked_film(
                [train_ids[j] for j in idx], [train_plens[j] for j in idx],
                [train_desc[j] for j in idx], n_desc,
                [train_mood[j] for j in idx], n_mood, V)
            opt.zero_grad()
            logits, _ = model(inputs.to(device), attrs.to(device))
            loss = loss_fn(logits.reshape(-1, V),
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
    for name in ("weight_ih", "weight_hh"):
        w = getattr(cell, name + "_raw").detach().clone()
        delattr(cell, name + "_raw")
        if hasattr(cell, name):
            delattr(cell, name)
        cell.register_parameter(name, nn.Parameter(w))
    w = head.weight_raw.detach().clone()
    delattr(head, "weight_raw")
    if hasattr(head, "weight"):
        del head.weight
    head.register_parameter("weight", nn.Parameter(w))
    w = film.weight_raw.detach().clone()
    delattr(film, "weight_raw")
    if hasattr(film, "weight"):
        del film.weight
    film.register_parameter("weight", nn.Parameter(w))

    model = model.to("cpu").eval()
    model.final_loss = best
    return model
