# 06 — The training pipeline (v1, M2)

Five stages, each gated by tests, ending in artifacts the console
replays bit-for-bit. Everything lives in `trainer/` (Python 3.12 +
PyTorch via uv; `uv run pytest` runs the 26-test suite).

## Why "overfit ONE line" is the right goal

M2's model memorizes exactly one sentence. That sounds useless — it's
deliberate. The milestone isn't ML quality; it's **the pipeline**:
train → quantize → integer reference → export → C engine → ROM
self-test, every joint proven. With one line, *correct* has a trivial
definition (the output equals the line, byte for byte), so any
divergence anywhere in the chain is unambiguous. Scaling data (M3, M4)
then changes only stage 1 — the proven joints stay frozen.

## The five stages

**1. Vocab** (`vocab.py`, `test_vocab.py`) — build the character table:
id 0 reserved for EOS, ids 1..V−1 the sorted unique corpus characters,
printable ASCII only. Tests prove round-trips (`decode(encode(s)) == s`,
bytes table ↔ vocab) and determinism — the same corpus must always
produce the same table, or committed blobs would churn.

**2. Float overfit** (`model.py`, `test_overfit.py`) — a one-layer
`nn.GRU` (H=32) plus linear head, teacher-forced on the single sequence
`EOS + line → line + EOS` until loss < 1e-3. Tests prove greedy decode
reproduces the line exactly and that a fixed seed gives identical
results. This is the only stage with floats — they never leave it.

**3. Quantize** (`quantize.py`, `test_quantize.py`) — floats → the
integer artifacts of `docs/04-fixed-point-inference.md`: int8 weights
with shared power-of-two shifts, int32 biases pre-scaled into the
accumulator domain, 256-entry σ/tanh LUTs. Tests bound every error
(weights within half a quantization step, LUTs within rounding error,
biases exact) so a quantizer regression can't hide.

**4. Integer reference** (`ref_impl.py`, `test_ref_impl.py`) — the
pipeline's keystone: GRU inference in integer-only NumPy, mirroring the
future C code operation for operation. The gate: **integer generation
must reproduce the training line byte-for-byte.** If quantization ate
too much precision, this test fails here, on the Mac, with a debugger —
not on the console with a blank screen. `trace()` records every step's
hidden state and argmax id as goldens for the C tests.

**5. Export** (`export.py`, `test_export.py`, `make_gru_blob.py`) —
serialize to the big-endian NGPT type-1 payload (layout:
`docs/milestones/m2.md`). Tests prove a full round-trip: parse the blob
back and the *reparsed* model still generates the line. The script then
emits every artifact in one run:

```
game/rawfs/model.bin           → packed into the ROM filesystem
tests/vectors/m2_gru.bin       → same blob, for the C host tests
tests/vectors/m2_expected.txt  → the expected bytes
tests/vectors/m2_trace.bin     → per-step goldens (ids + hidden states)
game/src/user/selftestGolden.h → expected text, baked into the ROM
```

## Why one script emits everything

`make_gru_blob.py` regenerates blob, goldens, and the ROM's self-test
header together and refuses to write anything if the integer reference
can't reproduce the corpus. Because no artifact is ever hand-edited or
generated separately, the blob and its expectations *cannot drift
apart* — a property the M1 canned pipeline established and every later
milestone inherits.

## M3: the conditioning stage

M3 adds one capability — *which* line comes out is chosen by a prompt —
without touching the pipeline's shape:

- **Corpus** (`corpus.py`): 12 hand-written `NPC/MOOD/EV` prompt→response
  pairs; the single source for training data, demo string tables, and
  goldens alike.
- **Prompted overfit** (`overfit_corpus`): one GRU (H=64) trained on all
  12 sequences (`EOS+prompt+response`) with a summed loss; training
  stops when every pair reproduces exactly under greedy decoding — the
  behavioral goal — with the loss threshold only as margin insurance.
- **Priming** (`ref_impl.prime`, mirrored by `ngpt_gru_prime` in C):
  consume the prompt with hidden-state updates only (no logits, nothing
  emitted), then generate as before. Same numbers, same shifts — the
  bit-exactness contract simply gained a prefix phase.
- **Artifacts** (`make_m3_blob.py`): refuses to emit unless the *integer*
  model reproduces all 12 from their prompts, then writes blob, 12
  per-prompt goldens, a priming-aware trace, and the ROM self-test
  header (now carrying prompts, goldens, and the demo's cycle tables).
- **Fast regression** (`test_m3_blob.py`): parses the *committed* blob
  and replays all 12 through the integer path — no training, seconds.
  Training-heavy tests are marked `slow` and skipped by default;
  milestone gates run `pytest -m ''` for the full suite.

## M4: the generalization stage

M4 is where the pipeline stops memorizing and starts *learning*. Two new
stages, everything else keeps its shape:

- **Generated corpus** (`corpus_gen.py`): a deterministic template
  grammar — per-NPC sentence skeletons with filler slots, mood-modulated
  punctuation and closers — yields 14,400 pairs (~1.5 MB), 762–1129
  *distinct* responses per condition. One `random.Random(seed)` in a
  fixed draw order makes the whole corpus reproducible byte-for-byte.
  Why a grammar and not scraped text? A rigid format is the mitigation
  for a ~70K-param model: it only has to learn voice + slots, not
  language.
- **Real training** (`model.train_corpus`): mini-batches with padding
  masks, a 90/10 train/val split taken along the corpus interleave (so
  every condition appears in both), early stopping on val loss with the
  best checkpoint restored. Float training remains throwaway
  scaffolding; nothing downstream depends on its numerics.
- **Sampling** (`ref_impl.sample_from_logits`, C twin
  `core/ngpt_sample.cpp`): temperature/top-k over the integer logits.
  Weights come from a 256-entry exp2 LUT (generated header — fixed math,
  not blob data), randomness from xorshift32. Seeded, so "random" output
  is still bit-exact: the goldens pin seed+params, and the ROM self-test
  replays them. Base-2 exponentials make "temperature" a base-2 knob
  (≈0.69× the natural-log convention) — documented, deliberate.
- **Acceptance** (`make_m4_blob.py` gates before emitting): int8-vs-float
  top-1 agreement ≥95% teacher-forced on held-out sequences — the metric
  that says quantization preserved the *decisions*, not just the values —
  plus non-degenerate seeded goldens per condition, printed for the
  human eyeball check.

## Downstream: how the C side consumes this

`tests/test_gru_model.cpp` loads `m2_gru.bin`, generates, and compares
against `m2_expected.txt` — then replays `m2_trace.bin` and checks the
hidden state after **every step** against the reference, all 32 int16
values, bit for bit. When that's green on the host, the ROM's boot
self-test (same engine, same blob, same golden) printing `SELFTEST PASS`
on the N64 is not a hope — it's arithmetic.
