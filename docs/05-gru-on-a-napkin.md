# 05 — A GRU on a napkin

You don't need an ML background for this project — you need this page.

## The problem shape

We want: given the characters typed so far, predict the next character.
Do that in a loop, feeding each prediction back in, and you have a text
generator. A model that does this needs *memory* — "the last letter was
`Q`, so `U` is likely" requires remembering the `Q`.

A **recurrent neural network** (RNN) is the smallest architecture with
that property. It keeps a **hidden state** `h` — a vector of `H` numbers
(ours: `H = 32`) — and updates it once per character:

```
h' = f(h, x)        x = current character, h' = new memory
```

Everything the network "knows" about the text so far must fit in those
32 numbers. The output side is one more matrix that turns `h` into a
score (**logit**) per character; the highest score is the prediction.

## The GRU cell

A plain RNN forgets too fast (each update overwrites memory). The
**Gated Recurrent Unit** fixes this with *gates* — small learned dials,
each a number in (0, 1), one per hidden dimension:

```
r = σ(W_ir·x + b_ir + W_hr·h + b_hr)     reset gate
z = σ(W_iz·x + b_iz + W_hz·h + b_hz)     update gate
n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))   candidate memory
h' = (1 − z) ⊙ n + z ⊙ h                 blend old and new
```

(`σ` squashes to (0,1), `tanh` to (−1,1), `⊙` is elementwise multiply.)

Read it bottom-up: `h'` is a **per-dimension crossfade** between the old
memory `h` and a freshly proposed memory `n`, with the update gate `z`
as the slider. `z ≈ 1` means "keep what I knew", `z ≈ 0` means "overwrite
with the new". The reset gate `r` decides how much of the old memory the
*proposal* itself gets to look at. That's the whole trick: the network
learns *when* to remember and when to forget, instead of always doing a
fixed amount of both.

## Why character-level, and why one-hot

Our vocabulary is characters, not words: id 0 is reserved for EOS (the
"generation over" token), ids 1..V−1 are the unique characters of the
corpus in sorted order. For M2's one line, `V = 14`.

The input `x` is **one-hot**: a length-`V` vector that is all zeros with
a single 1 at the current character's id. That makes the input-side
matrix products trivial — multiplying a matrix by a one-hot vector just
*selects a column*:

```
W_ir · x  ==  column x_id of W_ir      (no arithmetic at all)
```

On the N64 this means the entire input path costs a few array lookups.
This is also why there's no separate embedding layer: with a vocabulary
this small, the input weight columns *are* the embeddings.

## Parameter count (why the blob is 6.7 KB)

For `H = 32`, `V = 14`, PyTorch packs the three gates together:

| tensor | shape | params |
|---|---|---|
| `W_ih` (input → r,z,n) | 3·32 × 14 | 1,344 |
| `W_hh` (hidden → r,z,n) | 3·32 × 32 | 3,072 |
| `b_ih`, `b_hh` | 2 × 3·32 | 192 |
| `W_out` (head) | 14 × 32 | 448 |
| `b_out` | 14 | 14 |
| **total** | | **≈ 5,070** |

Quantized to int8, that's ~5 KB of weights; add int32 biases (768 B),
two 256-entry activation tables (1 KB), the charset table and headers,
and you get the 6,744-byte blob the ROM ships. The M4 target (~100K
params at `H ≈ 128`, `V ≈ 70`) flows through the exact same format —
only the dims fields change.

## Generation, greedily

Start with `h = 0` and `x = EOS`. Each step: update `h`, compute logits,
take the **argmax** (ties break toward the lowest id, deterministically),
emit that character, feed it back as the next `x`. Stop when argmax is
EOS. No randomness anywhere — which is what makes "the same bytes on
Mac, emulator, and console" a testable claim. Temperature and top-k
sampling arrive in M4, behind a seeded deterministic RNG for the same
reason.
