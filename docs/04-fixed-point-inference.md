# Fixed-Point Inference

## Introduction

Fixed-point arithmetic is a method of representing and manipulating numbers using integers. It is essential for achieving bit-exact inference across different platforms, especially when dealing with hardware constraints like the Nintendo 64 (N64). This document explains why integers are used, the Q14 format, shift-based scaling, rounding, saturation, and lookup table (LUT) activations.

## Why Integers?

Integers provide bit-exactness across different platforms. Floating-point arithmetic can lead to inconsistencies due to differences in FMA (Fused Multiply-Add) operations and rounding modes. By using integers, we ensure that the same operations yield identical results on a development Mac, Ares emulator, and the real N64.

## Q14 Format

The Q14 format is a fixed-point representation where 14 bits are used for the fractional part. This format is chosen for hidden states and gate outputs, allowing for a dynamic range of approximately -2 to 2 with a precision of 1/16384. The format is defined as:

- 1 bit for the sign
- 14 bits for the fractional part

This format provides a good balance between precision and range for the neural network's operations.

## Shift-Based Scales

Weights are quantized to int8 with power-of-two scales stored as shift amounts `k`. The quantization formula is:

```
W_q = round(W * 2^k), |W_q| <= 127
```

For the input-hidden and hidden-hidden weights (`W_ih` and `W_hh`), they share one `k` so their accumulators can add directly. The output head has its own `k_out`.

Biases are int32 and pre-scaled by the trainer into the accumulator scale `2^(k+14)`.

## Rounding Shift

Rescaling uses arithmetic right shift with round-half-up bias:

```
rescaled = (acc + (1 << (s-1))) / 2^s
```

C++20 defines the right shift on negative signed numbers as arithmetic, and NumPy matches this behavior, ensuring the same bits everywhere.

## Saturation

Elementwise Q14*Q14 products shift back by 14 and saturate to int16. This ensures that the results do not overflow and remain within the valid range.

## Lookup Table Activations

Sigmoid and tanh activations are implemented using 256-entry int16 lookup tables. The input is clamped to `[-8, 8)` in Q11 (1.0=2048), and the index is calculated as:

```
index = (x + 16384) / 128
```

The output is in Q14 format, with no interpolation. The maximum LUT error is approximately `3 * 10^-5`, which is irrelevant for an overfit model, and bit-exactness is more critical.

## Host Tests Prove Console Behavior

The host tests in the Python reference implementation (`ref_impl.py`) ensure that the integer-only NumPy inference matches the expected behavior. These tests are the 1:1 contract for the C port, and they must reproduce the line byte-for-byte. The C host tests compare against the committed goldens, and the ROM boot self-test replays generation against the same golden text, printing `SELFTEST PASS` if everything matches.

---

# GRU on a Napkin

## Introduction

A Gated Recurrent Unit (GRU) is a type of recurrent neural network (RNN) used for sequence modeling. It is designed to capture temporal dependencies in data, making it suitable for tasks like language modeling. This document explains what a GRU is, how it works, and why it is suitable for character-level modeling on the Nintendo 64.

## What is a GRU?

A GRU is a sequence model that maintains a hidden state as its memory. It uses gates to control the flow of information, allowing it to learn complex patterns in sequential data. The GRU has three gates: reset (r), update (z), and new gate (n).

## Gates Explained Intuitively

1. **Reset Gate (r)**: Determines how much of the previous hidden state should be passed to the new gate.
2. **Update Gate (z)**: Determines how much of the previous hidden state should be retained.
3. **New Gate (n)**: Computes a new candidate hidden state based on the input and the reset gate.

The equations for the GRU gates are:

```
r = σ(W_ir * x + b_ir + W_hr * h + b_hr)
z = σ(W_iz * x + b_iz + W_hz * h + b_hz)
n = tanh(W_in * x + b_in + r * (W_hn * h + b_hn))
h' = (1 - z) * n + z * h
```

## Why Char-Level?

Char-level modeling treats each character as a separate input. This approach is suitable for tasks like text generation, where the model needs to capture the nuances of individual characters to produce coherent text.

## One-Hot Input

For char-level modeling, the input is one-hot encoded. This means that each character is represented as a binary vector with a single 1 and the rest 0s. The input matrix-vector product (`W_i? * x`) is just a column lookup, which is free on the N64, eliminating the need for an embedding layer.

## Parameter Count Math

For a GRU with `H = 32` hidden units and `V = 14` (sorted unique printable-ASCII characters plus the EOS slot), the parameter count is ~5,000 (`W_ih` 3·32·14 = 1344, `W_hh` 3·32·32 = 3072, biases 192, head 462). With int8 weights, int32 biases, the two 256-entry LUTs and the charset table, the exported blob is 6,744 bytes — the M4 target (~100K params at `H≈128`) ships through the same format.

## Conclusion

The GRU is a powerful tool for sequence modeling, and its simplicity and efficiency make it suitable for character-level tasks on the Nintendo 64. By understanding the gates and their roles, we can appreciate how the GRU captures temporal dependencies in data.

---

# Training Pipeline v1

## Introduction

The training pipeline for the char-level GRU on the Nintendo 64 consists of five stages: building the vocabulary, overfitting the float GRU, quantizing the model, creating the reference implementation, and exporting the NGPT blob. This document explains each stage, what it proves, and why overfitting a single line is the correct M2 goal.

## Pipeline Stages

1. **Build Vocabulary**: Create a vocabulary of unique printable-ASCII characters, with id 0 reserved for EOS (End of Sequence). The vocabulary is sorted and unique.

2. **Overfit Float GRU**: Train a float GRU with `H = 32` on a single line 'HALT! WHO GOES THERE?' using teacher forcing. The goal is to achieve a loss of less than `1 * 10^-3`. Greedy decoding must reproduce the line byte-for-byte.

3. **Quantize to int8/LUTs**: Quantize the float GRU to int8 weights and create lookup tables for sigmoid and tanh activations.

4. **Ref Impl**: Implement the integer-only NumPy inference in `ref_impl.py`. This implementation must reproduce the line byte-for-byte, serving as the 1:1 contract for the C port.

5. **Export NGPT Blob**: Export the NGPT blob (big-endian, parsed byte-by-byte on the console) and per-step goldens (hidden states + argmax ids). The C tests compare against the committed goldens, and the ROM boot self-test replays generation against the same golden text, printing `SELFTEST PASS`.

## What Each Test Proves

- **Build Vocabulary**: Ensures that the vocabulary is correctly created and sorted.
- **Overfit Float GRU**: Proves that the float GRU can learn the specific line with high accuracy.
- **Quantize to int8/LUTs**: Ensures that the quantization process preserves the model's behavior.
- **Ref Impl**: Proves that the integer-only NumPy inference matches the expected behavior.
- **Export NGPT Blob**: Ensures that the exported blob and goldens are correct and consistent.

## Why Overfitting One Line?

Overfitting a single line is the correct M2 goal because it serves as a proof of concept. It demonstrates that the pipeline can produce a model that learns a specific pattern and can be quantized and ported to the N64 without losing bit-exactness. This approach is simpler and more focused than trying to learn a larger dataset, making it easier to verify the correctness of each stage.

## Quantize->Ref Impl->Goldens->C Chain

The quantization process converts the float model to int8 weights and LUTs. The reference implementation (`ref_impl.py`) verifies that the integer-only inference matches the expected behavior. The goldens are then created and used by the C tests to ensure bit-exactness. The `make_gru_blob.py` script keeps everything in sync, ensuring that the blob, goldens, and ROM self-test header are consistent.

## Conclusion

The training pipeline ensures that the char-level GRU model is correctly trained, quantized, and ported to the Nintendo 64. By following this pipeline, we can achieve bit-exact inference across different platforms, leveraging the power of fixed-point arithmetic and integer-only operations.
