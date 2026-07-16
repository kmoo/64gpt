# 001 — M2 concept docs (04, 05, 06) drafts

## CONTRACT
- worker: qwen 14b (direct API)
- output: drafts of docs/04-fixed-point-inference.md,
  docs/05-gru-on-a-napkin.md, docs/06-training-pipeline.md
- audience: good software engineer, zero game-dev / zero embedded-ML
  background; explain why, not just what
- constraints (verbatim from CLAUDE.md): "core/ style: C-style C++.
  No floats, no heap, no exceptions/RTTI, no libdragon includes. Integer
  math only — this is what makes host tests prove N64 behavior." Blob is
  big-endian, parsed byte-by-byte. Docs must match docs/milestones/m2.md
  numerics exactly (Q14, k shifts, Q11 LUT input, round-half-up).
- verification: human review by lead (docs are not mechanically
  verifiable); numbers cross-checked against m2.md

## COMPLETION
- attempt 1: three drafts, too thin + LaTeX math (unrendered on GitHub)
- retry 1: doc 04 came back good (141 lines, applied; lead fixed one
  wrong param count 26K→~5K that traced back to the lead's own prompt);
  output truncated before docs 05/06
- retry 2: **degenerated — 1529 lines of "The"** (greedy temp-0
  repetition loop on a revise-with-inlined-text prompt). Lesson: for
  prose revision, prefer fresh generation over revision prompts, or
  nonzero temperature; degeneration risk grows with inlined prior text.
- escalation per MODE.md: lead wrote docs 05/06 directly. Final: 04
  qwen+review, 05/06 hand-written.
