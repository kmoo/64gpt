# .ai/ — engineering-lead-mode task artifacts

This project is developed with a two-tier AI workflow: a frontier model
acts as engineering lead (requirements, task contracts, verification
gating, risk-based review) while a small local model (Qwen2.5-Coder via
an OpenAI-compatible local server) authors routine implementation drafts.
Artifacts here — not conversation history — are the durable record of
that delegation.

- `tasks/NNN-slug.md` — one file per delegated task: the CONTRACT the
  worker received, the COMPLETION evidence, and a metrics footer
  (dispatches, retries, escalations, review defects).
- Verification evidence comes from `scripts/verify.sh`, which runs the
  project's objective checks (host tests, trainer pytest, optionally the
  ROM build) and prints a review-package summary.

Durable knowledge graduates from here into `docs/`; task files stay as
the audit trail. The machine-local mode flag (`.ai/lead-mode`) is
git-ignored.
