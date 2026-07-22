# ADR 0002 — No Debug/Release build type; a source-level switch gates the ROM's boot self-test instead

**Status:** Accepted (2026-07-18).

## Context

A natural question once a project has a real boot-time self-test: does
it run in a "debug" build and get stripped from a "release" build, the
way `assert()` does under `-DNDEBUG` in most C/C++ toolchains?

The answer, checked directly rather than assumed: **this toolchain has
no Debug/Release build-type concept at all.** `n64.mk` (the shared SDK
build rules included by every Pyrite64 project's generated Makefile)
compiles with one fixed flag set — `-O2`, no `-DNDEBUG`, no variant —
confirmed by grep, not by absence of evidence:

```
$ grep -n "N64_DEBUG\|NDEBUG\|BUILD_TYPE" "$PYRITE64_SDK/include/n64.mk"
(no output)
```

Meanwhile this project's boot self-test (`DialogueDemo.cpp`'s
`bootAdvance()`: a CPU-vs-RSP cross-check, then every `SELFTEST_COUNT`
golden replayed through the RSP path and byte-compared) is plain,
unconditional C++ — not an `assert()`-style macro, not wrapped in any
existing `#ifdef`. It runs identically on every ROM this toolchain
produces, at a real cost that scales with golden count (~7s/golden on
the RSP path — a 29-golden self-test, current as of M11's gossip
retrain, is 90-200+ seconds of black-screen-then-progress-counter before
the scene is playable). That cost is the entire point on a build meant
to *prove* something (every milestone doc's `SELFTEST PASS`/`XCHK PASS`
screenshot depends on it having actually run) — and pure waste on a
build meant to just be *played* (a convention-floor demo, a quick manual
check of unrelated UI work).

## Decision

Don't invent a project-wide Debug/Release build-type distinction (a new
`Makefile.custom` variable threaded through the Pyrite64 CLI, a second
`n64.mk`-level flag set) — that's more machinery than the one real
callsite that needs it justifies, and Pyrite64's CLI build invocation
(`pyrite64 --cli --cmd build project.p64proj`) wasn't verified to pass
arbitrary `make` variables through cleanly. Instead: a single source-level
`#define NGPT_SELFTEST_ENABLED` in `DialogueDemo.cpp` (default `1`),
exactly the same convention this file already uses for
`NGPT_ATTRACT_MODE` — a deliberate, documented, hand-edited toggle, not a
build-flag surface.

`NGPT_SELFTEST_ENABLED=1` (default, "DEBUG" in spirit): `bootAdvance()`'s
`BOOT_WAIT` phase proceeds through `BOOT_XCHK_CPU` → `BOOT_XCHK_RSP` →
`BOOT_SELFTEST` → `BOOT_READY`, unchanged from every prior milestone.
This is the build every `docs/milestones/mN.md` verification step and
every `versions/mN_64gpt.z64` snapshot uses — **never build a milestone
ROM with this off.**

`NGPT_SELFTEST_ENABLED=0` ("RELEASE" in spirit): `BOOT_WAIT` skips
straight to `BOOT_READY`, but still calls `rspBackendInit(&model)` +
`ngpt_set_matvec(rspMatvec)` first — the RSP backend install is real
inference setup the game needs to run fast, not just a verification
step, so a RELEASE build is not slower to *play*, only faster to *boot*.
`draw()`'s two status lines (`SELFTEST PASS/FAIL`, `XCHK PASS/FAIL`) are
gated the same way, printing `SELFTEST SKIPPED (RELEASE BUILD)` /
`XCHK SKIPPED (RELEASE)` instead of a stale `false`-initialized PASS/FAIL
— the one failure mode this ADR is careful to avoid is a RELEASE build
silently claiming a verification result it never ran.

## Consequences

**Positive**: a fast-boot ROM is now one source edit away
(`NGPT_SELFTEST_ENABLED` 1→0, rebuild) for anyone who wants to hand
someone a ROM to just *play* — a convention demo, a quick sanity check of
an unrelated UI change — without waiting through an ever-growing golden
count every single boot.

**Negative / accepted**: this is a hand-edited source toggle, not a
build-time flag — building both a verified and an unverified ROM in the
same session means editing the `#define`, rebuilding, and remembering to
flip it back. That's a real ergonomic cost, accepted deliberately over
building out actual Makefile-variable plumbing for a switch that, as of
this ADR, has exactly one real use case.

**Binding rule for future Claude sessions and human contributors**:
`NGPT_SELFTEST_ENABLED` **must be `1`** for any ROM build that:
- gets boot-verified for a milestone's Definition of Done
- gets copied into `versions/mN_64gpt.z64`
- gets cited in a `docs/milestones/mN.md` write-up as "SELFTEST PASS" /
  "XCHK PASS"

If you find this define set to `0` in the working tree, that's either
someone's in-progress fast-boot demo build (check `git diff` before
assuming it's stray) or a mistake — flip it back to `1` before doing any
milestone verification work. This project's entire hardware-verification
discipline (`CLAUDE.md`: "Never let a solved problem quietly regress")
depends on the self-test actually running on the builds that claim it did.
