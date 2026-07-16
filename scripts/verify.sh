#!/bin/sh
# Verification gate: objective evidence that the tree is green, printed as
# a review-package summary. Exit nonzero on any failure.
#   scripts/verify.sh          host tests + trainer pytest
#   scripts/verify.sh --rom    additionally build the ROM (Pyrite64 toolchain)
set -u
cd "$(dirname "$0")/.."

echo "# Review package — $(date +%Y-%m-%d), HEAD $(git rev-parse --short HEAD)"
echo
echo "## Working tree"
git status --short
git diff --stat
echo

fail=0

echo "## Host tests (core/ integer engine)"
# rm build/ first: a stale sanitized CMake cache breaks the fresh configure
# (see CLAUDE.md — ASan deadlocks on this toolchain, sanitizers stay off)
rm -rf build
if cmake -B build tests -DNGPT_SANITIZE=OFF >/dev/null \
   && cmake --build build >/dev/null \
   && ctest --test-dir build --output-on-failure; then
    echo "HOST TESTS: PASS"
else
    echo "HOST TESTS: FAIL"
    fail=1
fi
echo

echo "## Trainer tests (pytest via uv)"
if (cd trainer && uv run pytest -q); then
    echo "TRAINER TESTS: PASS"
else
    echo "TRAINER TESTS: FAIL"
    fail=1
fi

if [ "${1:-}" = "--rom" ]; then
    echo
    echo "## ROM build"
    # Real path, not a symlink — see CLAUDE.md ROM build gotcha.
    if "$HOME/GitHub/Pyrite64-v0.4.0/Pyrite64.app/Contents/MacOS/pyrite64" \
        --cli --cmd build "$PWD/game/project.p64proj"; then
        echo "ROM BUILD: PASS (boot game/64gpt.z64 in Ares to confirm SELFTEST PASS)"
    else
        echo "ROM BUILD: FAIL"
        fail=1
    fi
fi

echo
if [ "$fail" -eq 0 ]; then echo "VERDICT: PASS"; else echo "VERDICT: FAIL"; fi
exit "$fail"
