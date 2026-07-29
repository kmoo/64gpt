#!/bin/sh
# Format only lines touched since a base ref, via .clang-format + git-clang-format.
# Deliberately NOT a repo-wide reformat: this codebase's hand-aligned enum/struct
# columns and hand-compacted core/ byte-offset parsing lines don't survive a
# full-file clang-format pass (see commit c8afd91), but new/touched lines can
# still be kept consistent without disturbing anything you didn't write.
#   scripts/format-diff.sh                 diff vs. main, print only (no changes)
#   scripts/format-diff.sh --apply         apply to working tree (unstaged files)
#   scripts/format-diff.sh --staged        format staged lines in place
#   scripts/format-diff.sh <ref>           diff vs. a specific ref instead of main
set -u
cd "$(dirname "$0")/.."

mode="diff"
ref="main"

for arg in "$@"; do
    case "$arg" in
        --apply) mode="apply" ;;
        --staged) mode="staged" ;;
        *) ref="$arg" ;;
    esac
done

case "$mode" in
    diff)
        git clang-format --diff "$ref"
        ;;
    apply)
        git clang-format --force "$ref"
        ;;
    staged)
        git clang-format --staged
        ;;
esac
