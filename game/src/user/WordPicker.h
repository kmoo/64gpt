#pragma once

// Word-picker selection validator (docs/ideas.md #7 "Player-composed
// prompts (word picker)": "a picker UI over topic/mood/place tokens
// from the training grammar, so the audience ASKS instead of watching a
// cycle"). The picker only ever offers tokens FROM a fixed,
// pre-validated list (not free text), so there's no vocabulary
// constraint logic to write here -- that's corpus_gates.py's job on the
// training side, a different layer entirely. What this header actually
// needs is safe bounds validation for player input selecting an INDEX
// into that list, so a malformed/out-of-range selection (garbled
// controller input, an off-by-one in a future picker UI) can never
// read out of bounds.
namespace WordPicker
{
  // Returns true and writes *outToken if index is a valid selection
  // into tokens[0..count). Returns false (outToken left untouched) for
  // any out-of-range index, including negative -- the caller decides
  // what to do with an invalid pick (ignore it, re-prompt), not this
  // header.
  inline bool resolveSelection(const char *const *tokens, int count, int index,
                                const char **outToken)
  {
    if (index < 0 || index >= count) return false;
    *outToken = tokens[index];
    return true;
  }
}
