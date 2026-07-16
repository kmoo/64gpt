# versions/ — built ROMs, one per milestone

Ready-to-run `.z64` images (EverDrive takes them as-is; see
`docs/hardware-checklist.md`). Each is the walking skeleton with a
progressively more real brain — together they tell the project's story
in cartridge form. All boot with an on-screen SELFTEST verdict; expect
~10s of black screen first on m4/m5-era ROMs (boot self-test).

| file | tag | what it proves |
|---|---|---|
| `m2_64gpt.z64` | `m2` | first neural net on N64 — int8 GRU, one memorized line, bit-exact vs the trainer |
| `m3_64gpt.z64` | `m3` | prompt conditioning — 12 lines, D-pad picks who speaks (rebuilt from the tag) |
| `m4_64gpt.z64` | `m4` | generalization — 1.5MB generated corpus, seeded temperature/top-k sampling, fresh lines per press |
| `m5_64gpt.z64` | `m5` | performance pass (v1.0-rc) — 9.8ms/char, sustained 60 chars/sec at 60 VPS, perf overlay on screen |
| `rsp_spike_64gpt.z64` | branch `spike/rsp-matvec` | the GRU matvec on the RSP: prints `RSP: G1+G2+G3 ALL PASS` + RSP-vs-CPU timings |

Rebuild any of them from source: check out the tag/branch, purge
`game/build/` and `game/filesystem/`, then the ROM build command in the
root README. The spike ROM additionally requires the spike branch's
`Makefile.custom` ucode hook (already on that branch).
