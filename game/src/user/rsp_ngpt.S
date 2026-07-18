########################################################
# rsp_ngpt.S — 64GPT GRU matvec on the RSP (rspq overlay)
#
# One command:
#   0 NgptMatvec — 768 rows x 256 cols int8xint16 matvec, DMA-tiled in
#                  BOTH dimensions (K-chunked spike, see
#                  docs/spikes/rsp-matvec-ktile.md), bit-exact vs the
#                  CPU engine
#
# CHUNK-SIZE EXPERIMENT (2026-07-17, same day as the H=256/chunk=64
# hardware run): that run found the H_BUF DMA merge cut DMA op count
# ~31% (312->216) for ~0% speed change, redirecting the diagnosis away
# from DMA op count and toward DotRowChunk CALL count (3,072 vs the old
# kernel's 768) as the likely dominant cost. This build tests that
# directly: CHUNK doubled 64->128, which HALVES call count (768 x
# NUM_KCHUNKS=2 = 1,536, vs 3,072 at chunk=64) at the same H=256, same
# model, same goldens. tile_rows correspondingly halves (32->16) to
# keep W_TILE at the same 2,048B (tile_rows*chunk is the invariant, not
# either factor alone) -- OUT_TILE DMA count roughly doubles (24->48)
# as a side effect, a small absolute cost next to halving 1,536 calls.
#
# The filename MUST start with "rsp": n64.mk keys RSP-ucode assembly
# off that prefix. Hard-won lessons encoded below (full ledger in
# docs/spikes/rsp-matvec.md):
#  - vector shifts are multiplies: they CLOBBER the accumulator, so all
#    unpacking runs before any vmudh/vmadh
#  - lpv rotates lanes by addr&15: weights are unpacked from lqv
#    byte-pairs instead (even/odd split; the CPU pre-shuffles h once)
#  - lqv/sqv only reach the next 16B boundary: buffers are .align 4
#  - mfc2 .eN maps lane N to byte offset 2N via byteVectorElements
#
# K-CHUNKED GENERALIZATION (this spike): the H=256 kernel
# (docs/spikes/rsp-matvec-h256.md) kept the ENTIRE hidden state h
# resident in DMEM for the whole matvec (H_BUF/W16_BUF both scale as
# 2H bytes) — that's the wall docs/milestones/m9.md §6 formalizes as
# DMEM(H) = 2600 + 4H + 8192/H, unsafe past H=368. This kernel tiles the
# REDUCTION dimension (K = H, columns of W_hh) the same way the existing
# kernel already tiles the OUTPUT dimension (M = 3H, rows): h is
# streamed CHUNK elements at a time instead of held whole. H_BUF/
# W16_BUF shrink to a FIXED size (2*CHUNK bytes each) independent of H;
# only the outer loop trip-counts (derived from H) change.
#
# CHUNK = 128 elements (this build; was 64). Row width in W_TILE during
# any given K-chunk pass is CHUNK bytes (packed int8), NOT H bytes —
# the RSP DMA engine's native "rectangular portion of a larger image"
# support (rsp_dma.inc) pulls a TILE_ROWS x CHUNK sub-block directly out
# of the H-wide RDRAM matrix via a pitched (2D) transfer, so W_hh's true
# row-major layout never needs restructuring in RDRAM. TILE_ROWS =
# 2048/CHUNK = 16 (this build; was 32) is a CONSTANT regardless of H.
#
# H=256 for this build (H/CHUNK and 3H/TILE_ROWS both divide evenly:
# 256/128 = 2 K-chunks, 768/16 = 48 row-tiles, zero wasted DMA bytes).
#
# DMEM budget: W_TILE 2048 + W16_BUF 256 + H_BUF 256 + OUT_TILE 64 =
# 2624B .bss (was 2432B at chunk=64 — W16_BUF/H_BUF doubled with chunk,
# OUT_TILE halved with tile_rows; still independent of H, still well
# under the 4,096B budget and under the old kernel's 3,656B).
########################################################
#include <rsp_queue.inc>   /* already provides the rsp_dma.inc routines */

.data

RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand NgptMatvec, 16 # cmd, dram_w(768x256), dram_h, dram_out
RSPQ_EndOverlayHeader

RSPQ_EmptySavedState

.bss
    .align 4
W_TILE:  .space 2048         # 16 packed int8 rows x 128B (one K-chunk's columns) per DMA tile
    .align 4
W16_BUF: .space 256          # one row's chunk unpacked: 64 quads EVEN w, then ODD w
    .align 4
H_BUF:   .space 256          # h chunk, CPU-shuffled globally (even stream, then odd); this
                              # buffer holds only this K-chunk's 64-even + 64-odd slice
    .align 4
OUT_TILE: .space 64          # 16 int32 row partial-sums for the current row-tile,
                              # accumulated ACROSS the K-chunk loop (not reset per chunk)

.text

########################################################
# DotRowChunk: exact int32 PARTIAL dot product of one 128-wide K-chunk
# of a single row (contrast the old DotRow, which did a FULL H-wide row
# in one call — this does 1/NUM_KCHUNKS of a row; the caller accumulates
# across chunks into OUT_TILE).
#   in : t6 = DMEM addr of 128 packed int8 weights for this (row, chunk)
#        H_BUF holds this chunk's CPU-shuffled h (even 64, then odd 64)
#   out: t7 = int32 partial sum for this chunk
#   clobbers: t0,t1,t3,t5,t6,t8, $v01-$v05, ACC
#   preserves: s0-s7, t9, a0-a3
#
# vmudh/vmadh accumulate (S16xS16 product << 16) into the 48-bit ACC;
# per lane products stay far under the 48-bit ceiling regardless of
# chunk size (same exactness argument as the full-row kernel, just
# fewer terms summed per call) — no clamp, no rounding. Exact per-lane
# int32 = (ACC_HI << 16) | (u16)ACC_MD.
########################################################
    .func DotRowChunk
DotRowChunk:
    # Phase 1: unpack int8 -> int16 (all shifts BEFORE any MAC).
    # 8 iterations x 16 packed weights/iter = 128 weights/chunk.
    li t8, %lo(W16_BUF)
    li t3, 8
UnpackLoop:
    lqv $v04.e0, 0,t6
    vsra $v02, $v04, 8       # even-index weights (BE high byte), exact
    sqv $v02.e0, 0,t8
    vsll8 $v05, $v04, 8
    vsra $v05, $v05, 8       # odd-index weights, sign-extended
    sqv $v05.e0, 128,t8      # odd half starts at +128 (== new even-half size)
    addi t6, 16
    addi t8, 16
    addi t3, -1
    bnez t3, UnpackLoop
    nop

    # Phase 2: clean accumulation; first vmudh clears ACC.
    # EVEN half: 64 h-values -> 8 groups of 8 lanes (1 initial vmudh +
    # 7 more in EvenLoop).
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t8
    vmudh $v03, $v01, $v02
    li t3, 7
EvenLoop:
    addi t5, 16
    addi t8, 16
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t8
    vmadh $v03, $v01, $v02
    addi t3, -1
    bnez t3, EvenLoop
    nop
    # ODD half: 64 h-values -> 8 groups. H_BUF's odd half starts at byte
    # 128 (even half is 128B at this chunk size); W16_BUF's odd half is
    # also at 128 (see file header math).
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    li t3, 8
OddLoop:
    lqv $v01.e0, 128,t5
    lqv $v02.e0, 128,t8
    vmadh $v03, $v01, $v02
    addi t5, 16
    addi t8, 16
    addi t3, -1
    bnez t3, OddLoop
    nop

    vsar $v04, COP2_ACC_MD
    vsar $v05, COP2_ACC_HI

    li t7, 0
.macro FoldLane lane
    mfc2 t0, $v05.e\lane
    sll t0, 16
    mfc2 t1, $v04.e\lane
    andi t1, 0xFFFF
    add t0, t0, t1
    add t7, t7, t0
.endm
    FoldLane 0
    FoldLane 1
    FoldLane 2
    FoldLane 3
    FoldLane 4
    FoldLane 5
    FoldLane 6
    FoldLane 7
    jr ra
    nop
    .endfunc

########################################################
# NgptMatvec: 768 rows x 256 cols, 2D-tiled — 48 row-tiles (16 rows
# each) x 2 K-chunks (128 cols each) per row-tile.
#   a1 = RDRAM addr of 768x256 int8 weights, row-major, 8-byte aligned
#        (real row stride is 256B — the K-chunk DMA below reads a
#        16-row x 128-col RECTANGLE out of this via the RSP DMA engine's
#        native pitch support, not a restructured/pre-tiled layout)
#   a2 = RDRAM addr of shuffled h (512B: 128 evens then 128 odds, the
#        SAME global shuffle convention as the old kernel — chunking
#        slices 64 contiguous elements out of each half per K-chunk,
#        it does not require the CPU to shuffle per-chunk)
#   a3 = RDRAM addr of 768 int32 outputs (3072B)
#
# Register map (s0-s7 fully committed, matches the old kernel's
# allocation plus one — s6 is new):
#   s0 = DMA scratch: current transfer's RDRAM source address
#   s1 = row-tile base address into a1, advances by TILE_ROWS*H (4096)
#        after all K-chunks for a row-tile are done
#   s2 = h base address (a2), fixed for the whole call
#   s3 = output cursor into a3, advances by TILE_ROWS*4 (64) per tile
#   s4 = DMA scratch: DMEM destination address
#   s5 = row-in-tile counter, 0..15
#   s6 = K-chunk counter, 0..1
#   s7 = saved ra (jal clobbers ra = RSPQ_Loop)
########################################################
    .func NgptMatvec
NgptMatvec:
    move s7, ra
    move s1, a1
    move s2, a2
    move s3, a3

    li t9, 48                 # 48 row-tiles x 16 rows = 768
TileLoop:
    li s6, 0                  # K-chunk 0..1 (H=256 / CHUNK=128 = 2)
KChunkLoop:
    # W_TILE <- pitched 2D DMA: 16 rows x 128B, pitch 256B (H), starting
    # at this row-tile's base + this chunk's 128B column offset.
    sll t4, s6, 7              # t4 = chunk * 128 (byte column offset)
    add s0, s1, t4
    li s4, %lo(W_TILE)
    li t1, 256                 # pitch = H (real RDRAM row width)
    li t0, DMA_SIZE(128, 16)
    jal DMAIn
    nop

    # H_BUF <- this chunk's 64-even + 64-odd slice, ONE pitched 2D DMA
    # (same merge as the chunk=64 build: even/odd regions are exactly H
    # bytes apart in RDRAM, a valid 2D pitch, the same trick W_TILE uses
    # above; DMEM dest is linear, so height=2 writes H_BUF+0 then
    # H_BUF+128 contiguously, matching DotRowChunk's read layout).
    add s0, s2, t4
    li s4, %lo(H_BUF)
    li t1, 256                 # pitch = H (odd region starts H bytes after even)
    jal DMAIn
    li t0, DMA_SIZE(128, 2)

    li s5, 0                   # row-in-tile 0..15
RowLoop:
    sll t6, s5, 7               # row * 128 (row width in W_TILE this pass)
    addi t6, %lo(W_TILE)
    jal DotRowChunk
    nop
    sll t0, s5, 2
    addi t0, %lo(OUT_TILE)
    bnez s6, AccumulateRow       # kc==0: direct store; else: load+add+store
    nop
    sw t7, 0(t0)
    b RowLoopNext
    nop
AccumulateRow:
    lw t1, 0(t0)
    add t7, t7, t1
    sw t7, 0(t0)
RowLoopNext:
    addi s5, 1
    slti t0, s5, 16
    bnez t0, RowLoop
    nop

    addi s6, 1
    slti t0, s6, 2
    bnez t0, KChunkLoop
    nop

    li s4, %lo(OUT_TILE)
    move s0, s3
    jal DMAOut
    li t0, DMA_SIZE(64, 1)

    addi s1, 4096                # TILE_ROWS(16) * H(256)
    addi s3, 64                  # TILE_ROWS(16) * 4
    addi t9, -1
    bnez t9, TileLoop
    nop

    move ra, s7
    jr ra
    nop
    .endfunc
