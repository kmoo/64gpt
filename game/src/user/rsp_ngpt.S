########################################################
# rsp_ngpt.S — 64GPT GRU matvec on the RSP (rspq overlay)
#
# One command:
#   0 NgptMatvec — 768 rows x 256 cols int8xint16 matvec, DMA-tiled in
#                  BOTH dimensions (K-chunked spike, see
#                  docs/spikes/rsp-matvec-ktile.md), bit-exact vs the
#                  CPU engine (unverified on hardware as of this write —
#                  see that doc's Decision section for the gate plan)
#
# QUICK HARDWARE VALIDATION NOTE: this build targets H=256, NOT the
# H=320 the design doc's Decision section calls for. Reason: core/'s
# NGPT_GRU_MAX_HIDDEN cap (frozen at 256 since M2) rejects any H=320
# blob before the RSP kernel is ever exercised — a frozen-interface
# change that needs explicit sign-off before touching core/, separate
# from this spike. Retargeting to H=256 instead validates the K-tiling
# MECHANISM (DMA correctness, chunk-accumulation logic, real speed)
# against the exact model/goldens already shipping on main — a true
# apples-to-apples old-kernel-vs-new-kernel comparison at identical H,
# no core/ or game-side changes needed. H=320/512 remain the next step,
# gated on that separate sign-off.
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
# Design chosen (docs/spikes/rsp-matvec-ktile.md, "Design A": row-tile
# outer, K-chunk inner) over the textbook K-chunk-outer/RDRAM-round-trip
# alternative ("Design B") because this is a matvec (N=1) — B's benefit
# (reusing an H-chunk across many output columns) doesn't apply, so its
# extra RDRAM round-trip cost has nothing to buy back.
#
# CHUNK = 64 elements. Row width in W_TILE during any given K-chunk pass
# is CHUNK bytes (packed int8), NOT H bytes — the RSP DMA engine's
# native "rectangular portion of a larger image" support (rsp_dma.inc)
# pulls a TILE_ROWS x CHUNK sub-block directly out of the H-wide RDRAM
# matrix via a pitched (2D) transfer, so W_hh's true row-major layout
# never needs restructuring in RDRAM. TILE_ROWS = 2048/CHUNK = 32 is now
# a CONSTANT regardless of H (contrast the old kernel, where tile depth
# = 2048/H shrinks toward 1 as H grows — a second, quieter wall this
# generalization also removes).
#
# H=256 HARDCODED for THIS validation build (H/CHUNK and 3H/TILE_ROWS
# both divide evenly: 256/64 = 4 K-chunks, 768/32 = 24 row-tiles, zero
# wasted DMA bytes). H=320 (320/64=5, 960/32=30) and H=512 (512/64=8,
# 1536/32=48) are the same mechanical constant swap (H, NUM_KCHUNKS,
# NUM_ROW_TILES, tile-base stride) once the core/ cap is raised — see
# docs/spikes/rsp-matvec-ktile.md's Decision section.
#
# DMEM budget: W_TILE 2048 + W16_BUF 128 + H_BUF 128 + OUT_TILE 128 =
# 2432B .bss, independent of H by construction (unaffected by which H
# this particular build targets) — vs. 3,656B for the existing
# non-tiled H=256 kernel this is being A/B'd against
# (docs/spikes/rsp-matvec-h256.md). This is the headline result this
# spike exists to get a real number for.
########################################################
#include <rsp_queue.inc>   /* already provides the rsp_dma.inc routines */

.data

RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand NgptMatvec, 16 # cmd, dram_w(768x256), dram_h, dram_out
RSPQ_EndOverlayHeader

RSPQ_EmptySavedState

.bss
    .align 4
W_TILE:  .space 2048         # 32 packed int8 rows x 64B (one K-chunk's columns) per DMA tile
    .align 4
W16_BUF: .space 128          # one row's chunk unpacked: 32 quads EVEN w, then ODD w
    .align 4
H_BUF:   .space 128          # h chunk, CPU-shuffled globally (even stream, then odd); this
                              # buffer holds only this K-chunk's 32-even + 32-odd slice
    .align 4
OUT_TILE: .space 128         # 32 int32 row partial-sums for the current row-tile,
                              # accumulated ACROSS the K-chunk loop (not reset per chunk)

.text

########################################################
# DotRowChunk: exact int32 PARTIAL dot product of one 64-wide K-chunk
# of a single row (contrast the old DotRow, which did a FULL H-wide row
# in one call — this does 1/NUM_KCHUNKS of a row; the caller accumulates
# across chunks into OUT_TILE).
#   in : t6 = DMEM addr of 64 packed int8 weights for this (row, chunk)
#        H_BUF holds this chunk's CPU-shuffled h (even 32, then odd 32)
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
    # 4 iterations x 16 packed weights/iter = 64 weights/chunk.
    li t8, %lo(W16_BUF)
    li t3, 4
UnpackLoop:
    lqv $v04.e0, 0,t6
    vsra $v02, $v04, 8       # even-index weights (BE high byte), exact
    sqv $v02.e0, 0,t8
    vsll8 $v05, $v04, 8
    vsra $v05, $v05, 8       # odd-index weights, sign-extended
    sqv $v05.e0, 64,t8       # odd half starts at +64 (== new even-half size)
    addi t6, 16
    addi t8, 16
    addi t3, -1
    bnez t3, UnpackLoop
    nop

    # Phase 2: clean accumulation; first vmudh clears ACC.
    # EVEN half: 32 h-values -> 4 groups of 8 lanes (1 initial vmudh +
    # 3 more in EvenLoop).
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t8
    vmudh $v03, $v01, $v02
    li t3, 3
EvenLoop:
    addi t5, 16
    addi t8, 16
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t8
    vmadh $v03, $v01, $v02
    addi t3, -1
    bnez t3, EvenLoop
    nop
    # ODD half: 32 h-values -> 4 groups. H_BUF's odd half starts at byte
    # 64 (even half is 64B at this chunk size); W16_BUF's odd half is
    # also at 64 (see file header math).
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    li t3, 4
OddLoop:
    lqv $v01.e0, 64,t5
    lqv $v02.e0, 64,t8
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
# NgptMatvec: 768 rows x 256 cols, 2D-tiled — 24 row-tiles (32 rows
# each) x 4 K-chunks (64 cols each) per row-tile.
#   a1 = RDRAM addr of 768x256 int8 weights, row-major, 8-byte aligned
#        (real row stride is 256B — the K-chunk DMA below reads a
#        32-row x 64-col RECTANGLE out of this via the RSP DMA engine's
#        native pitch support, not a restructured/pre-tiled layout)
#   a2 = RDRAM addr of shuffled h (512B: 128 evens then 128 odds, the
#        SAME global shuffle convention as the old kernel — chunking
#        slices 32 contiguous elements out of each half per K-chunk,
#        it does not require the CPU to shuffle per-chunk)
#   a3 = RDRAM addr of 768 int32 outputs (3072B)
#
# Register map (s0-s7 fully committed, matches the old kernel's
# allocation plus one — s6 is new):
#   s0 = DMA scratch: current transfer's RDRAM source address
#   s1 = row-tile base address into a1, advances by TILE_ROWS*H (8192)
#        after all K-chunks for a row-tile are done
#   s2 = h base address (a2), fixed for the whole call
#   s3 = output cursor into a3, advances by TILE_ROWS*4 (128) per tile
#   s4 = DMA scratch: DMEM destination address
#   s5 = row-in-tile counter, 0..31
#   s6 = K-chunk counter, 0..3 (NEW vs. the non-chunked kernel)
#   s7 = saved ra (jal clobbers ra = RSPQ_Loop)
########################################################
    .func NgptMatvec
NgptMatvec:
    move s7, ra
    move s1, a1
    move s2, a2
    move s3, a3

    li t9, 24                 # 24 row-tiles x 32 rows = 768
TileLoop:
    li s6, 0                  # K-chunk 0..3 (H=256 / CHUNK=64 = 4)
KChunkLoop:
    # W_TILE <- pitched 2D DMA: 32 rows x 64B, pitch 256B (H), starting
    # at this row-tile's base + this chunk's 64B column offset.
    sll t4, s6, 6              # t4 = chunk * 64 (byte column offset)
    add s0, s1, t4
    li s4, %lo(W_TILE)
    li t1, 256                 # pitch = H (real RDRAM row width)
    # DMA_SIZE(64,32) doesn't fit a single 16-bit li (height=32 pushes
    # it past 0xFFFF) -- the li macro would expand to 2 instructions,
    # illegal in a branch-delay slot, so set t0 BEFORE the jal instead
    # of in its delay slot (unlike every other DMA call in this file).
    li t0, DMA_SIZE(64, 32)
    jal DMAIn
    nop

    # H_BUF <- this chunk's 32-even + 32-odd slice (two small linear
    # DMAs; global shuffle convention unchanged from the old kernel, so
    # even-region starts at s2+0, odd-region starts at s2+256 == H).
    add s0, s2, t4
    li s4, %lo(H_BUF)
    jal DMAIn
    li t0, DMA_SIZE(64, 1)

    li t5, 256
    add s0, s2, t5
    add s0, s0, t4
    li s4, %lo(H_BUF)
    addi s4, 64
    jal DMAIn
    li t0, DMA_SIZE(64, 1)

    li s5, 0                   # row-in-tile 0..31
RowLoop:
    sll t6, s5, 6               # row * 64 (row width in W_TILE this pass)
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
    slti t0, s5, 32
    bnez t0, RowLoop
    nop

    addi s6, 1
    slti t0, s6, 4
    bnez t0, KChunkLoop
    nop

    li s4, %lo(OUT_TILE)
    move s0, s3
    jal DMAOut
    li t0, DMA_SIZE(128, 1)

    addi s1, 8192                # TILE_ROWS(32) * H(256)
    addi s3, 128                 # TILE_ROWS(32) * 4
    addi t9, -1
    bnez t9, TileLoop
    nop

    move ra, s7
    jr ra
    nop
    .endfunc
