########################################################
# rsp_ngpt.S — 64GPT GRU matvec on the RSP (rspq overlay)
#
# One command:
#   0 NgptMatvec — 3072 rows x 1024 cols int8xint16 matvec, DMA-tiled in
#                  BOTH dimensions (K-chunked kernel, M12 -- ported from
#                  the hardware-verified spike, docs/spikes/
#                  rsp-matvec-ktile.md, branch spike/rsp-matvec-ktile,
#                  commit bc3a2c9's exact H=1024/CHUNK=256 configuration:
#                  SELFTEST PASS, XCHK bit-exact PASS, 2.93x speedup,
#                  5.37 ch/s), bit-exact vs the CPU engine
#
# CHUNK=256 AT H=1024. The spike's own recommendation was H=768 (real
# margin against the ~5 ch/s comfort floor and RDRAM); H=1024 was chosen
# anyway for this milestone (explicit sign-off, core/ngpt.h) as "how far
# can this actually go," not the defensible default. Two real risks
# accepted deliberately, spelled out in core/ngpt.h, not repeated here:
# the RSP's own int32 accumulator (this file) has essentially zero
# margin left at this H against a REAL trained model's bias magnitudes
# (only a gibberish spike model has verified it in practice), and RDRAM
# is tight (~3.6MB static footprint of a base 4MB budget). If a real
# trained build at this H fails to link, crashes, or produces a wrong
# (not just slow) XCHK result, that IS the answer to "how large can it
# be" -- not a bug to chase before reporting it.
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
# K-CHUNKED GENERALIZATION: every kernel through M9-M11.1 (H up to 320)
# kept the ENTIRE hidden state h resident in DMEM for the whole matvec
# -- the wall docs/milestones/m9.md section 6 formalizes as DMEM(H) =
# 2600 + 4H + 8192/H, unsafe past H≈368. This kernel tiles the
# REDUCTION dimension the same way the pre-M12 kernel already tiled the
# OUTPUT (row) dimension.
#
# CHUNK = 256 elements. TILE_ROWS = 2048/CHUNK = 8, a CONSTANT
# regardless of H -- that's what makes chunk and H independent knobs.
#
# H=1024: H/CHUNK and 3H/TILE_ROWS both divide evenly: 1024/256 = 4
# K-chunks, 3072/8 = 384 row-tiles, zero wasted DMA bytes.
#
# DMEM budget: W_TILE 2048 + W16_BUF 512 + H_BUF 512 + OUT_TILE 32 =
# 3104B .bss -- structurally independent of H (chunk/tile_rows are both
# fixed constants), proven identical at H=256/512/768/1024 in the
# originating spike's own sweep.
########################################################
#include <rsp_queue.inc>   /* already provides the rsp_dma.inc routines */

.data

RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand NgptMatvec, 16 # cmd, dram_w(3072x1024), dram_h, dram_out
RSPQ_EndOverlayHeader

RSPQ_EmptySavedState

.bss
    .align 4
W_TILE:  .space 2048         # 8 packed int8 rows x 256B (one K-chunk's columns) per DMA tile
    .align 4
W16_BUF: .space 512          # one row's chunk unpacked: 128 quads EVEN w, then ODD w
    .align 4
H_BUF:   .space 512          # h chunk, CPU-shuffled globally (even stream, then odd); this
                              # buffer holds only this K-chunk's 128-even + 128-odd slice
    .align 4
OUT_TILE: .space 32          # 8 int32 row partial-sums for the current row-tile,
                              # accumulated ACROSS the K-chunk loop (not reset per chunk)

.text

########################################################
# DotRowChunk: exact int32 PARTIAL dot product of one 256-wide K-chunk
# of a single row (contrast the pre-M12 DotRow, which did a FULL H-wide
# row in one call -- this does 1/NUM_KCHUNKS of a row; the caller
# accumulates across chunks into OUT_TILE).
#   in : t6 = DMEM addr of 256 packed int8 weights for this (row, chunk)
#        H_BUF holds this chunk's CPU-shuffled h (even 128, then odd 128)
#   out: t7 = int32 partial sum for this chunk
#   clobbers: t0,t1,t3,t5,t6,t8, $v01-$v05, ACC
#   preserves: s0-s7, t9, a0-a3
#
# vmudh/vmadh accumulate (S16xS16 product << 16) into the 48-bit ACC;
# per lane products stay far under the 48-bit ceiling regardless of
# chunk size -- no clamp, no rounding. Exact per-lane int32 =
# (ACC_HI << 16) | (u16)ACC_MD.
########################################################
    .func DotRowChunk
DotRowChunk:
    # Phase 1: unpack int8 -> int16 (all shifts BEFORE any MAC).
    # 16 iterations x 16 packed weights/iter = 256 weights/chunk.
    li t8, %lo(W16_BUF)
    li t3, 16
UnpackLoop:
    lqv $v04.e0, 0,t6
    vsra $v02, $v04, 8       # even-index weights (BE high byte), exact
    sqv $v02.e0, 0,t8
    vsll8 $v05, $v04, 8
    vsra $v05, $v05, 8       # odd-index weights, sign-extended
    sqv $v05.e0, 256,t8      # odd half starts at +256 (== new even-half size)
    addi t6, 16
    addi t8, 16
    addi t3, -1
    bnez t3, UnpackLoop
    nop

    # Phase 2: clean accumulation; first vmudh clears ACC.
    # EVEN half: 128 h-values -> 16 groups of 8 lanes (1 initial vmudh +
    # 15 more in EvenLoop).
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t8
    vmudh $v03, $v01, $v02
    li t3, 15
EvenLoop:
    addi t5, 16
    addi t8, 16
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t8
    vmadh $v03, $v01, $v02
    addi t3, -1
    bnez t3, EvenLoop
    nop
    # ODD half: 128 h-values -> 16 groups. H_BUF's odd half starts at
    # byte 256 (even half is 256B at this chunk size); W16_BUF's odd
    # half is also at 256 (see file header math).
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    li t3, 16
OddLoop:
    lqv $v01.e0, 256,t5
    lqv $v02.e0, 256,t8
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
# NgptMatvec: 3072 rows x 1024 cols, 2D-tiled -- 384 row-tiles (8 rows
# each) x 4 K-chunks (256 cols each) per row-tile.
#   a1 = RDRAM addr of 3072x1024 int8 weights, row-major, 8-byte aligned
#   a2 = RDRAM addr of shuffled h (2048B: 512 evens then 512 odds)
#   a3 = RDRAM addr of 3072 int32 outputs (12288B)
#
# Register map (s0-s7 fully committed, one more than the pre-M12
# kernel's allocation -- s6 is new):
#   s0 = DMA scratch: current transfer's RDRAM source address
#   s1 = row-tile base address into a1, advances by TILE_ROWS*H (8192)
#        after all K-chunks for a row-tile are done
#   s2 = h base address (a2), fixed for the whole call
#   s3 = output cursor into a3, advances by TILE_ROWS*4 (32) per tile
#   s4 = DMA scratch: DMEM destination address
#   s5 = row-in-tile counter, 0..7
#   s6 = K-chunk counter, 0..3
#   s7 = saved ra (jal clobbers ra = RSPQ_Loop)
########################################################
    .func NgptMatvec
NgptMatvec:
    move s7, ra
    move s1, a1
    move s2, a2
    move s3, a3

    li t9, 384                # 384 row-tiles x 8 rows = 3072
TileLoop:
    li s6, 0                  # K-chunk 0..3 (H=1024 / CHUNK=256 = 4)
KChunkLoop:
    # W_TILE <- pitched 2D DMA: 8 rows x 256B, pitch 1024B (H), starting
    # at this row-tile's base + this chunk's 256B column offset.
    sll t4, s6, 8              # t4 = chunk * 256 (byte column offset)
    add s0, s1, t4
    li s4, %lo(W_TILE)
    li t1, 1024                # pitch = H (real RDRAM row width)
    li t0, DMA_SIZE(256, 8)
    jal DMAIn
    nop

    # H_BUF <- this chunk's 128-even + 128-odd slice, ONE pitched 2D DMA
    # (even/odd regions are exactly H bytes apart in RDRAM; DMEM dest is
    # linear, so height=2 writes H_BUF+0 then H_BUF+256 contiguously,
    # matching DotRowChunk's read layout).
    add s0, s2, t4
    li s4, %lo(H_BUF)
    li t1, 1024                # pitch = H (odd region starts H bytes after even)
    jal DMAIn
    li t0, DMA_SIZE(256, 2)

    li s5, 0                   # row-in-tile 0..7
RowLoop:
    sll t6, s5, 8               # row * 256 (row width in W_TILE this pass)
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
    slti t0, s5, 8
    bnez t0, RowLoop
    nop

    addi s6, 1
    slti t0, s6, 4
    bnez t0, KChunkLoop
    nop

    li s4, %lo(OUT_TILE)
    move s0, s3
    jal DMAOut
    li t0, DMA_SIZE(32, 1)

    addi s1, 8192                # TILE_ROWS(8) * H(1024)
    addi s3, 32                  # TILE_ROWS(8) * 4
    addi t9, -1
    bnez t9, TileLoop
    nop

    move ra, s7
    jr ra
    nop
    .endfunc
