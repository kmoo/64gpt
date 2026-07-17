########################################################
# rsp_ngpt.S — 64GPT GRU matvec on the RSP (rspq overlay)
#
# One command:
#   0 NgptMatvec — 768 rows x 256 cols int8xint16 matvec, DMA-tiled,
#                  bit-exact vs the CPU engine (spike: rsp-matvec-h256,
#                  generalized from the proven 128-wide kernel, see
#                  docs/spikes/rsp-matvec.md)
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
# H=256 generalization: cols doubled (128->256, so rows are 256B not
# 128B) and rows doubled (384->768, since W_hh is 3H x H). DMEM is the
# constraint: 16 rows/tile at 256B/row would be 4096B for W_TILE alone,
# the ENTIRE RSP DMEM budget with nothing left for code. Halved tile
# depth to 8 rows/tile instead (same 2048B tile size as the H=128
# kernel, just reshaped) to keep the budget sane: W_TILE 2048 + W16_BUF
# 512 + H_BUF 512 + OUT_TILE 32 = 3104B total .bss (was 2880B at H=128).
########################################################
#include <rsp_queue.inc>   /* already provides the rsp_dma.inc routines */

.data

RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand NgptMatvec, 16 # cmd, dram_w(768x256), dram_h, dram_out
RSPQ_EndOverlayHeader

RSPQ_EmptySavedState

.bss
    .align 4
W_TILE:  .space 2048         # 8 packed int8 rows (256B each) per DMA tile
    .align 4
W16_BUF: .space 512          # one row unpacked: 128 quads EVEN w, then ODD w
    .align 4
H_BUF:   .space 512          # h, CPU-shuffled: h[0,2,..254] then h[1,3,..255]
    .align 4
OUT_TILE: .space 32          # 8 int32 row sums per tile

.text

########################################################
# DotRow: exact int32 dot product of one 256-wide row.
#   in : t6 = DMEM addr of 256 packed int8 weights (clobbered)
#        H_BUF holds the CPU-shuffled h (even stream, then odd)
#   out: t7 = int32 sum
#   clobbers: t0,t1,t3,t5,t6,t8, $v01-$v05, ACC
#   preserves: s0-s7, t9, a0-a3
#
# vmudh/vmadh accumulate (S16xS16 product << 16) into the 48-bit ACC;
# per lane 32 products of <= 2^22 peak below 2^42 (well under the 48-bit
# ceiling) — no clamp, no rounding. Exact per-lane int32 =
# (ACC_HI << 16) | (u16)ACC_MD.
########################################################
    .func DotRow
DotRow:
    # Phase 1: unpack int8 -> int16 (all shifts BEFORE any MAC).
    # 16 iterations x 16 packed weights/iter = 256 weights/row (was 8x16=128).
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
    # EVEN half: 128 h-values (was 64) -> 16 chunks of 8 lanes (1 initial
    # vmudh + 15 more in EvenLoop, was 1+7).
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
    # ODD half: 128 h-values (was 64) -> 16 chunks (was 8). H_BUF's odd
    # half now starts at byte 256 (even half doubled to 256B); W16_BUF's
    # odd half stays at 256 (unchanged -- see file header math).
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
# NgptMatvec: 768 rows x 256, tiled 8 rows (2KB) per DMA.
#   a1 = RDRAM addr of 768x256 int8 weights, row-major, 8-byte aligned
#   a2 = RDRAM addr of shuffled h (512B)
#   a3 = RDRAM addr of 768 int32 outputs (3072B)
########################################################
    .func NgptMatvec
NgptMatvec:
    move s7, ra              # jal clobbers ra (= RSPQ_Loop)
    move s1, a1              # w cursor (advances 2KB per tile)
    move s2, a2
    move s3, a3              # out cursor (advances 32B per tile)

    li s4, %lo(H_BUF)
    move s0, s2
    jal DMAIn
    li t0, DMA_SIZE(512, 1)

    li t9, 96                # 96 tiles x 8 rows = 768
TileLoop:
    li s4, %lo(W_TILE)
    move s0, s1
    jal DMAIn
    li t0, DMA_SIZE(2048, 1)

    li s5, 0                 # row-in-tile 0..7
RowLoop:
    sll t6, s5, 8            # row * 256
    addi t6, %lo(W_TILE)
    jal DotRow
    nop
    sll t0, s5, 2
    addi t0, %lo(OUT_TILE)
    sw t7, 0(t0)
    addi s5, 1
    slti t0, s5, 8
    bnez t0, RowLoop
    nop

    li s4, %lo(OUT_TILE)
    move s0, s3
    jal DMAOut
    li t0, DMA_SIZE(32, 1)

    addi s1, 2048
    addi s3, 32
    addi t9, -1
    bnez t9, TileLoop
    nop

    move ra, s7
    jr ra
    nop
    .endfunc
