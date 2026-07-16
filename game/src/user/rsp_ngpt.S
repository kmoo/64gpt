########################################################
# rsp_ngpt.S — 64GPT GRU matvec on the RSP (rspq overlay)
#
# One command:
#   0 NgptMatvec — 384 rows x 128 cols int8xint16 matvec, DMA-tiled,
#                  bit-exact vs the CPU engine (proven: spike/rsp-matvec)
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
# Scaling note (H=256, the M7 target): DotRow is written for 128
# columns — 8 unpack chunks and 8+8 MAC quads with fixed 128/256-byte
# stream offsets. Generalize the loop counts + offsets from a DMEM
# header the C side writes (or emit per-H code) before raising H.
########################################################
#include <rsp_queue.inc>   /* already provides the rsp_dma.inc routines */

.data

RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand NgptMatvec, 16 # cmd, dram_w(384x128), dram_h, dram_out
RSPQ_EndOverlayHeader

RSPQ_EmptySavedState

.bss
    .align 4
W_TILE:  .space 2048         # 16 packed int8 rows per DMA tile
    .align 4
W16_BUF: .space 512          # one row unpacked: 16 quads EVEN w, then ODD w
    .align 4
H_BUF:   .space 256          # h, CPU-shuffled: h[0,2,..126] then h[1,3,..127]
    .align 4
OUT_TILE: .space 64          # 16 int32 row sums per tile

.text

########################################################
# DotRow: exact int32 dot product of one 128-wide row.
#   in : t6 = DMEM addr of 128 packed int8 weights (clobbered)
#        H_BUF holds the CPU-shuffled h (even stream, then odd)
#   out: t7 = int32 sum
#   clobbers: t0,t1,t3,t5,t6,t8, $v01-$v05, ACC
#   preserves: s0-s7, t9, a0-a3
#
# vmudh/vmadh accumulate (S16xS16 product << 16) into the 48-bit ACC;
# per lane 16 products of <= 2^22 peak below 2^42 — no clamp, no
# rounding. Exact per-lane int32 = (ACC_HI << 16) | (u16)ACC_MD.
########################################################
    .func DotRow
DotRow:
    # Phase 1: unpack int8 -> int16 (all shifts BEFORE any MAC)
    li t8, %lo(W16_BUF)
    li t3, 8
UnpackLoop:
    lqv $v04.e0, 0,t6
    vsra $v02, $v04, 8       # even-index weights (BE high byte), exact
    sqv $v02.e0, 0,t8
    vsll8 $v05, $v04, 8
    vsra $v05, $v05, 8       # odd-index weights, sign-extended
    sqv $v05.e0, 256,t8
    addi t6, 16
    addi t8, 16
    addi t3, -1
    bnez t3, UnpackLoop
    nop

    # Phase 2: clean accumulation; first vmudh clears ACC
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
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    li t3, 8
OddLoop:
    lqv $v01.e0, 128,t5
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
# NgptMatvec: 384 rows x 128, tiled 16 rows (2KB) per DMA.
#   a1 = RDRAM addr of 384x128 int8 weights, row-major, 8-byte aligned
#   a2 = RDRAM addr of shuffled h (256B)
#   a3 = RDRAM addr of 384 int32 outputs (1536B)
########################################################
    .func NgptMatvec
NgptMatvec:
    move s7, ra              # jal clobbers ra (= RSPQ_Loop)
    move s1, a1              # w cursor (advances 2KB per tile)
    move s2, a2
    move s3, a3              # out cursor (advances 64B per tile)

    li s4, %lo(H_BUF)
    move s0, s2
    jal DMAIn
    li t0, DMA_SIZE(256, 1)

    li t9, 24                # 24 tiles x 16 rows = 384
TileLoop:
    li s4, %lo(W_TILE)
    move s0, s1
    jal DMAIn
    li t0, DMA_SIZE(2048, 1)

    li s5, 0                 # row-in-tile 0..15
RowLoop:
    sll t6, s5, 7            # row * 128
    addi t6, %lo(W_TILE)
    jal DotRow
    nop
    sll t0, s5, 2
    addi t0, %lo(OUT_TILE)
    sw t7, 0(t0)
    addi s5, 1
    slti t0, s5, 16
    bnez t0, RowLoop
    nop

    li s4, %lo(OUT_TILE)
    move s0, s3
    jal DMAOut
    li t0, DMA_SIZE(64, 1)

    addi s1, 2048
    addi s3, 64
    addi t9, -1
    bnez t9, TileLoop
    nop

    move ra, s7
    jr ra
    nop
    .endfunc
