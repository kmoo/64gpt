########################################################
# rsp_ngpt.S — SPIKE: GRU matvec on the RSP (rspq overlay)
#
# Gate G1: command 0 (NgptEcho) DMAs an 8-byte magic constant from DMEM
# to a caller-supplied RDRAM address — proves assemble/register/dispatch.
# The filename MUST start with "rsp": n64.mk keys RSP-ucode assembly off
# that prefix.
########################################################
#include <rsp_queue.inc>   /* already provides the rsp_dma.inc routines */

.data

RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand NgptEcho, 8   # word0: cmd, word1: RDRAM dest
    RSPQ_DefineCommand NgptDot, 16   # cmd, dram_w, dram_h, dram_out
RSPQ_EndOverlayHeader

RSPQ_EmptySavedState

.align 3
NGPT_MAGIC:
    .long 0x600D64AA        # "good N64" + AA
    .long 0x364D4143        # '6','M','A','C'

.bss
    .align 3
W_BUF:   .space 128          # one row of int8 weights
    .align 3
H_BUF:   .space 256          # h vector, 128 x int16
    .align 3
OUT_BUF: .space 8            # int32 result (+ pad to DMA granule)

.text

########################################################
# NgptEcho: a0 = command word (unused args), a1 = RDRAM destination
# Tail-jump into DMAOut: ra still points at RSPQ_Loop, so the DMA
# routine's return hands control straight back to the queue engine.
########################################################
    .func NgptEcho
NgptEcho:
    li s4, %lo(NGPT_MAGIC)
    move s0, a1
    li t0, DMA_SIZE(8, 1)
    j DMAOut
    nop
    .endfunc

########################################################
# NgptDot (gate G2): exact int32 dot product of one weight row.
#   a1 = RDRAM addr of 128 int8 weights
#   a2 = RDRAM addr of 128 int16 h values (big-endian, as everywhere)
#   a3 = RDRAM addr of the int32 result
#
# Exactness scheme: vmudh/vmadh multiply S16xS16 and add (product << 16)
# into the 48-bit accumulators. Per lane: 16 products of <= 2^22, so the
# accumulator peaks below 2^42 — no clamp, no rounding. The exact 32-bit
# per-lane sum is then (ACC_HI << 16) | ACC_MD, folded on the scalar
# unit. Weights arrive as packed bytes via lpv (byte << 8 per lane) and
# are sign-restored with an arithmetic >> 8.
########################################################
    .func NgptDot
NgptDot:
    move s7, ra              # jal below clobbers ra (= RSPQ_Loop)
    move s1, a1
    move s2, a2
    move s3, a3

    li s4, %lo(H_BUF)
    move s0, s2
    jal DMAIn
    li t0, DMA_SIZE(256, 1)

    li s4, %lo(W_BUF)
    move s0, s1
    jal DMAIn
    li t0, DMA_SIZE(128, 1)

    # 16 chunks x 8 lanes; first chunk vmudh (clears ACC), rest vmadh
    li t5, %lo(H_BUF)
    li t6, %lo(W_BUF)
    lqv $v01.e0, 0,t5
    lpv $v02.e0, 0,t6
    vsra $v02, $v02, 8
    vmudh $v03, $v01, $v02
    li t3, 15
DotLoop:
    addi t5, 16
    addi t6, 8
    lqv $v01.e0, 0,t5
    lpv $v02.e0, 0,t6
    vsra $v02, $v02, 8
    vmadh $v03, $v01, $v02
    addi t3, -1
    bnez t3, DotLoop
    nop

    vsar $v04, COP2_ACC_MD
    vsar $v05, COP2_ACC_HI

    # fold 8 lanes on the scalar unit: sum += (HI << 16) | (u16)MD
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

    li t8, %lo(OUT_BUF)
    sw t7, 0(t8)

    li s4, %lo(OUT_BUF)
    move s0, s3
    li t0, DMA_SIZE(8, 1)
    move ra, s7              # tail into DMAOut returns to RSPQ_Loop
    j DMAOut
    nop
    .endfunc
