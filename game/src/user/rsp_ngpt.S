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
    # 16-byte alignment is load-bearing: lqv/sqv transfer only up to the
    # next 16B boundary, so an 8-aligned buffer half-fills the lanes and
    # leaves stale DMEM in the rest.
    .align 4
W_BUF:   .space 128          # one row of int8 weights (packed)
    .align 4
W16_BUF: .space 512          # unpacked: 16 quads of EVEN w, then ODD w
    .align 4
H_BUF:   .space 256          # h, CPU-shuffled: h[0,2,..126] then h[1,3,..127]
    .align 4
OUT_BUF: .space 16           # int32 result (+ pad)

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

    # Phase 1: unpack int8 -> int16 with lqv ONLY (lpv rotates lanes by
    # the address's low 4 bits — the deterministic wrong-sum bug). Each
    # lqv reads 16 packed weights as 8 big-endian byte-pairs; the even
    # byte is lane>>8 (exact: vsra is x*0x100 taking the high half), the
    # odd byte is (lane<<8)>>8. Even quads land at W16_BUF[k], odd quads
    # at W16_BUF+256[k]; the CPU pre-shuffles h to match. All shifts run
    # before any accumulation (vector shifts are multiplies and clobber
    # the accumulator — the earlier zero-sum bug).
    li t6, %lo(W_BUF)
    li t7, %lo(W16_BUF)
    li t3, 8
UnpackLoop:
    lqv $v04.e0, 0,t6
    vsra $v02, $v04, 8       # even-index weights, sign-extended
    sqv $v02.e0, 0,t7
    vsll8 $v05, $v04, 8
    vsra $v05, $v05, 8       # odd-index weights, sign-extended
    sqv $v05.e0, 256,t7
    addi t6, 16
    addi t7, 16
    addi t3, -1
    bnez t3, UnpackLoop
    nop

    # Phase 2: clean accumulation — 16 quads (8 even + 8 odd) against
    # the matching CPU-shuffled h stream; first vmudh clears ACC.
    li t5, %lo(H_BUF)
    li t7, %lo(W16_BUF)
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t7
    vmudh $v03, $v01, $v02
    li t3, 7
EvenLoop:
    addi t5, 16
    addi t7, 16
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t7
    vmadh $v03, $v01, $v02
    addi t3, -1
    bnez t3, EvenLoop
    nop
    li t5, %lo(H_BUF)
    li t7, %lo(W16_BUF)
    li t3, 8
OddLoop:
    lqv $v01.e0, 128,t5
    lqv $v02.e0, 256,t7
    vmadh $v03, $v01, $v02
    addi t5, 16
    addi t7, 16
    addi t3, -1
    bnez t3, OddLoop
    nop

    vsar $v04, COP2_ACC_MD
    vsar $v05, COP2_ACC_HI

    # fold 8 lanes on the scalar unit: sum += (HI << 16) | (u16)MD.
    # mfc2's element is a BYTE offset (0..15), not a lane index — lanes
    # live at even byte offsets 0,2,..,14. (Reading .e0..e7 folds lanes
    # 0-3 plus four misaligned lane-straddles — the deterministic wrong
    # sum that survived two unpack rewrites.)
    li t7, 0
.macro FoldLane byteoff
    mfc2 t0, $v05.e\byteoff
    sll t0, 16
    mfc2 t1, $v04.e\byteoff
    andi t1, 0xFFFF
    add t0, t0, t1
    add t7, t7, t0
.endm
    FoldLane 0
    FoldLane 2
    FoldLane 4
    FoldLane 6
    FoldLane 8
    FoldLane 10
    FoldLane 12
    FoldLane 14

    li t8, %lo(OUT_BUF)
    sw t7, 0(t8)

    li s4, %lo(OUT_BUF)
    move s0, s3
    li t0, DMA_SIZE(8, 1)
    move ra, s7              # tail into DMAOut returns to RSPQ_Loop
    j DMAOut
    nop
    .endfunc
