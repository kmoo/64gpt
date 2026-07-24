########################################################
# rsp_ngpt_h320.inc — H=320 body of the 64GPT GRU matvec RSP overlay.
#
# M12.1: split out of rsp_ngpt.S so the H=320 (this file, hardware-
# verified since M9, the DEFAULT/shipping kernel — capacity was
# exonerated at M12) and H=1024 (rsp_ngpt_h1024.inc, M12's K-chunked
# port) kernel bodies can coexist in the repo, selected at BUILD time
# by NGPT_RSP_H (game/Makefile.custom), instead of one replacing the
# other in git history every time the capacity question gets re-asked
# (docs/milestones/m12.1.md phase 6). rsp_ngpt.S itself is now a thin
# #include dispatcher — see that file. INSTRUCTION BYTES BELOW ARE
# UNCHANGED from the pre-M12 file (verified via diff at splitout time)
# — this is still the exact hardware-proven M9 kernel, not a rewrite.
#
# One command:
#   0 NgptMatvec — 960 rows x 320 cols int8xint16 matvec, DMA-tiled,
#                  bit-exact vs the CPU engine (M9: generalized from the
#                  H=256 kernel, see docs/spikes/rsp-matvec-h256.md and
#                  docs/milestones/m9.md section 6)
#
# Hard-won lessons encoded below (full ledger in
# docs/spikes/rsp-matvec.md):
#  - vector shifts are multiplies: they CLOBBER the accumulator, so all
#    unpacking runs before any vmudh/vmadh
#  - lpv rotates lanes by addr&15: weights are unpacked from lqv
#    byte-pairs instead (even/odd split; the CPU pre-shuffles h once)
#  - lqv/sqv only reach the next 16B boundary: buffers are .align 4
#  - mfc2 .eN maps lane N to byte offset 2N via byteVectorElements
#
# H=320 generalization (M9): unlike H=128->256 (both powers of 2), 320
# does NOT divide 2048 evenly, so W_TILE can no longer be pinned at a
# constant 2048B by construction. tile_rows = floor(2048/320) = 6 (not
# an idealized fraction) -- 6*320 = 1920B, LESS than 2048B, which is
# fine (more DMEM headroom, not less) as long as 3H (=960, total W_hh
# rows) divides evenly by tile_rows: 960/6 = 160 tiles exactly, no
# remainder tile needed. Row width is no longer a power of 2, so the
# row-address multiply (row * 320) can't be a single shift: 320 =
# 256+64 = (1<<8)+(1<<6), so it's decomposed into two shifts + an add
# (exact, cheap, and generalizes to any H that's a short sum of powers
# of 2 -- true of every H this project has used or is likely to use).
# H=320 is still a multiple of 16 (320/16=20) and 8 (160/8=20), so the
# SIMD unpack/accumulate loop *structure* is unchanged, only iteration
# counts scale (16->20 chunks). DMEM: W_TILE 1920 + W16_BUF 640 +
# H_BUF 640 + OUT_TILE 24 = 3224B .bss (was 3104B at H=256) + 552B
# .data = 3776B total, 320B headroom (vs. H=256's 440B) -- thinner but
# real, computed from the actual integer tile-depth, not the idealized
# continuous 8192/H approximation (docs/milestones/m9.md section 6's
# own flagged caveat).
########################################################
#include <rsp_queue.inc>   /* already provides the rsp_dma.inc routines */

.data

RSPQ_BeginOverlayHeader
    RSPQ_DefineCommand NgptMatvec, 16 # cmd, dram_w(960x320), dram_h, dram_out
RSPQ_EndOverlayHeader

RSPQ_EmptySavedState

.bss
    .align 4
W_TILE:  .space 1920         # 6 packed int8 rows (320B each) per DMA tile
    .align 4
W16_BUF: .space 640          # one row unpacked: 160 quads EVEN w, then ODD w
    .align 4
H_BUF:   .space 640          # h, CPU-shuffled: h[0,2,..318] then h[1,3,..319]
    .align 4
OUT_TILE: .space 24          # 6 int32 row sums per tile

.text

########################################################
# DotRow: exact int32 dot product of one 320-wide row.
#   in : t6 = DMEM addr of 320 packed int8 weights (clobbered)
#        H_BUF holds the CPU-shuffled h (even stream, then odd)
#   out: t7 = int32 sum
#   clobbers: t0,t1,t3,t5,t6,t8, $v01-$v05, ACC
#   preserves: s0-s7, t9, a0-a3
#
# vmudh/vmadh accumulate (S16xS16 product << 16) into the 48-bit ACC;
# per lane 40 products of <= 2^22 peak below 2^42 (well under the 48-bit
# ceiling) — no clamp, no rounding. Exact per-lane int32 =
# (ACC_HI << 16) | (u16)ACC_MD.
########################################################
    .func DotRow
DotRow:
    # Phase 1: unpack int8 -> int16 (all shifts BEFORE any MAC).
    # 20 iterations x 16 packed weights/iter = 320 weights/row (was 16x16=256).
    li t8, %lo(W16_BUF)
    li t3, 20
UnpackLoop:
    lqv $v04.e0, 0,t6
    vsra $v02, $v04, 8       # even-index weights (BE high byte), exact
    sqv $v02.e0, 0,t8
    vsll8 $v05, $v04, 8
    vsra $v05, $v05, 8       # odd-index weights, sign-extended
    sqv $v05.e0, 320,t8      # odd half starts at +320 (== new even-half size, == H)
    addi t6, 16
    addi t8, 16
    addi t3, -1
    bnez t3, UnpackLoop
    nop

    # Phase 2: clean accumulation; first vmudh clears ACC.
    # EVEN half: 160 h-values (was 128) -> 20 chunks of 8 lanes (1 initial
    # vmudh + 19 more in EvenLoop, was 1+15).
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t8
    vmudh $v03, $v01, $v02
    li t3, 19
EvenLoop:
    addi t5, 16
    addi t8, 16
    lqv $v01.e0, 0,t5
    lqv $v02.e0, 0,t8
    vmadh $v03, $v01, $v02
    addi t3, -1
    bnez t3, EvenLoop
    nop
    # ODD half: 160 h-values (was 128) -> 20 chunks (was 16). H_BUF's odd
    # half now starts at byte 320 (even half grew to 320B); W16_BUF's
    # odd half likewise moved to 320 -- see file header math.
    li t5, %lo(H_BUF)
    li t8, %lo(W16_BUF)
    li t3, 20
OddLoop:
    lqv $v01.e0, 320,t5
    lqv $v02.e0, 320,t8
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
# NgptMatvec: 960 rows x 320, tiled 6 rows (1920B) per DMA.
#   a1 = RDRAM addr of 960x320 int8 weights, row-major, 8-byte aligned
#   a2 = RDRAM addr of shuffled h (640B)
#   a3 = RDRAM addr of 960 int32 outputs (3840B)
########################################################
    .func NgptMatvec
NgptMatvec:
    move s7, ra              # jal clobbers ra (= RSPQ_Loop)
    move s1, a1              # w cursor (advances 1920B per tile)
    move s2, a2
    move s3, a3              # out cursor (advances 24B per tile)

    li s4, %lo(H_BUF)
    move s0, s2
    jal DMAIn
    li t0, DMA_SIZE(640, 1)

    li t9, 160                # 160 tiles x 6 rows = 960
TileLoop:
    li s4, %lo(W_TILE)
    move s0, s1
    jal DMAIn
    li t0, DMA_SIZE(1920, 1)

    li s5, 0                 # row-in-tile 0..5
RowLoop:
    sll t6, s5, 8             # row * 256
    sll t1, s5, 6             # row * 64
    addu t6, t6, t1           # row * 320 (256+64, exact for H=320)
    addi t6, %lo(W_TILE)
    jal DotRow
    nop
    sll t0, s5, 2
    addi t0, %lo(OUT_TILE)
    sw t7, 0(t0)
    addi s5, 1
    slti t0, s5, 6
    bnez t0, RowLoop
    nop

    li s4, %lo(OUT_TILE)
    move s0, s3
    jal DMAOut
    li t0, DMA_SIZE(24, 1)

    addi s1, 1920
    addi s3, 24
    addi t9, -1
    bnez t9, TileLoop
    nop

    move ra, s7
    jr ra
    nop
    .endfunc
