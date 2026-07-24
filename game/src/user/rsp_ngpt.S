########################################################
# rsp_ngpt.S — 64GPT GRU matvec RSP overlay: BUILD-TIME KERNEL SELECTOR.
#
# M12.1: this file used to BE the kernel; M12's H=1024 port overwrote
# M9's H=320 kernel in place, which meant re-running the capacity
# question later (docs/milestones/m12.1.md phase 6) would have meant
# hand-reverting/re-porting assembly again. Instead, both hardware-
# verified kernel bodies now live side by side, untouched byte-for-byte
# (verified via diff at splitout time):
#   rsp_ngpt_h320.inc  — M9-proven, DMEM-bound, the shipping default
#                        (M12 exonerated capacity: same corpus/seed,
#                        H=1024 was NOT more coherent, just 9x slower)
#   rsp_ngpt_h1024.inc — M12's K-chunked port, RDRAM-bound, proven at
#                        H=1024 on real hardware, kept for whenever the
#                        capacity question gets re-asked on a de-noised
#                        metric (docs/ideas-coherence-rescue-plan.md)
#
# NGPT_RSP_H selects which body gets assembled -- game/Makefile.custom
# sets it via RSPASFLAGS (default 320) and threads the SAME value into
# DialogueDemo.cpp's RSP_H via N64_CXXFLAGS, so the two can never drift
# apart the way a hand-maintained constant could. n64.mk's %.S rule
# invokes $(N64_CC) (a C-compiler frontend), so this uppercase-.S file
# runs through cpp -- #include/#if below work exactly like a .cpp file.
#
# The filename MUST start with "rsp": n64.mk keys RSP-ucode assembly
# off that prefix -- true of this dispatcher, NOT of the two .inc
# bodies (deliberately given a non-".S", non-"rsp"-prefixed name so the
# build system's glob never tries to assemble them directly; they only
# ever reach the assembler via this #include).
########################################################
#ifndef NGPT_RSP_H
#define NGPT_RSP_H 320   /* defensive default: matches DialogueDemo.cpp's */
#endif

#if NGPT_RSP_H == 1024
#include "rsp_ngpt_h1024.inc"
#elif NGPT_RSP_H == 320
#include "rsp_ngpt_h320.inc"
#else
#error "NGPT_RSP_H must be 320 or 1024 -- no kernel body ported for this H (see docs/milestones/m12.1.md phase 6 before adding one)"
#endif
