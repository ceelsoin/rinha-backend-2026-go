#include "textflag.h"

// func sqDist16(a, b *[stride]int16) int32
//
// Computes Manhattan (L1) distance between two 16-element int16 vectors
// using a single AVX2 256-bit load per operand.
//
// All 16 int16 fit in one 256-bit YMM register, so this function issues
// just 2 memory loads (vs 4 in the old SSE version) and no loop.
//
// Values are quantized floats in [-32767, 32767].
// Max |diff| = 65534 — fits in uint16, so VPABSW (treats as signed) is safe
// up to 32767 per element after the wrapping subtraction.
// Worst-case sum = 16 × 32767 = 524272 < 2³¹ — no int32 overflow.
//
// ABI0 (stack-based):
//   a   at  0(FP)  8 bytes
//   b   at  8(FP)  8 bytes
//   ret at 16(FP)  4 bytes
//   frame: $0-20

TEXT ·sqDist16(SB),NOSPLIT,$0-20
    MOVQ a+0(FP), SI
    MOVQ b+8(FP), DI

    // ── Load 32 bytes (16 × int16) in a single AVX2 256-bit load ─────────
    // VMOVDQU ymm0, [rsi]  (VEX.256.F3.0F 6F /r)
    BYTE $0xC5; BYTE $0xFE; BYTE $0x6F; BYTE $0x06
    // VMOVDQU ymm1, [rdi]
    BYTE $0xC5; BYTE $0xFE; BYTE $0x6F; BYTE $0x0F

    // ── Subtract: ymm0 = a − b (int16 wrapping, safe for quantized range) ─
    // VPSUBW ymm0, ymm0, ymm1  (VEX.256.66.0F F9 /r)
    BYTE $0xC5; BYTE $0xFD; BYTE $0xF9; BYTE $0xC1

    // ── Absolute value: ymm0 = |a − b|  (SSSE3 VPABSW in AVX2 encoding) ──
    // VPABSW ymm0, ymm0  (VEX.256.66.0F38 1D /r)
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x1D; BYTE $0xC0

    // ── Zero-extend 16 × int16 → 8+8 × int32 ────────────────────────────
    // Extract high 128 bits into xmm1.
    // VEXTRACTI128 xmm1, ymm0, 1  (VEX.256.66.0F3A 39 /r imm8)
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xC1; BYTE $0x01

    // VPMOVZXWD ymm2, xmm0: zero-extend low  8 int16 → 8 int32
    // (VEX.256.66.0F38 33 /r)
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xD0

    // VPMOVZXWD ymm3, xmm1: zero-extend high 8 int16 → 8 int32
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xD9

    // ── Sum the two groups of 8 int32 ─────────────────────────────────────
    // VPADDD ymm0, ymm2, ymm3  (VEX.256.66.0F FE /r, vvvv=ymm2)
    BYTE $0xC5; BYTE $0xED; BYTE $0xFE; BYTE $0xC3

    // ── Reduce 8 int32 in ymm0 to a single int32 ─────────────────────────
    // Extract high 4 int32 into xmm1.
    // VEXTRACTI128 xmm1, ymm0, 1
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xC1; BYTE $0x01

    // Clear upper YMM state before using legacy SSE instructions.
    // VZEROUPPER
    BYTE $0xC5; BYTE $0xF8; BYTE $0x77

    // xmm0 = low 4 int32, xmm1 = high 4 int32.
    PADDD X1, X0           // X0 = 4 partial sums

    // Horizontal reduction: 4 int32 → 1 int32.
    PSHUFD $0x4E, X0, X1   // X1 = [X0[2], X0[3], X0[0], X0[1]]
    PADDD  X1, X0           // X0 = [s0+s2, s1+s3, ...]
    PSHUFD $0xB1, X0, X1   // X1 = [X0[1], X0[0], ...]
    PADDD  X1, X0           // X0[0] = total Manhattan distance

    MOVL X0, ret+16(FP)
    RET

// ── prefetchVec ───────────────────────────────────────────────────────────────

// func prefetchVec(p *int16)
//
// Issues a PREFETCHT0 hint for the cache line containing *p.
// Called from the ANN scan loop ~24 records ahead of current position to
// hide the L2/L3 cache miss latency of the int16 vector data.
//
// ABI0: p at 0(FP), 8 bytes; frame $0-8.

TEXT ·prefetchVec(SB),NOSPLIT,$0-8
    MOVQ p+0(FP), SI
    // PREFETCHT0 [rsi]  (0F 18 /1 with ModRM for [rsi])
    BYTE $0x0F; BYTE $0x18; BYTE $0x06
    RET

// ── dist4 ─────────────────────────────────────────────────────────────────────

// func dist4(q *[stride]int16, refs *int16, out *[4]int32)
//
// Computes Manhattan (L1) distance from q to four consecutive stride-size
// reference vectors starting at refs, writing results to out[0..3].
//
// All 16 int16 of each vector fit in one 256-bit YMM register. The function
// loads the query once (ymm0) and the four references sequentially (ymm1-4),
// then processes all four in batch before reducing to scalar int32 values.
//
// Pipeline overview (AVX2):
//   VMOVDQU ×5          — 1 query + 4 ref loads
//   VPSUBW  ×4          — diffs (wrapping int16)
//   VPABSW  ×4          — absolute values
//   VEXTRACTI128 ×4     — extract high 128 bits per vector
//   VPMOVZXWD ×8        — zero-extend int16→int32 (low+high per vector)
//   VPADDD  ×4          — sum low+high int32 groups per vector
//   VEXTRACTI128 ×4     — extract high 4 int32 of each sum
//   VZEROUPPER          — clear upper YMM bits before SSE reduction
//   PADDD + PSHUFD ×12  — horizontal reduce 4×(4 int32 → 1 int32)
//   MOVL ×4             — store results
//
// ABI0:
//   q    at  0(FP)  8 bytes
//   refs at  8(FP)  8 bytes
//   out  at 16(FP)  8 bytes
//   frame: $0-24

TEXT ·dist4(SB),NOSPLIT,$0-24
    MOVQ q+0(FP),    SI   // query pointer
    MOVQ refs+8(FP), DI   // base of 4 × 32-byte reference vectors
    MOVQ out+16(FP), AX   // output [4]int32 pointer

    // ── Load query into ymm0 (stays for all 4 distance computations) ─────
    // VMOVDQU ymm0, [rsi]
    BYTE $0xC5; BYTE $0xFE; BYTE $0x6F; BYTE $0x06

    // ── Load ref[0] → ymm1; compute |query − ref[0]| ────────────────────
    // VMOVDQU ymm1, [rdi]
    BYTE $0xC5; BYTE $0xFE; BYTE $0x6F; BYTE $0x0F
    // VPSUBW ymm1, ymm0, ymm1  (ymm1 = ymm0 - ymm1, wrapping int16)
    BYTE $0xC5; BYTE $0xFD; BYTE $0xF9; BYTE $0xC9
    // VPABSW ymm1, ymm1
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x1D; BYTE $0xC9

    // ── Load ref[1] → ymm2; compute |query − ref[1]| ────────────────────
    // VMOVDQU ymm2, [rdi+32]
    BYTE $0xC5; BYTE $0xFE; BYTE $0x6F; BYTE $0x57; BYTE $0x20
    // VPSUBW ymm2, ymm0, ymm2
    BYTE $0xC5; BYTE $0xFD; BYTE $0xF9; BYTE $0xD2
    // VPABSW ymm2, ymm2
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x1D; BYTE $0xD2

    // ── Load ref[2] → ymm3; compute |query − ref[2]| ────────────────────
    // VMOVDQU ymm3, [rdi+64]
    BYTE $0xC5; BYTE $0xFE; BYTE $0x6F; BYTE $0x5F; BYTE $0x40
    // VPSUBW ymm3, ymm0, ymm3
    BYTE $0xC5; BYTE $0xFD; BYTE $0xF9; BYTE $0xDB
    // VPABSW ymm3, ymm3
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x1D; BYTE $0xDB

    // ── Load ref[3] → ymm4; compute |query − ref[3]| ────────────────────
    // VMOVDQU ymm4, [rdi+96]
    BYTE $0xC5; BYTE $0xFE; BYTE $0x6F; BYTE $0x67; BYTE $0x60
    // VPSUBW ymm4, ymm0, ymm4
    BYTE $0xC5; BYTE $0xFD; BYTE $0xF9; BYTE $0xE4
    // VPABSW ymm4, ymm4
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x1D; BYTE $0xE4

    // ── Widen int16 → int32 for each abs-diff vector ─────────────────────
    // Strategy: VEXTRACTI128 pulls high 128 bits into an XMM, then
    // VPMOVZXWD zero-extends both halves to 8×int32 each, VPADDD sums them.
    // We reuse ymm5 as scratch across all four vectors (safe: each step
    // consumes ymm5 via VPADDD before the next VEXTRACTI128 overwrites it).

    // ref[0]: extract ymm1 high → xmm5, widen both halves, sum
    // VEXTRACTI128 xmm5, ymm1, 1
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xCD; BYTE $0x01
    // VPMOVZXWD ymm1, xmm1  (low 8 int16 → 8 int32, overwrites ymm1 low half)
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xC9
    // VPMOVZXWD ymm5, xmm5  (high 8 int16 → 8 int32)
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xED
    // VPADDD ymm1, ymm1, ymm5  (8 int32 partial sums for ref[0])
    BYTE $0xC5; BYTE $0xF5; BYTE $0xFE; BYTE $0xCD

    // ref[1]: extract ymm2 high → xmm5, widen both halves, sum
    // VEXTRACTI128 xmm5, ymm2, 1
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xD5; BYTE $0x01
    // VPMOVZXWD ymm2, xmm2
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xD2
    // VPMOVZXWD ymm5, xmm5
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xED
    // VPADDD ymm2, ymm2, ymm5
    BYTE $0xC5; BYTE $0xED; BYTE $0xFE; BYTE $0xD5

    // ref[2]: extract ymm3 high → xmm5, widen both halves, sum
    // VEXTRACTI128 xmm5, ymm3, 1
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xDD; BYTE $0x01
    // VPMOVZXWD ymm3, xmm3
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xDB
    // VPMOVZXWD ymm5, xmm5
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xED
    // VPADDD ymm3, ymm3, ymm5
    BYTE $0xC5; BYTE $0xE5; BYTE $0xFE; BYTE $0xDD

    // ref[3]: extract ymm4 high → xmm5, widen both halves, sum
    // VEXTRACTI128 xmm5, ymm4, 1
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xE5; BYTE $0x01
    // VPMOVZXWD ymm4, xmm4
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xE4
    // VPMOVZXWD ymm5, xmm5
    BYTE $0xC4; BYTE $0xE2; BYTE $0x7D; BYTE $0x33; BYTE $0xED
    // VPADDD ymm4, ymm4, ymm5
    BYTE $0xC5; BYTE $0xDD; BYTE $0xFE; BYTE $0xE5

    // ── Extract high 4 int32 of each 8-wide sum into xmm5/6/7/0 ─────────
    // After this, xmm1-4 hold the LOW 4 int32, xmm5-7/xmm0 hold the HIGH 4.
    // ymm0 (query) is no longer needed so xmm0 is free for reuse.

    // VEXTRACTI128 xmm5, ymm1, 1  (high 4 int32 of ref[0])
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xCD; BYTE $0x01
    // VEXTRACTI128 xmm6, ymm2, 1  (high 4 int32 of ref[1])
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xD6; BYTE $0x01
    // VEXTRACTI128 xmm7, ymm3, 1  (high 4 int32 of ref[2])
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xDF; BYTE $0x01
    // VEXTRACTI128 xmm0, ymm4, 1  (high 4 int32 of ref[3]; reuses xmm0)
    BYTE $0xC4; BYTE $0xE3; BYTE $0x7D; BYTE $0x39; BYTE $0xE0; BYTE $0x01

    // ── Clear upper YMM bits; safe to use Plan9 SSE mnemonics from here ──
    // VZEROUPPER
    BYTE $0xC5; BYTE $0xF8; BYTE $0x77

    // ── Combine low + high for each vector (4 int32 → 4 int32) ──────────
    PADDD X5, X1    // X1 = 4 final int32 sums for ref[0]
    PADDD X6, X2    // X2 = 4 final int32 sums for ref[1]
    PADDD X7, X3    // X3 = 4 final int32 sums for ref[2]
    PADDD X0, X4    // X4 = 4 final int32 sums for ref[3]

    // ── Horizontal reduce 4 int32 → 1 int32, store result ────────────────
    // ref[0]: X1 → scalar → out[0]
    PSHUFD $0x4E, X1, X5;  PADDD X5, X1
    PSHUFD $0xB1, X1, X5;  PADDD X5, X1
    MOVL X1, 0(AX)

    // ref[1]: X2 → scalar → out[1]
    PSHUFD $0x4E, X2, X5;  PADDD X5, X2
    PSHUFD $0xB1, X2, X5;  PADDD X5, X2
    MOVL X2, 4(AX)

    // ref[2]: X3 → scalar → out[2]
    PSHUFD $0x4E, X3, X5;  PADDD X5, X3
    PSHUFD $0xB1, X3, X5;  PADDD X5, X3
    MOVL X3, 8(AX)

    // ref[3]: X4 → scalar → out[3]  (X5 is free; use X5 as scratch)
    PSHUFD $0x4E, X4, X5;  PADDD X5, X4
    PSHUFD $0xB1, X4, X5;  PADDD X5, X4
    MOVL X4, 12(AX)

    RET
