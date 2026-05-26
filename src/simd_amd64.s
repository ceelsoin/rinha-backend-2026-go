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
