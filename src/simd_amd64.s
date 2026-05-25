#include "textflag.h"

// func sqDist16(a, b *[stride]int16) int32
//
// Computes squared L2 distance between two 16-element int16 vectors.
// stride = 16 × int16 = 32 bytes = 2 × 128-bit XMM registers.
//
// Values are quantized floats in [-10000, 10000]; max diff = 20000 which
// fits int16 so PSUBW (wrapping) is safe. PMADDWD produces int32, so
// (20000)² = 4×10⁸ < 2³¹ — no overflow.
//
// ABI0 (stack-based):
//   a   at  0(FP)  8 bytes
//   b   at  8(FP)  8 bytes
//   ret at 16(FP)  4 bytes
//   frame: $0-20

TEXT ·sqDist16(SB),NOSPLIT,$0-20
    MOVQ a+0(FP), SI
    MOVQ b+8(FP), DI

    // Load 32 bytes (16 × int16) for each pointer using two 128-bit loads.
    MOVOU  0(SI), X0   // a[0..7]
    MOVOU 16(SI), X2   // a[8..15]
    MOVOU  0(DI), X1   // b[0..7]
    MOVOU 16(DI), X3   // b[8..15]

    // Compute differences (wrapping int16 subtraction).
    PSUBW X1, X0      // X0 = a[0..7] − b[0..7]
    PSUBW X3, X2      // X2 = a[8..15] − b[8..15]

    // Square and sum adjacent pairs: int16 × int16 → int32 (SSE2 PMADDWD).
    // Go 1.22 assembler lacks the PMADDWD mnemonic; encode as raw bytes.
    // PMADDWD X0, X0 = 66 0F F5 C0  (ModRM: mod=3, reg=0, r/m=0)
    BYTE $0x66; BYTE $0x0F; BYTE $0xF5; BYTE $0xC0
    // PMADDWD X2, X2 = 66 0F F5 D2  (ModRM: mod=3, reg=2, r/m=2)
    BYTE $0x66; BYTE $0x0F; BYTE $0xF5; BYTE $0xD2

    // Combine the two halves.
    PADDD X2, X0      // X0 = 4 × int32 sums

    // Horizontal reduction: 4 int32s → 1 int32.
    PSHUFD $0x4E, X0, X1   // X1 = [X0[2], X0[3], X0[0], X0[1]]
    PADDD  X1, X0           // X0 = [X0[0]+X0[2], X0[1]+X0[3], ...]
    PSHUFD $0xB1, X0, X1   // X1 = [X0[1], X0[0], X0[3], X0[2]]
    PADDD  X1, X0           // X0[0] = sum of all 4

    MOVL X0, ret+16(FP)
    RET
