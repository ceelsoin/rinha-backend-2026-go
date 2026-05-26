//go:build amd64

package main

// sqDist16 computes the Manhattan (L1) distance between two stride-length int16
// vectors. Implemented in simd_amd64.s using a single AVX2 256-bit load per
// operand (vs two 128-bit loads in the old SSE version), VPABSW, VPMOVZXWD,
// and VPADDD — no multiplication, no loop. No heap allocation; pointers must
// remain valid during the call.
//
//go:noescape
func sqDist16(a, b *[stride]int16) int32
