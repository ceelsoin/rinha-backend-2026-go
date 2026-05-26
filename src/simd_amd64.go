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

// dist4 computes Manhattan distances from q to four consecutive stride-size
// reference vectors starting at refs, writing int32 results to out[0..3].
// Uses a single AVX2 query load amortised over 4 references; eliminates 3
// function-call overheads per 4 comparisons vs calling sqDist16 separately.
//
//go:noescape
func dist4(q *[stride]int16, refs *int16, out *[4]int32)

// prefetchVec issues a PREFETCHT0 hint for the cache line at p.
// Called from the ANN inner loop ~24 records ahead to hide DRAM latency.
//
//go:noescape
func prefetchVec(p *int16)
