//go:build amd64

package main

// sqDist16 computes the squared L2 distance between two stride-length int16
// vectors. Implemented in simd_amd64.s using SSE4.1 PMADDWD for 4× throughput
// vs scalar. No heap allocation; pointers must remain valid during the call.
//
//go:noescape
func sqDist16(a, b *[stride]int16) int32
