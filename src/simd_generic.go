//go:build !amd64

package main

import "unsafe"

// sqDist16 is the pure-Go fallback for non-amd64 platforms.
// On amd64 the AVX2 version in simd_amd64.s is used instead.
func sqDist16(a, b *[stride]int16) int32 {
	var s int32
	for i := 0; i < stride; i++ {
		d := int32(a[i]) - int32(b[i])
		if d < 0 {
			d = -d
		}
		s += d
	}
	return s
}

// dist4 computes Manhattan distances from q to four consecutive reference
// vectors starting at refs, storing results in out[0..3].
func dist4(q *[stride]int16, refs *int16, out *[4]int32) {
	p := (*[4 * stride]int16)(unsafe.Pointer(refs))
	for j := 0; j < 4; j++ {
		v := (*[stride]int16)(unsafe.Pointer(&p[j*stride]))
		out[j] = sqDist16(q, v)
	}
}

// prefetchVec is a no-op on non-amd64 platforms.
func prefetchVec(_ *int16) {}
