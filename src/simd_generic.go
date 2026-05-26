//go:build !amd64

package main

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
