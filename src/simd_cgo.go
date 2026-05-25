package main

// CGo wrapper for SIMD-accelerated IVF nearest-neighbour search.
// Kept in a separate file so import "C" doesn't pollute main.go.
// ONE CGo call per HTTP request: centroid scan + cluster scan done entirely
// in C, eliminating all per-cluster CGo transition overhead.

/*
#cgo CFLAGS: -O3
#cgo amd64 CFLAGS: -msse4.1
#include "simd.h"
*/
import "C"
import "unsafe"

// scoreRequest calls scoreRequestSIMD: one CGo transition for the full IVF
// nearest-neighbour search (centroid scan + cluster scan + repair).
func (idx *IVFIndex) scoreRequest(q *[stride]int16) int {
	return int(C.scoreRequestSIMD(
		(*C.short)(unsafe.Pointer(&idx.centroids[0])),
		C.int(idx.k),
		(*C.short)(unsafe.Pointer(&idx.vecs[0])),
		(*C.uint)(unsafe.Pointer(&idx.offsets[0])),
		(*C.uint)(unsafe.Pointer(&idx.counts[0])),
		(*C.uchar)(unsafe.Pointer(&idx.labels[0])),
		(*C.short)(unsafe.Pointer(&q[0])),
		C.int(stride),
		C.int(nProbe),
		C.int(nProbeRepair),
		C.int(nNeigh),
	))
}
