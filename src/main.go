package main

// Rinha de Backend 2026 — Fraud Detection via Vector Search
// Single-file Go backend using fasthttp + IVF approximate KNN.
//
// Usage:
//   Build index: ./rinha build <references.json.gz> <output.bin> <norm.json> <mcc.json>
//   Run server:  ./rinha serve <index.bin> <norm.json> <mcc.json> [port]

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"unsafe"

	"github.com/valyala/fasthttp"
	"github.com/valyala/fastjson"
)

// ── Constants ────────────────────────────────────────────────────────────────

const (
	dims         = 14   // feature dimensions (14 input features)
	stride       = 16   // storage stride per vector: padded to 16 int16 for 128-bit SIMD alignment
	ivfK         = 2048 // number of IVF clusters
	nProbe       = 8    // clusters probed per query (fast path)
	nProbeRepair = 48   // total clusters probed when result is uncertain
	nNeigh       = 5    // k-NN neighbors
	trainSample  = 50000
	trainIters   = 50
)

var ivfMagic = [8]byte{'G', 'O', 'I', 'V', 'F', '0', '2', '7'} // v2: stride=16 (SIMD-padded)

// Pre-built HTTP responses for all possible fraud counts (0..5).
// Index = number of fraud neighbors found.
var responses = [nNeigh + 1][]byte{
	[]byte(`{"approved": true, "fraud_score": 0.00}`),
	[]byte(`{"approved": true, "fraud_score": 0.20}`),
	[]byte(`{"approved": true, "fraud_score": 0.40}`),
	[]byte(`{"approved": false, "fraud_score": 0.60}`),
	[]byte(`{"approved": false, "fraud_score": 0.80}`),
	[]byte(`{"approved": false, "fraud_score": 1.00}`),
}

// ── IVF Index ────────────────────────────────────────────────────────────────

// IVFIndex holds the approximate nearest-neighbour index.
// All int16 values are quantized: float × 10000.
type IVFIndex struct {
	n         int      // total reference vectors
	k         int      // number of clusters
	centroids []int16  // k×dims, row-major
	counts    []uint32 // per-cluster count
	offsets   []uint32 // per-cluster start position in vecs/labels
	vecs      []int16  // n×dims, organized by cluster
	labels    []uint8  // n: 1=fraud, 0=legit
}

// ── Normalization config ──────────────────────────────────────────────────────

// NormConfig holds pre-computed reciprocals of the normalization constants.
type NormConfig struct {
	InvMaxAmount            float32
	InvMaxInstallments      float32
	InvAmountVsAvgRatio     float32
	InvMaxMinutes           float32
	InvMaxKm                float32
	InvMaxTxCount24h        float32
	InvMaxMerchantAvgAmount float32
}

// ── Math helpers ─────────────────────────────────────────────────────────────

func clamp32(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// quantizeF64 converts a normalized float64 to int16 (×10000), clamped to [-1,1].
func quantizeF64(v float64) int16 {
	if v < -1.0 {
		v = -1.0
	}
	if v > 1.0 {
		v = 1.0
	}
	r := v * 10000.0
	if r >= 0 {
		r += 0.5
	} else {
		r -= 0.5
	}
	return int16(r)
}

// quantizeF32 converts a float32 (already in [-1,1]) to int16 (×10000).
func quantizeF32(v float32) int16 {
	if v < -1.0 {
		v = -1.0
	}
	if v > 1.0 {
		v = 1.0
	}
	return int16(math.Round(float64(v) * 10000.0))
}

// sqDist14 computes squared Euclidean distance between a fixed [14]int16 and a slice.
// Manually unrolled to help the compiler avoid bounds checks.
func sqDist14(a [dims]int16, b []int16) int64 {
	_ = b[13] // bounds check hint
	d0 := int64(a[0]) - int64(b[0])
	d1 := int64(a[1]) - int64(b[1])
	d2 := int64(a[2]) - int64(b[2])
	d3 := int64(a[3]) - int64(b[3])
	d4 := int64(a[4]) - int64(b[4])
	d5 := int64(a[5]) - int64(b[5])
	d6 := int64(a[6]) - int64(b[6])
	d7 := int64(a[7]) - int64(b[7])
	d8 := int64(a[8]) - int64(b[8])
	d9 := int64(a[9]) - int64(b[9])
	d10 := int64(a[10]) - int64(b[10])
	d11 := int64(a[11]) - int64(b[11])
	d12 := int64(a[12]) - int64(b[12])
	d13 := int64(a[13]) - int64(b[13])
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 +
		d4*d4 + d5*d5 + d6*d6 + d7*d7 +
		d8*d8 + d9*d9 + d10*d10 + d11*d11 +
		d12*d12 + d13*d13
}

// sqDist14Slice computes squared distance between two int16 slices (each len≥14).
func sqDist14Slice(a, b []int16) int64 {
	_ = a[13]
	_ = b[13]
	d0 := int64(a[0]) - int64(b[0])
	d1 := int64(a[1]) - int64(b[1])
	d2 := int64(a[2]) - int64(b[2])
	d3 := int64(a[3]) - int64(b[3])
	d4 := int64(a[4]) - int64(b[4])
	d5 := int64(a[5]) - int64(b[5])
	d6 := int64(a[6]) - int64(b[6])
	d7 := int64(a[7]) - int64(b[7])
	d8 := int64(a[8]) - int64(b[8])
	d9 := int64(a[9]) - int64(b[9])
	d10 := int64(a[10]) - int64(b[10])
	d11 := int64(a[11]) - int64(b[11])
	d12 := int64(a[12]) - int64(b[12])
	d13 := int64(a[13]) - int64(b[13])
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 +
		d4*d4 + d5*d5 + d6*d6 + d7*d7 +
		d8*d8 + d9*d9 + d10*d10 + d11*d11 +
		d12*d12 + d13*d13
}

// ── Time helpers ─────────────────────────────────────────────────────────────

// fastWeekday returns weekday (0=Mon, 6=Sun) using Tomohiko Sakamoto's algorithm.
func fastWeekday(y, m, d int) int {
	t := [12]int{0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4}
	if m < 3 {
		y--
	}
	dow := (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7
	return (dow + 6) % 7 // convert Sunday=0 to Monday=0
}

// fastEpoch converts a civil date+time to Unix epoch seconds (Howard Hinnant).
func fastEpoch(y, m, d, h, min, s int) int64 {
	if m <= 2 {
		y--
		m += 9
	} else {
		m -= 3
	}
	era := y / 400
	if y < 0 {
		era = (y - 399) / 400
	}
	yoe := y - era*400
	doy := (153*m+2)/5 + d - 1
	doe := yoe*365 + yoe/4 - yoe/100 + doy
	days := int64(era)*146097 + int64(doe) - 719468
	return days*86400 + int64(h)*3600 + int64(min)*60 + int64(s)
}

// parseTS parses "2006-01-02T15:04:05Z" (no allocation).
// Returns (hour 0-23, weekday 0-6, epochSeconds).
func parseTS(s []byte) (int, int, int64) {
	if len(s) < 19 {
		return 0, 0, 0
	}
	y := int(s[0]-'0')*1000 + int(s[1]-'0')*100 + int(s[2]-'0')*10 + int(s[3]-'0')
	mo := int(s[5]-'0')*10 + int(s[6]-'0')
	d := int(s[8]-'0')*10 + int(s[9]-'0')
	h := int(s[11]-'0')*10 + int(s[12]-'0')
	mi := int(s[14]-'0')*10 + int(s[15]-'0')
	sec := int(s[17]-'0')*10 + int(s[18]-'0')
	return h, fastWeekday(y, mo, d), fastEpoch(y, mo, d, h, mi, sec)
}

// ── IVF search ───────────────────────────────────────────────────────────────

// centHeap is a fixed-capacity max-heap over centroid distances used to track
// the n nearest centroids while scanning all ivfK centroids once.
// Capacity is always ≤ nProbeRepair so the backing arrays live on the stack.
type centHeap struct {
	d    [nProbeRepair]int32
	idx  [nProbeRepair]int32
	size int
	n    int // capacity: nProbe or nProbeRepair
}

func (h *centHeap) maxD() int32 {
	if h.size < h.n {
		return math.MaxInt32
	}
	return h.d[0]
}

func (h *centHeap) siftUp(i int) {
	for i > 0 {
		p := (i - 1) >> 1
		if h.d[i] > h.d[p] {
			h.d[i], h.d[p] = h.d[p], h.d[i]
			h.idx[i], h.idx[p] = h.idx[p], h.idx[i]
			i = p
		} else {
			break
		}
	}
}

func (h *centHeap) siftDown(i int) {
	n := h.size
	for {
		l, r, lg := 2*i+1, 2*i+2, i
		if l < n && h.d[l] > h.d[lg] {
			lg = l
		}
		if r < n && h.d[r] > h.d[lg] {
			lg = r
		}
		if lg == i {
			break
		}
		h.d[i], h.d[lg] = h.d[lg], h.d[i]
		h.idx[i], h.idx[lg] = h.idx[lg], h.idx[i]
		i = lg
	}
}

func (h *centHeap) insert(d int32, ci int32) {
	if h.size < h.n {
		h.d[h.size] = d
		h.idx[h.size] = ci
		h.size++
		h.siftUp(h.size - 1)
	} else if d < h.d[0] {
		h.d[0] = d
		h.idx[0] = ci
		h.siftDown(0)
	}
}

// neighHeap is a fixed-capacity max-heap for the nNeigh=5 nearest neighbors.
// The maximum distance element is always at index 0 (root of max-heap).
type neighHeap struct {
	d    [nNeigh]int32
	lbl  [nNeigh]uint8
	size int
}

func (h *neighHeap) maxD() int32 {
	if h.size < nNeigh {
		return math.MaxInt32
	}
	return h.d[0]
}

func (h *neighHeap) siftUp(i int) {
	for i > 0 {
		p := (i - 1) >> 1
		if h.d[i] > h.d[p] {
			h.d[i], h.d[p] = h.d[p], h.d[i]
			h.lbl[i], h.lbl[p] = h.lbl[p], h.lbl[i]
			i = p
		} else {
			break
		}
	}
}

func (h *neighHeap) siftDown(i int) {
	for {
		l, r, lg := 2*i+1, 2*i+2, i
		if l < nNeigh && h.d[l] > h.d[lg] {
			lg = l
		}
		if r < nNeigh && h.d[r] > h.d[lg] {
			lg = r
		}
		if lg == i {
			break
		}
		h.d[i], h.d[lg] = h.d[lg], h.d[i]
		h.lbl[i], h.lbl[lg] = h.lbl[lg], h.lbl[i]
		i = lg
	}
}

func (h *neighHeap) insert(d int32, lbl uint8) {
	if h.size < nNeigh {
		h.d[h.size] = d
		h.lbl[h.size] = lbl
		h.size++
		h.siftUp(h.size - 1)
	} else {
		h.d[0] = d
		h.lbl[0] = lbl
		h.siftDown(0)
	}
}

func (h *neighHeap) fraudCount() int {
	n := 0
	for i := 0; i < h.size; i++ {
		n += int(h.lbl[i])
	}
	return n
}

// scoreRequest performs the full IVF nearest-neighbour search in pure Go.
// sqDist16 is called via Go assembly (SSE4.1) on amd64, scalar fallback elsewhere.
// No CGo: runs on the single GOMAXPROCS=1 OS thread, no CFS throttle contention.
func (idx *IVFIndex) scoreRequest(q *[stride]int16) int {
	// ── Phase 1: find nProbe nearest centroids ────────────────────────────────
	var ch centHeap
	ch.n = nProbe
	for ci := 0; ci < ivfK; ci++ {
		c := (*[stride]int16)(unsafe.Pointer(&idx.centroids[ci*stride]))
		ch.insert(sqDist16(q, c), int32(ci))
	}

	// ── Phase 2: scan selected clusters, track nNeigh nearest neighbors ───────
	var nh neighHeap
	for pi := 0; pi < ch.size; pi++ {
		ci := int(ch.idx[pi])
		start := int(idx.offsets[ci])
		count := int(idx.counts[ci])
		maxD := nh.maxD()
		for vi := start; vi < start+count; vi++ {
			v := (*[stride]int16)(unsafe.Pointer(&idx.vecs[vi*stride]))
			if d := sqDist16(q, v); d < maxD {
				nh.insert(d, idx.labels[vi])
				maxD = nh.maxD()
			}
		}
	}

	fc := nh.fraudCount()

	// ── Phase 3: repair when result is borderline (2 or 3 fraud among 5) ──────
	// Re-scan all centroids with a wider probe to reduce false positives/negatives.
	if fc == 2 || fc == 3 {
		var ch2 centHeap
		ch2.n = nProbeRepair
		for ci := 0; ci < ivfK; ci++ {
			c := (*[stride]int16)(unsafe.Pointer(&idx.centroids[ci*stride]))
			ch2.insert(sqDist16(q, c), int32(ci))
		}
		nh = neighHeap{} // reset
		for pi := 0; pi < ch2.size; pi++ {
			ci := int(ch2.idx[pi])
			start := int(idx.offsets[ci])
			count := int(idx.counts[ci])
			maxD := nh.maxD()
			for vi := start; vi < start+count; vi++ {
				v := (*[stride]int16)(unsafe.Pointer(&idx.vecs[vi*stride]))
				if d := sqDist16(q, v); d < maxD {
					nh.insert(d, idx.labels[vi])
					maxD = nh.maxD()
				}
			}
		}
		fc = nh.fraudCount()
	}

	return fc
}

// getFraudCount is the HTTP handler entry point into the IVF search.
func (idx *IVFIndex) getFraudCount(q [stride]int16) int {
	return idx.scoreRequest(&q)
}

// ── K-means builder ─────────────────────────────────────────────────────────

// kmeansDistF64 computes squared Euclidean distance between float64 centroid and int16 vector.
func kmeansDistF64(c []float64, v []int16) float64 {
	var s float64
	for i := 0; i < dims; i++ {
		d := c[i] - float64(v[i])
		s += d * d
	}
	return s
}

// nearestCentroidIdx returns the index of the nearest centroid to vec.
func nearestCentroidIdx(vec []int16, centroids [][]float64) int {
	best := 0
	bestD := kmeansDistF64(centroids[0], vec)
	for c := 1; c < len(centroids); c++ {
		if d := kmeansDistF64(centroids[c], vec); d < bestD {
			bestD = d
			best = c
		}
	}
	return best
}

// buildIVF runs k-means on a sample of the training data and builds the IVF index.
func buildIVF(allVecs []int16, allLabels []uint8, k, sample, iters int) *IVFIndex {
	n := len(allVecs) / dims
	log.Printf("[build] n=%d k=%d sample=%d iters=%d", n, k, sample, iters)

	rng := rand.New(rand.NewSource(0x52696E6861)) //nolint

	// Sample indices (without replacement is nice but not required for k-means)
	sampleIdx := make([]int, sample)
	for i := range sampleIdx {
		sampleIdx[i] = rng.Intn(n)
	}

	// ── k-means++ initialization ──────────────────────────────────────────
	log.Printf("[build] k-means++ init...")
	centroids := make([][]float64, k)
	for i := range centroids {
		centroids[i] = make([]float64, dims)
	}
	first := sampleIdx[rng.Intn(sample)]
	for d := 0; d < dims; d++ {
		centroids[0][d] = float64(allVecs[first*dims+d])
	}

	dmin := make([]float64, sample)
	for i := range dmin {
		dmin[i] = math.MaxFloat64
	}

	for c := 1; c < k; c++ {
		prev := centroids[c-1]
		total := 0.0
		for i, si := range sampleIdx {
			d := kmeansDistF64(prev, allVecs[si*dims:si*dims+dims])
			if d < dmin[i] {
				dmin[i] = d
			}
			total += dmin[i]
		}
		if total == 0 {
			ri := sampleIdx[rng.Intn(sample)]
			for d := 0; d < dims; d++ {
				centroids[c][d] = float64(allVecs[ri*dims+d])
			}
			continue
		}
		target := rng.Float64() * total
		acc := 0.0
		chosen := sampleIdx[sample-1]
		for i, si := range sampleIdx {
			acc += dmin[i]
			if acc >= target {
				chosen = si
				break
			}
		}
		for d := 0; d < dims; d++ {
			centroids[c][d] = float64(allVecs[chosen*dims+d])
		}
		if c%256 == 0 {
			log.Printf("[build]   init %d/%d", c, k)
		}
	}

	// ── K-means training on sample ────────────────────────────────────────
	assignment := make([]int, sample)
	for iter := 0; iter < iters; iter++ {
		changed := 0
		for i, si := range sampleIdx {
			c := nearestCentroidIdx(allVecs[si*dims:si*dims+dims], centroids)
			if c != assignment[i] {
				changed++
				assignment[i] = c
			}
		}
		// Recompute centroids
		sums := make([]float64, k*dims)
		counts := make([]int, k)
		for i, si := range sampleIdx {
			c := assignment[i]
			counts[c]++
			for d := 0; d < dims; d++ {
				sums[c*dims+d] += float64(allVecs[si*dims+d])
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				ri := sampleIdx[rng.Intn(sample)]
				for d := 0; d < dims; d++ {
					centroids[c][d] = float64(allVecs[ri*dims+d])
				}
			} else {
				inv := 1.0 / float64(counts[c])
				for d := 0; d < dims; d++ {
					centroids[c][d] = sums[c*dims+d] * inv
				}
			}
		}
		log.Printf("[build]   iter %d/%d changed=%d", iter+1, iters, changed)
		if changed == 0 {
			break
		}
	}

	// ── Assign all N vectors to nearest cluster (parallel) ────────────────
	log.Printf("[build] assigning %d vectors...", n)
	fullAssign := make([]int, n)
	nw := runtime.NumCPU()
	chunk := (n + nw - 1) / nw
	var wg sync.WaitGroup
	for w := 0; w < nw; w++ {
		s, e := w*chunk, (w+1)*chunk
		if e > n {
			e = n
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				fullAssign[i] = nearestCentroidIdx(allVecs[i*dims:i*dims+dims], centroids)
			}
		}(s, e)
	}
	wg.Wait()

	// ── Build cluster layout ──────────────────────────────────────────────
	log.Printf("[build] building cluster layout...")
	clusterCounts := make([]uint32, k)
	for _, c := range fullAssign {
		clusterCounts[c]++
	}
	clusterOffsets := make([]uint32, k)
	var off uint32
	for c := 0; c < k; c++ {
		clusterOffsets[c] = off
		off += clusterCounts[c]
	}

	// Write vectors in cluster order with stride=16 padding (last 2 int16 stay 0).
	flatVecs := make([]int16, n*stride) // Go zero-init: elements dims..stride-1 are 0
	flatLabels := make([]uint8, n)
	cursor := make([]uint32, k)
	copy(cursor, clusterOffsets)
	for i := 0; i < n; i++ {
		c := fullAssign[i]
		pos := cursor[c]
		cursor[c]++
		copy(flatVecs[int(pos)*stride:int(pos)*stride+dims], allVecs[i*dims:i*dims+dims])
		flatLabels[pos] = allLabels[i]
	}

	// Quantize centroids to int16 with stride=16 padding.
	centI16 := make([]int16, k*stride) // last 2 per centroid stay 0
	for c := 0; c < k; c++ {
		for d := 0; d < dims; d++ {
			v := centroids[c][d]
			if v < -10000 {
				v = -10000
			}
			if v > 10000 {
				v = 10000
			}
			centI16[c*stride+d] = int16(math.Round(v))
		}
	}

	log.Printf("[build] IVF built: %d vectors, %d clusters", n, k)
	return &IVFIndex{
		n:         n,
		k:         k,
		centroids: centI16,
		counts:    clusterCounts,
		offsets:   clusterOffsets,
		vecs:      flatVecs,
		labels:    flatLabels,
	}
}

// ── Binary I/O (little-endian, platform must be LE — all amd64 are) ──────────

// int16ToBytes reinterprets []int16 as []byte (unsafe, LE only).
func int16ToBytes(s []int16) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*2)
}

// uint32ToBytes reinterprets []uint32 as []byte.
func uint32ToBytes(s []uint32) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*4)
}

func writeIndex(path string, idx *IVFIndex) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	// Header
	w.Write(ivfMagic[:])
	binary.Write(w, binary.LittleEndian, uint32(1))      // version
	binary.Write(w, binary.LittleEndian, uint32(idx.n))
	binary.Write(w, binary.LittleEndian, uint32(idx.k))

	// Centroids, counts, offsets
	w.Write(int16ToBytes(idx.centroids))
	w.Write(uint32ToBytes(idx.counts))
	w.Write(uint32ToBytes(idx.offsets))

	// Vectors and labels
	w.Write(int16ToBytes(idx.vecs))
	w.Write(idx.labels)

	return w.Flush()
}

func readIndex(path string) (*IVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	// Header
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, err
	}
	if magic != ivfMagic {
		return nil, fmt.Errorf("bad magic")
	}
	var version, n32, k32 uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &n32); err != nil {
		return nil, fmt.Errorf("read n: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &k32); err != nil {
		return nil, fmt.Errorf("read k: %w", err)
	}
	n, k := int(n32), int(k32)

	readI16 := func(count int) ([]int16, error) {
		s := make([]int16, count)
		_, err := io.ReadFull(r, int16ToBytes(s))
		return s, err
	}
	readU32 := func(count int) ([]uint32, error) {
		s := make([]uint32, count)
		_, err := io.ReadFull(r, uint32ToBytes(s))
		return s, err
	}

	centroids, err := readI16(k * stride)
	if err != nil {
		return nil, fmt.Errorf("read centroids: %w", err)
	}
	counts, err := readU32(k)
	if err != nil {
		return nil, fmt.Errorf("read counts: %w", err)
	}
	offsets, err := readU32(k)
	if err != nil {
		return nil, fmt.Errorf("read offsets: %w", err)
	}
	vecs, err := readI16(n * stride)
	if err != nil {
		return nil, fmt.Errorf("read vecs: %w", err)
	}
	labels := make([]uint8, n)
	if _, err := io.ReadFull(r, labels); err != nil {
		return nil, fmt.Errorf("read labels: %w", err)
	}

	log.Printf("[serve] loaded index: n=%d k=%d", n, k)
	return &IVFIndex{
		n: n, k: k,
		centroids: centroids,
		counts:    counts,
		offsets:   offsets,
		vecs:      vecs,
		labels:    labels,
	}, nil
}

// ── References loader ────────────────────────────────────────────────────────

// loadReferences reads references.json or references.json.gz.
// Returns quantized int16 vectors and uint8 labels (1=fraud, 0=legit).
func loadReferences(path string) ([]int16, []uint8, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	var reader io.Reader = f
	// Auto-detect gzip
	var gzr *gzip.Reader
	buf := make([]byte, 2)
	if _, err := io.ReadFull(f, buf); err != nil {
		return nil, nil, err
	}
	f.Seek(0, io.SeekStart)
	if buf[0] == 0x1f && buf[1] == 0x8b {
		gzr, err = gzip.NewReader(f)
		if err != nil {
			return nil, nil, err
		}
		defer gzr.Close()
		reader = gzr
	}

	log.Printf("[build] parsing references (streaming JSON)...")

	type refEntry struct {
		Vector [dims]float64 `json:"vector"`
		Label  string        `json:"label"`
	}

	vecs := make([]int16, 0, 3_000_000*dims)
	labels := make([]uint8, 0, 3_000_000)

	dec := json.NewDecoder(bufio.NewReaderSize(reader, 1<<20))
	// Consume opening '['
	if _, err := dec.Token(); err != nil {
		return nil, nil, err
	}
	var entry refEntry
	count := 0
	for dec.More() {
		entry.Label = ""
		if err := dec.Decode(&entry); err != nil {
			return nil, nil, fmt.Errorf("decode entry %d: %w", count, err)
		}
		for _, v := range entry.Vector {
			vecs = append(vecs, quantizeF64(v))
		}
		if entry.Label == "fraud" {
			labels = append(labels, 1)
		} else {
			labels = append(labels, 0)
		}
		count++
		if count%500_000 == 0 {
			log.Printf("[build]   parsed %d entries", count)
		}
	}
	log.Printf("[build] loaded %d reference vectors", count)
	return vecs, labels, nil
}

// ── Config loaders ───────────────────────────────────────────────────────────

func loadNormConfig(path string) (NormConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return NormConfig{}, err
	}
	var raw struct {
		MaxAmount            float64 `json:"max_amount"`
		MaxInstallments      float64 `json:"max_installments"`
		AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
		MaxMinutes           float64 `json:"max_minutes"`
		MaxKm                float64 `json:"max_km"`
		MaxTxCount24h        float64 `json:"max_tx_count_24h"`
		MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return NormConfig{}, err
	}
	return NormConfig{
		InvMaxAmount:            float32(1.0 / raw.MaxAmount),
		InvMaxInstallments:      float32(1.0 / raw.MaxInstallments),
		InvAmountVsAvgRatio:     float32(1.0 / raw.AmountVsAvgRatio),
		InvMaxMinutes:           float32(1.0 / raw.MaxMinutes),
		InvMaxKm:                float32(1.0 / raw.MaxKm),
		InvMaxTxCount24h:        float32(1.0 / raw.MaxTxCount24h),
		InvMaxMerchantAvgAmount: float32(1.0 / raw.MaxMerchantAvgAmount),
	}, nil
}

func loadMCCRisk(path string) (map[string]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw map[string]float64
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	m := make(map[string]float32, len(raw))
	for k, v := range raw {
		m[k] = float32(v)
	}
	return m, nil
}

// ── HTTP handler ─────────────────────────────────────────────────────────────

var parserPool fastjson.ParserPool

// Server state (read-only after startup).
var (
	globalIdx     *IVFIndex
	globalMCCRisk map[string]float32
	globalNorm    NormConfig
)

func handleReady(ctx *fasthttp.RequestCtx) {
	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetBodyString("OK")
}

func handleFraudScore(ctx *fasthttp.RequestCtx) {
	p := parserPool.Get()
	defer parserPool.Put(p)

	v, err := p.ParseBytes(ctx.PostBody())
	if err != nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}

	// ── Extract fields (fastjson returns []byte backed by input — no alloc) ──

	txn := v.Get("transaction")
	cust := v.Get("customer")
	merch := v.Get("merchant")
	term := v.Get("terminal")

	if txn == nil || cust == nil || merch == nil || term == nil {
		ctx.SetStatusCode(fasthttp.StatusBadRequest)
		return
	}

	amount := float32(txn.GetFloat64("amount"))
	installments := float32(txn.GetInt("installments"))
	reqAt := txn.GetStringBytes("requested_at")

	avgAmount := float32(cust.GetFloat64("avg_amount"))
	txCount24h := float32(cust.GetInt("tx_count_24h"))
	knownMerchants := cust.GetArray("known_merchants")

	merchantID := merch.GetStringBytes("id")
	mccBytes := merch.GetStringBytes("mcc")
	merchantAvg := float32(merch.GetFloat64("avg_amount"))

	isOnline := term.GetBool("is_online")
	cardPresent := term.GetBool("card_present")
	kmFromHome := float32(term.GetFloat64("km_from_home"))

	lastTx := v.Get("last_transaction")

	// ── Normalize → 14-dim float32 vector ────────────────────────────────────
	cfg := globalNorm
	var vec [dims]float32

	// 0. amount / max_amount
	vec[0] = clamp32(amount * cfg.InvMaxAmount)

	// 1. installments / max_installments
	vec[1] = clamp32(installments * cfg.InvMaxInstallments)

	// 2. (amount / avg_amount) / amount_vs_avg_ratio
	if avgAmount <= 0 {
		avgAmount = 1
	}
	vec[2] = clamp32((amount / avgAmount) * cfg.InvAmountVsAvgRatio)

	// 3. hour_of_day / 23  &  4. day_of_week / 6
	hour, weekday, epochSec := parseTS(reqAt)
	vec[3] = float32(hour) / 23.0
	vec[4] = float32(weekday) / 6.0

	// 5 & 6. last_transaction (or -1 sentinel)
	if lastTx == nil || lastTx.Type() == fastjson.TypeNull {
		vec[5] = -1.0
		vec[6] = -1.0
	} else {
		lastTs := lastTx.GetStringBytes("timestamp")
		kmFromCurrent := float32(lastTx.GetFloat64("km_from_current"))
		_, _, lastEpoch := parseTS(lastTs)
		minutes := float32(epochSec-lastEpoch) / 60.0
		vec[5] = clamp32(minutes * cfg.InvMaxMinutes)
		vec[6] = clamp32(kmFromCurrent * cfg.InvMaxKm)
	}

	// 7. km_from_home / max_km
	vec[7] = clamp32(kmFromHome * cfg.InvMaxKm)

	// 8. tx_count_24h / max_tx_count_24h
	vec[8] = clamp32(txCount24h * cfg.InvMaxTxCount24h)

	// 9. is_online
	if isOnline {
		vec[9] = 1.0
	}

	// 10. card_present
	if cardPresent {
		vec[10] = 1.0
	}

	// 11. unknown_merchant (1 = unknown, 0 = known)
	known := false
	for _, km := range knownMerchants {
		if bytes.Equal(km.GetStringBytes(), merchantID) {
			known = true
			break
		}
	}
	if !known {
		vec[11] = 1.0
	}

	// 12. mcc_risk (default 0.5)
	if risk, ok := globalMCCRisk[string(mccBytes)]; ok { // string([]byte) in map lookup: no alloc (Go compiler opt)
		vec[12] = risk
	} else {
		vec[12] = 0.5
	}

	// 13. merchant avg_amount / max_merchant_avg_amount
	vec[13] = clamp32(merchantAvg * cfg.InvMaxMerchantAvgAmount)

	// ── Quantize to int16 ─────────────────────────────────────────────────────
	var q [stride]int16 // last 2 elements stay 0 (SIMD padding)
	for i, fv := range vec {
		q[i] = quantizeF32(fv)
	}

	// ── KNN search ───────────────────────────────────────────────────────────
	fraudCount := globalIdx.getFraudCount(q)

	// ── Return pre-computed response ─────────────────────────────────────────
	ctx.SetContentTypeBytes([]byte("application/json"))
	ctx.SetBody(responses[fraudCount])
}

func requestHandler(ctx *fasthttp.RequestCtx) {
	path := ctx.Path()
	method := ctx.Method()

	switch {
	case bytes.Equal(path, []byte("/ready")) && bytes.Equal(method, []byte("GET")):
		handleReady(ctx)
	case bytes.Equal(path, []byte("/fraud-score")) && bytes.Equal(method, []byte("POST")):
		handleFraudScore(ctx)
	default:
		ctx.SetStatusCode(fasthttp.StatusNotFound)
	}
}

// ── Build mode ───────────────────────────────────────────────────────────────

func cmdBuild(args []string) {
	fs := flag.NewFlagSet("build", flag.ExitOnError)
	kFlag := fs.Int("k", ivfK, "number of IVF clusters")
	sFlag := fs.Int("sample", trainSample, "k-means sample size")
	iFlag := fs.Int("iters", trainIters, "k-means iterations")
	fs.Parse(args)

	remaining := fs.Args()
	if len(remaining) < 4 {
		fmt.Fprintln(os.Stderr, "Usage: rinha build <references.json[.gz]> <output.bin> <norm.json> <mcc.json> [flags]")
		os.Exit(1)
	}
	refPath, outPath, normPath, mccPath := remaining[0], remaining[1], remaining[2], remaining[3]

	// Validate config files load correctly
	if _, err := loadNormConfig(normPath); err != nil {
		log.Fatalf("norm config: %v", err)
	}
	if _, err := loadMCCRisk(mccPath); err != nil {
		log.Fatalf("mcc risk: %v", err)
	}

	// Load references
	vecs, labels, err := loadReferences(refPath)
	if err != nil {
		log.Fatalf("load references: %v", err)
	}

	// Build index
	idx := buildIVF(vecs, labels, *kFlag, *sFlag, *iFlag)

	// Write index
	log.Printf("[build] writing %s...", outPath)
	if err := writeIndex(outPath, idx); err != nil {
		log.Fatalf("write index: %v", err)
	}
	log.Printf("[build] done.")
}

// ── Serve mode ───────────────────────────────────────────────────────────────

// preWarmIndex reads every OS page of the vectors and labels slices so the
// kernel faults them into RAM before the server starts accepting connections.
// Without this, cold-start requests take extra latency for page faults.
func preWarmIndex(idx *IVFIndex) {
	const pageStride = 2048 // touch one int16 per 4KB page (2048 int16 = 4096 bytes)
	var acc int32
	for i := 0; i < len(idx.vecs); i += pageStride {
		acc += int32(idx.vecs[i])
	}
	const labelStride = 4096 // one byte per 4KB page
	for i := 0; i < len(idx.labels); i += labelStride {
		acc += int32(idx.labels[i])
	}
	_ = acc // prevent compiler from optimising the reads away
	log.Printf("[serve] pre-warmed %d MB of index vectors", len(idx.vecs)*2>>20)
}

func cmdServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	port := fs.Int("port", 8080, "HTTP listen port")
	fs.Parse(args)

	remaining := fs.Args()
	if len(remaining) < 3 {
		fmt.Fprintln(os.Stderr, "Usage: rinha serve <index.bin> <norm.json> <mcc.json> [flags]")
		os.Exit(1)
	}
	idxPath, normPath, mccPath := remaining[0], remaining[1], remaining[2]

	// Override port from env (container-friendly)
	if p := os.Getenv("SERVER_PORT"); p != "" {
		fmt.Sscan(p, port)
	}
	// Unix socket path (optional; preferred over TCP when set)
	socketPath := os.Getenv("SOCKET_PATH")

	// Load config
	var err error
	globalNorm, err = loadNormConfig(normPath)
	if err != nil {
		log.Fatalf("norm config: %v", err)
	}
	globalMCCRisk, err = loadMCCRisk(mccPath)
	if err != nil {
		log.Fatalf("mcc risk: %v", err)
	}

	// Lock to 1 OS thread immediately — each container has exactly 1 CPU (cpuset).
	runtime.GOMAXPROCS(1)

	// Disable GC: hot path has near-zero allocations (pooled parsers, stack arrays).
	// Eliminates GC-induced latency spikes that inflate p99.
	debug.SetGCPercent(-1)

	// Load index
	globalIdx, err = readIndex(idxPath)
	if err != nil {
		log.Fatalf("read index: %v", err)
	}

	// Fault all 84MB of quantized vectors into RAM before accepting connections.
	// Prevents OS page-fault latency spikes on first requests.
	preWarmIndex(globalIdx)

	srv := &fasthttp.Server{
		Handler:                       requestHandler,
		Name:                          "rinha",
		NoDefaultDate:                 true,
		NoDefaultServerHeader:         true,
		DisableHeaderNamesNormalizing: true,
		MaxConnsPerIP:                 0,
		Concurrency:                   4096,
		ReadBufferSize:                512,
		WriteBufferSize:               256,
		ReadTimeout:                   5000000000,
		WriteTimeout:                  5000000000,
		TCPKeepalive:                  true,
	}

	if socketPath != "" {
		// Unix domain socket: eliminate TCP stack overhead (~10-20µs per request).
		os.Remove(socketPath) // clean up any leftover socket from previous run
		log.Printf("[serve] listening on unix:%s", socketPath)
		if err := srv.ListenAndServeUNIX(socketPath, 0777); err != nil {
			log.Fatalf("server unix: %v", err)
		}
	} else {
		log.Printf("[serve] listening on :%d", *port)
		if err := srv.ListenAndServe(fmt.Sprintf(":%d", *port)); err != nil {
			log.Fatalf("server tcp: %v", err)
		}
	}
}

// ── Main ─────────────────────────────────────────────────────────────────────

func main() {
	log.SetFlags(log.Ltime | log.Lmsgprefix)

	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: rinha <build|serve> [args...]")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "build":
		cmdBuild(os.Args[2:])
	case "serve":
		cmdServe(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}
