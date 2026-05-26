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
	"net"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"syscall"
	"unsafe"

	"github.com/valyala/fasthttp"
	"github.com/valyala/fastjson"
)

// ── Constants ────────────────────────────────────────────────────────────────

const (
	dims         = 14   // feature dimensions
	stride       = 16   // storage stride per vector: padded to 16 int16 for SIMD alignment
	l1K          = 256  // HIVF level-1 cluster count
	l2KPerL1     = 256  // HIVF level-2 clusters per L1 cluster
	l2KTotal     = l1K * l2KPerL1 // 65,536 total L2 clusters
	nProbeL1     = 16   // top L1 clusters to probe per query
	nProbeL2     = 256  // top L2 clusters to probe (initial pass)
	nProbeL2Ext  = 512  // extended probe when score is uncertain
	nNeigh       = 7    // k-NN neighbors (was 5)
	fraudThresh  = float32(0.44) // score >= this → not approved
	confLow      = float32(0.38) // adaptive probe lower bound
	confHigh     = float32(0.50) // adaptive probe upper bound
	nProbeRepair = 48   // centHeap backing-array size (L1 uses n=nProbeL1≤48)
	trainSample  = 50000
	trainItersL1 = 30
	trainItersL2 = 20
)

var hivfMagic = [8]byte{'G', 'O', 'H', 'I', 'V', 'F', '0', '2'} // HIVF v2: +feature weights

// featureWeights are per-dimension importance multipliers applied to both reference
// vectors at build time and query vectors at serve time. Sourced from top1-new Rust impl.
// Pre-multiplying reference vectors at build time avoids per-distance-call multiplications.
var featureWeights = [dims]float32{
	1.0038165, 0.665417, 0.8668326, 0.5379362,
	0.5, 0.3, 0.3701757, 1.0,
	1.2, 1.2648705, 0.81239825, 1.051987,
	0.8247206, 2.0315619,
}

// kernelCoeff scales L2 Euclidean distance for the Gaussian kernel exp(-L2 * kernelCoeff).
// Calibrated to match the Rust implementation's Manhattan kernel exp(-manhattan * 0.5)
// via manhattan ≈ sqrt(dims) * L2_euclidean → coeff = 0.5 * sqrt(14) ≈ 1.87.
const kernelCoeff = float32(1.87)

// Pre-built HTTP responses: approvedResponses[i] and deniedResponses[i] for
// fraud_score = i/100 (i = 0..100). Selected by bucket = int(score*100 + 0.5).
var approvedResponses [101][]byte
var deniedResponses [101][]byte

func init() {
	for i := 0; i <= 100; i++ {
		score := float64(i) / 100.0
		approvedResponses[i] = []byte(fmt.Sprintf(`{"approved":true,"fraud_score":%.2f}`, score))
		deniedResponses[i] = []byte(fmt.Sprintf(`{"approved":false,"fraud_score":%.2f}`, score))
	}
}

// ── IVF Index ────────────────────────────────────────────────────────────────

// HIVFIndex holds the 2-level hierarchical IVF index.
// All int16 values are quantized: float × 10000.
type HIVFIndex struct {
	n       int      // total reference vectors
	l1Cent  []int16  // l1K × stride L1 super-centroids
	l2Cent  []int16  // l2KTotal × stride L2 sub-centroids
	offsets []uint32 // l2KTotal+1: offsets[i] = start in vecs for L2 cluster i
	vecs    []int16  // n × stride, sorted by L2 cluster assignment
	labels  []uint8  // n: 1=fraud, 0=legit
}

// distIdx pairs a distance with a cluster index for partial-sort helpers.
type distIdx struct {
	d int32
	i int32
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

// quantizeF64 converts a float64 to int16 (×10000), clamped to int16 range.
// Accepts values outside [-1,1] to accommodate feature-weighted inputs (max weight ≈ 2.03).
func quantizeF64(v float64) int16 {
	r := v * 10000.0
	if r > 32767 {
		return 32767
	}
	if r < -32767 {
		return -32767
	}
	if r >= 0 {
		return int16(r + 0.5)
	}
	return int16(r - 0.5)
}

// quantizeF32 converts a float32 to int16 (×10000), clamped to int16 range.
// Accepts values outside [-1,1] to accommodate feature-weighted inputs (max weight ≈ 2.03).
func quantizeF32(v float32) int16 {
	r := float64(v) * 10000.0
	if r > 32767 {
		return 32767
	}
	if r < -32767 {
		return -32767
	}
	return int16(math.Round(r))
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
// Capacity is nProbe so the backing arrays live on the stack.
type centHeap struct {
	d    [nProbeRepair]int32
	idx  [nProbeRepair]int32
	size int
	n    int // capacity: nProbe
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

// neighHeap is a fixed-capacity max-heap for the nNeigh=7 nearest neighbors.
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

// distWeightedScore computes a Gaussian kernel-weighted fraud score from the k neighbors.
// weight_i = exp(-L2_euclidean_i * kernelCoeff), score = sum(w_i * label_i) / sum(w_i).
// This gives closer neighbors more influence than distant ones, matching the Rust top1-new impl.
func (h *neighHeap) distWeightedScore() float32 {
	var fraudW, totalW float32
	for i := 0; i < h.size; i++ {
		// Convert int32 squared-L2 (int16 ×10000 units) to float L2 euclidean in [0, ~sqrt(14)].
		distL2 := float32(math.Sqrt(float64(h.d[i]))) / 10000.0
		w := float32(math.Exp(float64(-distL2 * kernelCoeff)))
		totalW += w
		fraudW += w * float32(h.lbl[i])
	}
	if totalW <= 0 {
		return 0
	}
	return fraudW / totalW
}

// ── distIdx partial-sort helpers ─────────────────────────────────────────────

// distIdxMaxHeapDown sifts element at position i down in a max-heap of []distIdx.
func distIdxMaxHeapDown(s []distIdx, i, n int) {
	for {
		l, r, lg := 2*i+1, 2*i+2, i
		if l < n && s[l].d > s[lg].d {
			lg = l
		}
		if r < n && s[r].d > s[lg].d {
			lg = r
		}
		if lg == i {
			break
		}
		s[i], s[lg] = s[lg], s[i]
		i = lg
	}
}

// partialSortDistIdx rearranges s so that s[:k] contains the k nearest elements
// (smallest .d) in sorted ascending order. Uses max-heap selection — O(N log k).
func partialSortDistIdx(s []distIdx, k int) {
	if k <= 0 {
		return
	}
	if k >= len(s) {
		k = len(s)
	}
	// Build max-heap over first k elements.
	for i := k/2 - 1; i >= 0; i-- {
		distIdxMaxHeapDown(s, i, k)
	}
	// For each remaining element: if closer, replace heap root and re-heapify.
	for i := k; i < len(s); i++ {
		if s[i].d < s[0].d {
			s[0] = s[i]
			distIdxMaxHeapDown(s, 0, k)
		}
	}
	// Heap-sort the k elements into ascending order.
	for i := k - 1; i > 0; i-- {
		s[0], s[i] = s[i], s[0]
		distIdxMaxHeapDown(s, 0, i)
	}
}

// ── HIVF search ───────────────────────────────────────────────────────────────

// scoreRequest performs the 3-phase HIVF nearest-neighbour search.
// Phase 1: scan nProbeL1 nearest L1 super-centroids.
// Phase 2: scan nProbeL2 nearest L2 sub-centroids from those L1 clusters.
// Phase 3: exact scan of records in top nProbeL2 L2 clusters.
// Adaptive: if score is uncertain, extend to nProbeL2Ext L2 clusters.
// Returns a Gaussian kernel-weighted fraud score in [0, 1].
func (idx *HIVFIndex) scoreRequest(q *[stride]int16) float32 {
	// ── Phase 1: scan 256 L1 centroids → top nProbeL1 ─────────────────────
	var ch centHeap
	ch.n = nProbeL1
	for ci := 0; ci < l1K; ci++ {
		c := (*[stride]int16)(unsafe.Pointer(&idx.l1Cent[ci*stride]))
		ch.insert(sqDist16(q, c), int32(ci))
	}

	// ── Phase 2: scan L2 centroids for top nProbeL1 L1 clusters ───────────
	// l2All holds all nProbeL1×l2KPerL1 = 4096 (L2 distance, L2 global index) pairs.
	// Allocated on heap to avoid large stack frames; pooled to avoid per-request GC pressure.
	l2Buf := l2BufPool.Get().(*[nProbeL1 * l2KPerL1]distIdx)
	defer l2BufPool.Put(l2Buf)

	l2Count := 0
	for pi := 0; pi < ch.size; pi++ {
		l1i := int(ch.idx[pi])
		base := l1i * l2KPerL1
		for j := 0; j < l2KPerL1; j++ {
			c := (*[stride]int16)(unsafe.Pointer(&idx.l2Cent[(base+j)*stride]))
			l2Buf[l2Count] = distIdx{sqDist16(q, c), int32(base + j)}
			l2Count++
		}
	}

	// Partial sort: l2Buf[:nProbeL2Ext] = nProbeL2Ext nearest L2 clusters, ascending.
	partialSortDistIdx(l2Buf[:l2Count], nProbeL2Ext)
	if l2Count > nProbeL2Ext {
		l2Count = nProbeL2Ext
	}

	// ── Phase 3: exact scan of records in top nProbeL2 clusters ───────────
	var nh neighHeap
	probeEnd := nProbeL2
	if probeEnd > l2Count {
		probeEnd = l2Count
	}
	for pi := 0; pi < probeEnd; pi++ {
		l2i := int(l2Buf[pi].i)
		start := int(idx.offsets[l2i])
		end := int(idx.offsets[l2i+1])
		maxD := nh.maxD()
		for vi := start; vi < end; vi++ {
			v := (*[stride]int16)(unsafe.Pointer(&idx.vecs[vi*stride]))
			if d := sqDist16(q, v); d < maxD {
				nh.insert(d, idx.labels[vi])
				maxD = nh.maxD()
			}
		}
	}

	// ── Adaptive probing: extend scan when score is in uncertain zone ──────
	initScore := float32(nh.fraudCount()) / float32(nNeigh)
	if initScore > confLow && initScore < confHigh {
		for pi := probeEnd; pi < l2Count; pi++ {
			l2i := int(l2Buf[pi].i)
			start := int(idx.offsets[l2i])
			end := int(idx.offsets[l2i+1])
			maxD := nh.maxD()
			for vi := start; vi < end; vi++ {
				v := (*[stride]int16)(unsafe.Pointer(&idx.vecs[vi*stride]))
				if d := sqDist16(q, v); d < maxD {
					nh.insert(d, idx.labels[vi])
					maxD = nh.maxD()
				}
			}
		}
	}

	return nh.distWeightedScore()
}

// l2BufPool reuses the 32 KB L2 distance buffer across requests.
var l2BufPool = sync.Pool{
	New: func() interface{} { return new([nProbeL1 * l2KPerL1]distIdx) },
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

// trainKMeans runs k-means++ initialization followed by k-means training.
// vecs is a flat int16 array of n×dims; allN is the total number of vectors in vecs.
// Returns a slice of k float64 centroids, each of length dims.
func trainKMeans(vecs []int16, allN, k, sample, iters int, rng *rand.Rand) [][]float64 {
	if allN == 0 || k == 0 {
		return nil
	}
	if k > allN {
		k = allN
	}
	if sample > allN {
		sample = allN
	}

	sampleIdx := make([]int, sample)
	for i := range sampleIdx {
		sampleIdx[i] = rng.Intn(allN)
	}

	centroids := make([][]float64, k)
	for i := range centroids {
		centroids[i] = make([]float64, dims)
	}
	first := sampleIdx[rng.Intn(sample)]
	for d := 0; d < dims; d++ {
		centroids[0][d] = float64(vecs[first*dims+d])
	}

	dmin := make([]float64, sample)
	for i := range dmin {
		dmin[i] = math.MaxFloat64
	}
	for c := 1; c < k; c++ {
		prev := centroids[c-1]
		total := 0.0
		for i, si := range sampleIdx {
			d := kmeansDistF64(prev, vecs[si*dims:si*dims+dims])
			if d < dmin[i] {
				dmin[i] = d
			}
			total += dmin[i]
		}
		ri := sampleIdx[rng.Intn(sample)]
		if total > 0 {
			target := rng.Float64() * total
			acc := 0.0
			for i, si := range sampleIdx {
				acc += dmin[i]
				if acc >= target {
					ri = si
					break
				}
			}
		}
		for d := 0; d < dims; d++ {
			centroids[c][d] = float64(vecs[ri*dims+d])
		}
	}

	assignment := make([]int, sample)
	for iter := 0; iter < iters; iter++ {
		changed := 0
		for i, si := range sampleIdx {
			c := nearestCentroidIdx(vecs[si*dims:si*dims+dims], centroids)
			if c != assignment[i] {
				changed++
				assignment[i] = c
			}
		}
		sums := make([]float64, k*dims)
		counts := make([]int, k)
		for i, si := range sampleIdx {
			c := assignment[i]
			counts[c]++
			for d := 0; d < dims; d++ {
				sums[c*dims+d] += float64(vecs[si*dims+d])
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				ri := sampleIdx[rng.Intn(sample)]
				for d := 0; d < dims; d++ {
					centroids[c][d] = float64(vecs[ri*dims+d])
				}
			} else {
				inv := 1.0 / float64(counts[c])
				for d := 0; d < dims; d++ {
					centroids[c][d] = sums[c*dims+d] * inv
				}
			}
		}
		if changed == 0 {
			break
		}
	}
	return centroids
}

// quantizeCentroids converts float64 centroids to int16 (×10000) with stride padding.
// Nil entries in centroids (empty clusters) are left as all-zeros.
func quantizeCentroids(centroids [][]float64, k int) []int16 {
	out := make([]int16, k*stride)
	for c := 0; c < k; c++ {
		if c >= len(centroids) || centroids[c] == nil {
			continue // zeros for missing/empty cluster centroids
		}
		for d := 0; d < dims; d++ {
			v := centroids[c][d]
			if v < -10000 {
				v = -10000
			}
			if v > 10000 {
				v = 10000
			}
			out[c*stride+d] = int16(math.Round(v))
		}
	}
	return out
}

// buildHIVF trains a 2-level hierarchical IVF (HIVF) index.
// Level 1: l1K=256 super-centroids trained on a global sample.
// Level 2: l2KPerL1=256 sub-centroids per L1 cluster, trained on cluster members.
func buildHIVF(allVecs []int16, allLabels []uint8) *HIVFIndex {
	n := len(allVecs) / dims
	log.Printf("[build] HIVF n=%d l1K=%d l2KPerL1=%d", n, l1K, l2KPerL1)

	rng := rand.New(rand.NewSource(0x52696E6861)) //nolint

	// ── Level 1: train l1K super-centroids on a global sample ─────────────
	log.Printf("[build] L1 k-means++ k=%d sample=%d iters=%d", l1K, trainSample, trainItersL1)
	l1Centroids := trainKMeans(allVecs, n, l1K, trainSample, trainItersL1, rng)

	// ── Assign all N vectors to their nearest L1 cluster (parallel) ────────
	log.Printf("[build] L1 assigning %d vectors...", n)
	l1Assign := make([]int, n)
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
				l1Assign[i] = nearestCentroidIdx(allVecs[i*dims:i*dims+dims], l1Centroids)
			}
		}(s, e)
	}
	wg.Wait()

	// Group vector indices by L1 cluster.
	l1Groups := make([][]int, l1K)
	for i := 0; i < l1K; i++ {
		l1Groups[i] = make([]int, 0, n/l1K+64)
	}
	for i, c := range l1Assign {
		l1Groups[c] = append(l1Groups[c], i)
	}

	// ── Level 2: train l2KPerL1 sub-centroids for each L1 cluster (parallel) ─
	log.Printf("[build] L2 training %d sub-clusters per L1 (parallel)...", l2KPerL1)
	l2CentroidsAll := make([][]float64, l2KTotal)
	l2AssignPerGroup := make([][]int, l1K) // L2 local assignment per group member

	var mu sync.Mutex
	type workItem struct{ l1i int }
	work := make(chan workItem, l1K)
	for i := 0; i < l1K; i++ {
		work <- workItem{i}
	}
	close(work)

	var wg2 sync.WaitGroup
	for w := 0; w < nw; w++ {
		wg2.Add(1)
		go func() {
			defer wg2.Done()
			localRng := rand.New(rand.NewSource(rng.Int63()))
			for item := range work {
				l1i := item.l1i
				members := l1Groups[l1i]
				cnt := len(members)
				if cnt == 0 {
					continue
				}
				k2 := l2KPerL1
				if k2 > cnt {
					k2 = cnt
				}
				// Build a flat int16 array for just these members.
				memberVecs := make([]int16, cnt*dims)
				for mi, gi := range members {
					copy(memberVecs[mi*dims:mi*dims+dims], allVecs[gi*dims:gi*dims+dims])
				}
				samp := 2000
				if samp > cnt {
					samp = cnt
				}
				c2 := trainKMeans(memberVecs, cnt, k2, samp, trainItersL2, localRng)
				// Assign members to L2 sub-clusters.
				assign2 := make([]int, cnt)
				for mi := 0; mi < cnt; mi++ {
					assign2[mi] = nearestCentroidIdx(memberVecs[mi*dims:mi*dims+dims], c2)
				}
				base := l1i * l2KPerL1
				mu.Lock()
				for ci, cen := range c2 {
					l2CentroidsAll[base+ci] = cen
				}
				l2AssignPerGroup[l1i] = assign2
				mu.Unlock()
			}
		}()
	}
	wg2.Wait()

	// ── Build final layout: sort vectors by (L1, L2) assignment ───────────
	log.Printf("[build] building final cluster layout...")
	// Compute per-L2-cluster counts.
	l2Counts := make([]uint32, l2KTotal)
	for l1i, members := range l1Groups {
		assign2 := l2AssignPerGroup[l1i]
		base := l1i * l2KPerL1
		for mi := range members {
			if mi < len(assign2) {
				l2Counts[base+assign2[mi]]++
			}
		}
	}
	// Compute offsets (prefix sum) + sentinel.
	l2Offsets := make([]uint32, l2KTotal+1)
	var off uint32
	for ci := 0; ci < l2KTotal; ci++ {
		l2Offsets[ci] = off
		off += l2Counts[ci]
	}
	l2Offsets[l2KTotal] = off

	// Write vectors in L2-cluster order.
	flatVecs := make([]int16, n*stride)
	flatLabels := make([]uint8, n)
	cursor := make([]uint32, l2KTotal)
	copy(cursor, l2Offsets[:l2KTotal])
	for l1i, members := range l1Groups {
		assign2 := l2AssignPerGroup[l1i]
		base := l1i * l2KPerL1
		for mi, gi := range members {
			var l2loc int
			if mi < len(assign2) {
				l2loc = base + assign2[mi]
			} else {
				l2loc = base
			}
			pos := cursor[l2loc]
			cursor[l2loc]++
			copy(flatVecs[int(pos)*stride:int(pos)*stride+dims], allVecs[gi*dims:gi*dims+dims])
			flatLabels[pos] = allLabels[gi]
		}
	}

	l1CentI16 := quantizeCentroids(l1Centroids, l1K)
	l2CentI16 := quantizeCentroids(l2CentroidsAll, l2KTotal)

	log.Printf("[build] HIVF built: n=%d L1=%d L2=%d", n, l1K, l2KTotal)
	return &HIVFIndex{
		n:       n,
		l1Cent:  l1CentI16,
		l2Cent:  l2CentI16,
		offsets: l2Offsets,
		vecs:    flatVecs,
		labels:  flatLabels,
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

func writeIndex(path string, idx *HIVFIndex) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)

	// Header: magic + version + n
	w.Write(hivfMagic[:])
	binary.Write(w, binary.LittleEndian, uint32(1))     // version
	binary.Write(w, binary.LittleEndian, uint32(idx.n)) // total vectors
	// l1K and l2KPerL1 are compile-time constants; no need to store.

	// L1 centroids (l1K × stride × int16)
	w.Write(int16ToBytes(idx.l1Cent))
	// L2 centroids (l2KTotal × stride × int16)
	w.Write(int16ToBytes(idx.l2Cent))
	// Offsets (l2KTotal+1 × uint32)
	w.Write(uint32ToBytes(idx.offsets))
	// Vectors and labels
	w.Write(int16ToBytes(idx.vecs))
	w.Write(idx.labels)

	return w.Flush()
}

func readIndex(path string) (*HIVFIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, err
	}
	if magic != hivfMagic {
		return nil, fmt.Errorf("bad magic (got %q, want %q)", magic, hivfMagic)
	}
	var version, n32 uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &n32); err != nil {
		return nil, fmt.Errorf("read n: %w", err)
	}
	n := int(n32)

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

	l1Cent, err := readI16(l1K * stride)
	if err != nil {
		return nil, fmt.Errorf("read l1Cent: %w", err)
	}
	l2Cent, err := readI16(l2KTotal * stride)
	if err != nil {
		return nil, fmt.Errorf("read l2Cent: %w", err)
	}
	offsets, err := readU32(l2KTotal + 1)
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

	log.Printf("[serve] loaded HIVF index: n=%d L1=%d L2=%d", n, l1K, l2KTotal)
	return &HIVFIndex{
		n:       n,
		l1Cent:  l1Cent,
		l2Cent:  l2Cent,
		offsets: offsets,
		vecs:    vecs,
		labels:  labels,
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
		for i, v := range entry.Vector {
			vecs = append(vecs, quantizeF64(float64(featureWeights[i])*v))
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
	globalIdx     *HIVFIndex
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

	// ── Quantize to int16 (with feature weights baked in) ────────────────────
	var q [stride]int16 // last 2 elements stay 0 (SIMD padding)
	for i, fv := range vec {
		q[i] = quantizeF32(fv * featureWeights[i])
	}

	// ── HIVF search ──────────────────────────────────────────────────────────
	score := globalIdx.scoreRequest(&q)

	// ── Return pre-computed response ─────────────────────────────────────────
	approved := score < fraudThresh
	bucket := int(score*100 + 0.5)
	if bucket > 100 {
		bucket = 100
	}
	ctx.SetContentTypeBytes([]byte("application/json"))
	if approved {
		ctx.SetBody(approvedResponses[bucket])
	} else {
		ctx.SetBody(deniedResponses[bucket])
	}
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
	fs.Parse(args)

	remaining := fs.Args()
	if len(remaining) < 4 {
		fmt.Fprintln(os.Stderr, "Usage: rinha build <references.json[.gz]> <output.bin> <norm.json> <mcc.json>")
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

	// Build HIVF index
	idx := buildHIVF(vecs, labels)

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
func preWarmIndex(idx *HIVFIndex) {
	const pageStride = 2048 // touch one int16 per 4KB page (2048 int16 = 4096 bytes)
	var acc int32
	for i := 0; i < len(idx.l1Cent); i += pageStride {
		acc += int32(idx.l1Cent[i])
	}
	for i := 0; i < len(idx.l2Cent); i += pageStride {
		acc += int32(idx.l2Cent[i])
	}
	for i := 0; i < len(idx.vecs); i += pageStride {
		acc += int32(idx.vecs[i])
	}
	const labelStride = 4096
	for i := 0; i < len(idx.labels); i += labelStride {
		acc += int32(idx.labels[i])
	}
	_ = acc
	log.Printf("[serve] pre-warmed HIVF index: vecs=%d MB l2Cent=%d KB",
		len(idx.vecs)*2>>20, len(idx.l2Cent)*2>>10)
}

// ── FD-passing receiver (Unix DGRAM + SCM_RIGHTS) ────────────────────────────

// bindDGRAMSocket creates and binds a Unix DGRAM socket to path.
// The LB sends client file descriptors here via sendmsg/SCM_RIGHTS.
func bindDGRAMSocket(path string) (int, error) {
	fd, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_DGRAM, 0)
	if err != nil {
		return 0, fmt.Errorf("socket: %w", err)
	}
	// Large receive buffer: absorbs fd bursts during peak load without blocking the LB.
	syscall.SetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_RCVBUF, 16*1024*1024) //nolint
	addr := &syscall.SockaddrUnix{Name: path}
	if err := syscall.Bind(fd, addr); err != nil {
		syscall.Close(fd) //nolint
		return 0, fmt.Errorf("bind: %w", err)
	}
	// Make socket world-writable so the LB (different process/container) can send to it.
	os.Chmod(path, 0777) //nolint
	return fd, nil
}

// recvFDLoop blocks on recvmsg, receiving client fds from the LB.
// For each received fd, wraps it as a net.Conn and calls srv.ServeConn in a goroutine.
func recvFDLoop(udsFd int, srv *fasthttp.Server) {
	const oobSize = 256 // CMSG_SPACE(sizeof(int)) ≈ 24; 256 is ample
	oob := make([]byte, oobSize)
	dummy := make([]byte, 1)

	for {
		_, oobn, _, _, err := syscall.Recvmsg(udsFd, dummy, oob, 0)
		if err != nil {
			continue
		}
		if oobn == 0 {
			continue
		}

		scms, err := syscall.ParseSocketControlMessage(oob[:oobn])
		if err != nil || len(scms) == 0 {
			continue
		}
		fds, err := syscall.ParseUnixRights(&scms[0])
		if err != nil || len(fds) == 0 {
			continue
		}

		// net.FileConn dups the fd internally (sets O_NONBLOCK on the dup).
		// We then close the original; the conn owns the dup'd fd.
		f := os.NewFile(uintptr(fds[0]), "tcp-conn")
		conn, err := net.FileConn(f)
		f.Close()
		if err != nil {
			syscall.Close(fds[0]) //nolint
			continue
		}

		go func() {
			defer conn.Close()
			srv.ServeConn(conn)
		}()
	}
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
		// FD-passing mode: receive client fds from the Go LB via Unix DGRAM + SCM_RIGHTS.
		// The LB accepts TCP connections and passes each fd here via sendmsg/SCM_RIGHTS.
		// This eliminates the proxy double-copy: the API handles the raw TCP fd directly.
		log.Printf("[serve] FD-passing mode on unix-dgram:%s", socketPath)
		os.Remove(socketPath) // clean up leftover from previous run

		udsFd, err := bindDGRAMSocket(socketPath)
		if err != nil {
			log.Fatalf("bind DGRAM socket %s: %v", socketPath, err)
		}
		log.Printf("[serve] ready, receiving fds from LB")
		recvFDLoop(udsFd, srv)
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
