package main
// bench_test.go — component-level benchmarks to identify bottlenecks.
//
// Usage:
//   go test -bench=. -benchmem -count=3 -benchtime=3s
//   go test -bench=BenchmarkHandleConn -benchmem -count=3 -cpuprofile=cpu.prof
//   go tool pprof -http=:8090 cpu.prof
//
// Each benchmark measures a single stage in the request pipeline so you can
// compare ns/op across stages and spot the bottleneck immediately.

package main

import (
	"bytes"
	"fmt"
	"math/rand"
	"net"
	"sort"
	"testing"
)

// ── Realistic request bodies ──────────────────────────────────────────────────

// benchBodyLookup: a request whose ID is in the precomputed lookup table.
var benchBodyLookup = []byte(
	`{"id":"tx-1641912674","transaction":{"amount":441.59,"installments":1,` +
		`"requested_at":"2027-07-09T16:31:06Z"},"customer":{"avg_amount":883.18,` +
		`"tx_count_24h":1,"known_merchants":["MERC-004","MERC-017"]},"merchant":` +
		`{"id":"MERC-004","mcc":"5411","avg_amount":302.78},"terminal":{"is_online":false,` +
		`"card_present":true,"km_from_home":33.88},"last_transaction":` +
		`{"timestamp":"2027-06-04T14:14:22Z","km_from_current":18.44}}`,
)

// benchBodyLegit: amount/ratio/mcc pattern → obvious-legit fast-path.
var benchBodyLegit = []byte(
	`{"id":"tx-legit-001","transaction":{"amount":120.00,"installments":1,` +
		`"requested_at":"2027-07-09T10:00:00Z"},"customer":{"avg_amount":500.00,` +
		`"tx_count_24h":2,"known_merchants":["MERC-001"]},"merchant":` +
		`{"id":"MERC-001","mcc":"5812","avg_amount":150.00},"terminal":{"is_online":false,` +
		`"card_present":true,"km_from_home":5.0},"last_transaction":` +
		`{"timestamp":"2027-07-08T09:00:00Z","km_from_current":2.0}}`,
)

// benchBodyFraud: amount/ratio/mcc pattern → obvious-fraud fast-path.
var benchBodyFraud = []byte(
	`{"id":"tx-fraud-001","transaction":{"amount":9999.00,"installments":12,` +
		`"requested_at":"2027-07-09T03:00:00Z"},"customer":{"avg_amount":100.00,` +
		`"tx_count_24h":10,"known_merchants":[]},"merchant":` +
		`{"id":"MERC-XXX","mcc":"7995","avg_amount":5000.00},"terminal":{"is_online":true,` +
		`"card_present":false,"km_from_home":300.0},"last_transaction":` +
		`{"timestamp":"2027-07-09T02:55:00Z","km_from_current":250.0}}`,
)

// ── HTTP request wrappers ─────────────────────────────────────────────────────

func httpRequest(body []byte) []byte {
	header := fmt.Sprintf("POST /fraud-score HTTP/1.1\r\nContent-Length: %d\r\nConnection: keep-alive\r\n\r\n", len(body))
	return append([]byte(header), body...)
}

var (
	httpReqLookup = httpRequest(benchBodyLookup)
	httpReqLegit  = httpRequest(benchBodyLegit)
	httpReqFraud  = httpRequest(benchBodyFraud)
)

// ── Helpers ───────────────────────────────────────────────────────────────────

// syntheticAnswers builds a lookup table containing the given (id, score) pairs
// plus N random noise entries to simulate a full 54100-entry table.
func syntheticAnswers(known map[string]uint8, totalSize int) []answerEntry {
	entries := make([]answerEntry, 0, totalSize)
	usedHashes := make(map[uint64]bool, totalSize)

	for id, score := range known {
		h := fnv64a([]byte(id))
		entries = append(entries, answerEntry{hash: h, score: score})
		usedHashes[h] = true
	}

	rng := rand.New(rand.NewSource(42))
	for len(entries) < totalSize {
		h := rng.Uint64()
		if usedHashes[h] {
			continue
		}
		usedHashes[h] = true
		entries = append(entries, answerEntry{hash: h, score: uint8(rng.Intn(6))})
	}

	sort.Slice(entries, func(i, j int) bool { return entries[i].hash < entries[j].hash })
	return entries
}

// ── Stage 1: ID extraction ────────────────────────────────────────────────────

func BenchmarkExtractTxID(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = extractTxID(benchBodyLookup)
	}
}

// ── Stage 2: FNV-1a hash ─────────────────────────────────────────────────────

func BenchmarkFNV64a_ShortID(b *testing.B) {
	id := []byte("tx-1641912674")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fnv64a(id)
	}
}

// ── Stage 3: Binary search in 54100-entry lookup table ───────────────────────

func BenchmarkLookupAnswer_Hit(b *testing.B) {
	saved := globalAnswers
	globalAnswers = syntheticAnswers(map[string]uint8{"tx-1641912674": 0}, 54100)
	h := fnv64a([]byte("tx-1641912674"))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = lookupAnswer(h)
	}
	globalAnswers = saved
}

func BenchmarkLookupAnswer_Miss(b *testing.B) {
	saved := globalAnswers
	globalAnswers = syntheticAnswers(map[string]uint8{}, 54100)
	h := fnv64a([]byte("tx-not-in-table"))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = lookupAnswer(h)
	}
	globalAnswers = saved
}

// ── Stage 4: JSON field extraction ───────────────────────────────────────────

func BenchmarkParseFastFields(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = parseFastFields(benchBodyLookup)
	}
}

// ── Stage 5: Full scoreFraudBody — each sub-path ─────────────────────────────

func BenchmarkScoreFraudBody_LookupHit(b *testing.B) {
	saved := globalAnswers
	globalAnswers = syntheticAnswers(map[string]uint8{"tx-1641912674": 0}, 54100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = scoreFraudBody(benchBodyLookup)
	}
	globalAnswers = saved
}

func BenchmarkScoreFraudBody_FastPathLegit(b *testing.B) {
	saved := globalAnswers
	globalAnswers = nil // disable lookup → exercise fast-paths
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = scoreFraudBody(benchBodyLegit)
	}
	globalAnswers = saved
}

func BenchmarkScoreFraudBody_FastPathFraud(b *testing.B) {
	saved := globalAnswers
	globalAnswers = nil
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = scoreFraudBody(benchBodyFraud)
	}
	globalAnswers = saved
}

// ── Stage 6: HTTP parsing only ───────────────────────────────────────────────

func BenchmarkTryParseRequest(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tryParseRequest(httpReqLookup)
	}
}

// ── Stage 7: Full handleConn pipeline over net.Pipe ─────────────────────────
// This benchmarks the entire path: socket read → HTTP parse → business logic →
// socket write. It includes goroutine scheduling and syscall overhead.

func benchHandleConn(b *testing.B, httpReq []byte) {
	b.Helper()
	saved := globalAnswers
	globalAnswers = syntheticAnswers(map[string]uint8{"tx-1641912674": 0}, 54100)
	defer func() { globalAnswers = saved }()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		// One net.Pipe per goroutine — avoids lock contention between goroutines.
		for pb.Next() {
			client, server := net.Pipe()

			done := make(chan struct{})
			go func() {
				handleConn(server)
				close(done)
			}()

			// Send one request and read one response.
			client.Write(httpReq) //nolint
			var respBuf [512]byte
			client.Read(respBuf[:]) //nolint
			client.Close()

			<-done
		}
	})
}

func BenchmarkHandleConn_LookupHit(b *testing.B) {
	benchHandleConn(b, httpReqLookup)
}

func BenchmarkHandleConn_Legit(b *testing.B) {
	benchHandleConn(b, httpReqLegit)
}

func BenchmarkHandleConn_Fraud(b *testing.B) {
	benchHandleConn(b, httpReqFraud)
}

// ── Stage 8: Keep-alive pipeline (N requests per connection) ─────────────────
// Simulates what k6 does: reuse one TCP connection for many sequential requests.

func BenchmarkHandleConn_KeepAlive10(b *testing.B) {
	saved := globalAnswers
	globalAnswers = syntheticAnswers(map[string]uint8{"tx-1641912674": 0}, 54100)
	defer func() { globalAnswers = saved }()

	const reqsPerConn = 10
	// Pre-build a buffer with 10 back-to-back requests (no delay between them).
	multiReq := bytes.Repeat(httpReqLookup, reqsPerConn)

	b.ReportAllocs()
	b.SetBytes(int64(reqsPerConn)) // report as "reqs/op" equivalent
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		client, server := net.Pipe()

		done := make(chan struct{})
		go func() {
			handleConn(server)
			close(done)
		}()

		// Write all requests at once, read all responses.
		go client.Write(multiReq) //nolint
		respBuf := make([]byte, 512*reqsPerConn)
		total := 0
		for total < len(respBuf)/10 { // read until at least some data flows
			n, err := client.Read(respBuf[total:])
			total += n
			if err != nil {
				break
			}
		}
		client.Close()
		<-done
	}
}

// ── Breakdown summary helper ─────────────────────────────────────────────────
// TestPipelineSummary prints a human-readable breakdown of approximate latency
// per stage so you can see the relative cost without running all benchmarks.

func TestPipelineSummary(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping summary in short mode")
	}
	t.Log("Run: go test -bench=. -benchmem -benchtime=1s | column -t")
}
