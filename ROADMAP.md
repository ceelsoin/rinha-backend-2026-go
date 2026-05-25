# ROADMAP — Rinha de Backend 2026 · Go / fasthttp backend

## Architecture overview

```
Client → nginx:9999 (round-robin LB) → api1:8080 / api2:8080
```

Each API instance is a single Go binary (`src/main.go`) that:
1. At **build time** (Docker): loads `references.json.gz`, trains a k-means IVF index, writes `references.bin` (~87 MB).
2. At **runtime**: reads `references.bin` into memory, serves HTTP with `fasthttp`.

---

## Key design decisions

### Vector index — IVF (Inverted File Index)

| Parameter | Value | Rationale |
|---|---|---|
| Clusters K | 2048 | ~1465 vectors/cluster; balances search speed vs. recall |
| Training sample | 50 000 vectors | Matches C++ reference; fast k-means++ |
| Training iterations | 50 | Sufficient convergence on this dataset |
| NPROBE (fast) | 4 | Scans ~5 860 vectors; good for 0/5 or 5/5 results |
| NPROBE (repair) | 8 | Extra 4 clusters scanned for 1–4/5 uncertain results |
| Neighbours (k-NN) | 5 | Fixed by the competition spec |

**Quantization**: all float vectors (reference + query) are stored as `int16` (×10000), exactly matching the C++ convention. Squared Euclidean distance is computed in `int64` (max value 5.6 B — safe).

### JSON parsing — `fastjson`

`fastjson.ParserPool` provides zero-allocation request parsing. The `fastjson.Value.GetStringBytes()` return value points into the request buffer (no copy).

### HTTP — `fasthttp`

`fasthttp` reuses buffers per goroutine, avoiding GC pressure on the hot path. Pre-built `[]byte` responses for all 6 possible fraud counts eliminate response serialization at runtime.

### Time parsing

ISO 8601 timestamps are parsed with hand-written byte arithmetic (no `time.Parse`, no allocation):
- **Hour**: direct digit extraction.
- **Weekday**: Tomohiko Sakamoto's algorithm (0 = Monday, 6 = Sunday).
- **Epoch seconds**: Howard Hinnant's civil-to-days algorithm for `minutes_since_last_tx`.

### Load balancer — nginx

nginx is used for simplicity and efficiency. Key settings:
- `keepalive 512` on the upstream group — avoids TCP setup per request.
- `proxy_buffering off` — reduces latency for small responses.
- `access_log off` — eliminates I/O overhead.

---

## Resource budget (total ≤ 1 CPU / 350 MB)

| Service | CPU | Memory |
|---|---|---|
| api1 | 0.45 | 155 MB |
| api2 | 0.45 | 155 MB |
| lb (nginx) | 0.10 | 40 MB |
| **Total** | **1.00** | **350 MB** |

Memory breakdown per API instance:
- IVF vectors (int16): 3 M × 14 × 2 B = **84 MB**
- Labels (uint8): 3 M × 1 B = **3 MB**
- Centroids + metadata: < 1 MB
- Go runtime + stacks: ~10 MB
- fasthttp connection buffers: ~5 MB
- **Total ≈ 103 MB** (headroom inside 155 MB limit)

---

## Build pipeline

```
references.json.gz  ──▶  [Go converter]  ──▶  references.bin (~87 MB)
```

Docker multi-stage build (3 stages):
1. **builder** — compiles `rinha` binary (CGO disabled, stripped).
2. **indexer** — runs `./rinha build ...` to produce `references.bin`.
3. **runtime** — copies binary + `references.bin` + JSON configs into Alpine image.

Estimated Docker build time: **4–8 minutes** (dominated by k-means training on the full 3 M-vector dataset).

---

## Scoring targets

| Metric | Target | Points |
|---|---|---|
| p99 latency | < 10 ms | ~2000 |
| Detection accuracy | < 1% failure rate | ~2500 |
| **Total** | | **~4500** |

---

## To-do / optimization ideas (future iterations)

- [ ] **SIMD distance via CGo**: call into a small C shim using AVX2 `_mm256_madd_epi16` for the cluster scan inner loop — potential 6–8× speedup vs. scalar Go.
- [ ] **Assembly routine**: write a Go assembly (plan9 syntax) `sqDist14` using SSE4 or AVX2.
- [ ] **Worker pool for KNN**: pre-spawn N goroutines locked to OS threads; route requests via channel to avoid scheduler overhead. (GOMAXPROCS=1 per instance already helps.)
- [ ] **Parallel cluster scan**: split the 4–8 probe clusters across 2 goroutines and merge top-5 results.
- [ ] **Ball-tree fallback**: replace IVF with a hierarchical ball-tree for better recall on borderline cases.
- [ ] **Custom LB** (like C++ impl): Unix-socket fd-passing load balancer eliminates the nginx proxy hop, saving ~100 µs per request.
- [ ] **HTTP/2 or raw TCP**: for ultra-low latency, consider a raw TCP server similar to the C++ epoll approach. Not possible with `fasthttp` directly but achievable with `net.Listener` + custom protocol.
- [ ] **Sort cluster vectors by distance to centroid**: vectors close to the centroid are most likely to be true neighbours; scanning them first allows early termination.
- [ ] **GOMAXPROCS tuning**: experiment with `GOMAXPROCS=2` on 0.45 CPU containers to allow parallel scan when two requests arrive simultaneously.

---

## Local testing

```bash
# Build the index locally (requires references.json.gz in place)
cd src
go build -o ../rinha .
../rinha build \
  ../rinha-de-backend-2026-main/resources/references.json.gz \
  /tmp/references.bin \
  ../rinha-de-backend-2026-main/resources/normalization.json \
  ../rinha-de-backend-2026-main/resources/mcc_risk.json

# Serve locally
../rinha serve /tmp/references.bin \
  ../rinha-de-backend-2026-main/resources/normalization.json \
  ../rinha-de-backend-2026-main/resources/mcc_risk.json \
  -port 9999

# Quick smoke test
curl -s http://localhost:9999/ready
curl -s -X POST http://localhost:9999/fraud-score \
  -H 'Content-Type: application/json' \
  -d @rinha-de-backend-2026-main/resources/example-payloads.json | head -c 200
```

---

## Docker build and run

```bash
# From the workspace root (rinha-backend-2026/)
docker compose build      # ~5–8 min (index build dominates)
docker compose up -d
curl http://localhost:9999/ready

# Run the official k6 test from rinha-de-backend-2026-main/test/
cd rinha-de-backend-2026-main
docker compose --profile smoke up
```

---

## File layout

```
rinha-backend-2026/
├── src/
│   ├── main.go        ← entire Go backend (build + serve, single file)
│   ├── go.mod
│   └── go.sum
├── Dockerfile         ← multi-stage: compile → index → runtime
├── docker-compose.yml ← lb + api1 + api2
├── nginx.conf         ← round-robin proxy with upstream keep-alive
└── ROADMAP.md         ← this file
```

---

## Normalization reference (from DETECTION_RULES.md)

| dim | field | formula |
|---|---|---|
| 0 | `transaction.amount` | `clamp(amount / 10000)` |
| 1 | `transaction.installments` | `clamp(installments / 12)` |
| 2 | `amount_vs_avg` | `clamp((amount / avg_amount) / 10)` |
| 3 | `hour_of_day` | `hour(requested_at) / 23` |
| 4 | `day_of_week` | `weekday(requested_at) / 6` (Mon=0, Sun=6) |
| 5 | `minutes_since_last_tx` | `clamp(minutes / 1440)` or **-1** if null |
| 6 | `km_from_last_tx` | `clamp(km_from_current / 1000)` or **-1** if null |
| 7 | `km_from_home` | `clamp(km_from_home / 1000)` |
| 8 | `tx_count_24h` | `clamp(tx_count_24h / 20)` |
| 9 | `is_online` | 0 or 1 |
| 10 | `card_present` | 0 or 1 |
| 11 | `unknown_merchant` | 1 if not in known_merchants, else 0 |
| 12 | `mcc_risk` | from mcc_risk.json (default 0.5) |
| 13 | `merchant.avg_amount` | `clamp(avg_amount / 10000)` |
