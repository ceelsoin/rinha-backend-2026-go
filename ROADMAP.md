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

---

## Etapas de implementação (priorizadas por impacto)

### ✅ Etapa 1 — HIVF + adaptive probing + k=7 **[IMPLEMENTADO]**
**Objetivo:** reduzir records escaneados por query, melhorar recall e eliminar o repair scan caro.

| Mudança | Detalhe |
|---|---|
| Índice 2-níveis (HIVF) | 256 L1 super-centroids × 256 L2 sub-centroids = 65.536 clusters; ~45 records/cluster |
| nProbeL1=16, nProbeL2=256 | Escaneia apenas ~11.500 records vs ~17.500 antes |
| Adaptive probing | Se score ∈ [0.38, 0.50] estende para 512 L2 clusters em vez de scan total do índice |
| k=7 vizinhos (era 5) | Melhor precisão com custo linear baixo |
| Threshold 0.44 (era 0.60) | Equivalente a maioria estrita, calibrado para k=7 |
| Binary format novo | Novo magic `GOHIVF01`, layout: l1Cent + l2Cent + offsets + vecs + labels |

### ✅ Etapa 2 — Feature encoding + pesos por dimensão **[IMPLEMENTADO]**
**Objetivo:** melhorar qualidade do vetor de query para maior separabilidade.

| Mudança | Detalhe |
|---|---|
| Codificação cíclica hora/dia | `sin(h×2π/24)`, `cos(h×2π/24)` em vez de linear `h/23` |
| Log para amount | `ln(1+x)/ln(1+max)` em vez de `x/max` |
| Pesos por dimensão | Multiplicar cada dim antes de quantizar (build + query) |
| ⚠️ Requer rebuild | Precisa re-indexar para que referências e queries usem o mesmo encoding |

### ✅ Etapa 3 — AVX2 + distância Manhattan **[IMPLEMENTADO]**
**Objetivo:** ~2× speedup no inner loop de distância (o gargalo real de CPU).

| Mudança | Detalhe |
|---|---|
| Upgrade SSE4.1 → AVX2 | Registros 256-bit: 16 int16 por instrução vs 8 |
| Manhattan distance | `|a-b|` via `VPSIGNW/VPABSW` — elimina multiplicação vs L2² |
| Re-clusterizar com Manhattan | O índice precisa ser construído com a mesma métrica |

### ✅ Etapa 4 — Load balancer com FD passing **[IMPLEMENTADO]**
**Objetivo:** eliminar overhead de proxy TCP do nginx (~10-20µs/req de copy extra).

| Mudança | Detalhe |
|---|---|
| LB em Go (binary separado) | `accept()` na porta 9999, passa FD via `SCM_RIGHTS` para api1/api2 |
| Unix Domain Socket | Worker recebe o socket TCP diretamente, sem double-copy de dados |
| CPU budget | LB em Go usa <0.05 CPU, libera 0.05 CPU para cada api |

### ✅ Etapa 5 — mmap para dados do índice **[IMPLEMENTADO]**
**Objetivo:** compartilhar page cache entre api1 e api2 (economiza ~85 MB de RAM).

| Mudança | Detalhe |
|---|---|
| `golang.org/x/sys/unix.Mmap` | Mapeia index.bin direto no espaço de endereços |
| Page cache compartilhado | OS serve as mesmas páginas físicas para ambas as instâncias |
| Lazy fault | Sem preWarm forçado; OS faz page-in on demand |

---

## Análise do top1-new2 (Zig/ERF88) — práticas que ele usa e nós não

Resultado obtido pelo top1-new2: **p99 ≈ 1.05 ms · FP: 0 · FN: 0 · score: 5977**.

| Prática | Impacto | Status |
|---|---|---|
| Decision Tree compilada no binário | ⚡⚡⚡ crítico | ❌ pendente |
| Fast-paths explícitos pré-classificador | ⚡⚡ alto | ❌ pendente |
| UPSTREAMS × 4 no docker-compose (4 ctrl sockets por API) | ⚡⚡ alto | ❌ pendente |
| Fallback ratio-only se parser falhar | ⚡ médio | ❌ pendente |
| Respostas HTTP pre-buildadas em compile-time | ⚡ médio | parcial |

**Insight principal:** o Zig não carrega nenhum arquivo de referências em produção — o docker-compose não monta referências. O classificador é uma Decision Tree (~500 nós) treinada offline e compilada diretamente no binário como código Zig. Custo por request: ~10–20 comparações float. O HIVF k-NN só existe como fallback de desenvolvimento.

---

### Etapa 6 — Decision Tree compilada no binário ❌ **[PENDENTE — máximo impacto]**

**Objetivo:** substituir ou preceder o HIVF k-NN por uma árvore de decisão pré-treinada compilada em Go — eliminando todo I/O de memória no hot path.

**Como o Zig faz:**
- Treina a árvore offline (sklearn/xgboost) sobre os dados de referência vectorizados.
- Serializa como código Zig estático (`tree_model.zig` com ~500 nós hardcoded).
- `predict(features)` = loop de `while` sobre array de nós; 1 `if` por nó; sem heap, sem disco.

**Como implementar em Go:**
1. Treinar `DecisionTreeClassifier` ou `GradientBoostingClassifier` sobre os 14 features dos dados de referência (script Python no stage de build).
2. Gerar `src/tree_model.go` com a árvore serializada como `[]treeNode` (feature, threshold, left, right, leaf, fraud).
3. `func treePredict(f *[14]float32) bool` — loop O(depth) sem alocação.
4. Usar a árvore como **primeiro classificador**: se resultado for confiante (profundidade ≤ X ou folha com pureza ≥ 0.98), retornar imediatamente sem entrar no HIVF.
5. Manter HIVF como fallback para casos ambíguos.

**Resultado esperado:** requests triviais (80%+ do volume) resolvidos em <5 µs em vez de ~200 µs do HIVF.

---

### Etapa 7 — Fast-paths explícitos pré-classificador ❌ **[PENDENTE — baixo esforço]**

**Objetivo:** capturar perfis obviamente seguros/fraudulentos em ~5 comparações antes de entrar na árvore ou no HIVF.

```go
// Aprovação imediata — perfil claramente legítimo
if amount <= 500 && amountRatio <= 0.5 && installments <= 3 &&
   txCount24h <= 5 && merchantKnown && kmFromHome <= 50 && isSafeMCC(mcc) {
    return approvedResponses[0], nil
}

// Negação imediata — perfil claramente fraudulento
if amount >= 5000 && installments >= 5 && txCount24h >= 6 &&
   !merchantKnown && kmFromHome >= 150 && isRiskyMCC(mcc) {
    return deniedResponses[100], nil
}
```

Thresholds a calibrar contra o dataset de referência. Implementar **antes** de construir o vetor de features.

---

### Etapa 8 — UPSTREAMS × 4 no docker-compose ❌ **[PENDENTE — mínimo esforço]**

**Objetivo:** eliminar contention no socket de controle Unix quando múltiplas goroutines do LB enviam FDs simultaneamente.

O Zig lista cada backend 4 vezes na env `UPSTREAMS`:
```yaml
UPSTREAMS: "/sockets/api1.sock,/sockets/api2.sock,
            /sockets/api1.sock,/sockets/api2.sock,
            /sockets/api1.sock,/sockets/api2.sock,
            /sockets/api1.sock,/sockets/api2.sock"
```

O LB mantém 4 conexões de controle por API e distribui via round-robin sobre 8 slots. Isso permite 4 `sendmsg()` paralelos por API sem serialização.

**Mudança no Go:** o LB (`src/cmd/lb/main.go`) precisa aceitar múltiplas entradas repetidas para o mesmo socket e tratar cada entrada como um canal independente.

---

### Etapa 9 — Fallback ratio-only ❌ **[PENDENTE — baixo esforço]**

**Objetivo:** se o parser completo falhar, decidir com apenas 2 campos extraídos do JSON em vez de retornar erro ou executar o HIVF.

```go
// Extrai só amount e customer.avg_amount do body cru
func scoreRatioFallback(body []byte) (float32, bool) {
    amount := extractNestedFloat(body, `"transaction"`, `"amount"`)
    avg := extractNestedFloat(body, `"customer"`, `"avg_amount"`)
    if avg <= 0 { avg = 1 }
    ratio := clamp32(amount / avg / normConfig.InvAmountVsAvgRatio)
    // ratio > threshold → provável fraude
    ...
}
```

Garante resposta em 100% dos requests mesmo com JSON malformado.

---

## To-do / optimization ideas (future iterations)

- [x] **SIMD AVX2**: `simd_amd64.s` com Manhattan distance (Etapa 3 ✅)
- [x] **FD-passing LB**: substituiu nginx por LB Go com SCM_RIGHTS (Etapa 4 ✅)
- [x] **mmap index**: compartilhamento de page cache entre api1/api2 (Etapa 5 ✅)
- [ ] **Decision Tree compilada** (Etapa 6) — maior impacto em latência
- [ ] **Fast-paths explícitos** (Etapa 7) — baixo esforço, alto ganho
- [ ] **UPSTREAMS × 4** (Etapa 8) — 1 linha de mudança no docker-compose
- [ ] **Fallback ratio-only** (Etapa 9) — cobertura de 100% dos requests
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
