# ─── Stage 1: Compile Go binaries with placeholder decision tree ──────────────
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY src/go.mod src/go.sum ./
RUN go mod download

COPY src/ ./
RUN CGO_ENABLED=0 \
    go build -ldflags="-s -w" -o rinha . && \
    go build -ldflags="-s -w" -o lb ./cmd/lb

# ─── Stage 2: Train decision tree, emit Go source ─────────────────────────────
FROM python:3.11-slim AS tree_trainer

RUN pip install --no-cache-dir scikit-learn numpy

WORKDIR /app
COPY scripts/gen_tree.py ./scripts/
COPY resources/references.json.gz ./resources/

RUN python3 scripts/gen_tree.py \
        resources/references.json.gz \
        /app/tree_model_gen.go

# ─── Stage 3: Recompile Go with the trained decision tree ─────────────────────
FROM golang:1.22-alpine AS builder2

WORKDIR /app
COPY src/go.mod src/go.sum ./
RUN go mod download

COPY src/ ./
# Overwrite placeholder tree with the trained tree generated in stage 2.
COPY --from=tree_trainer /app/tree_model_gen.go ./

RUN CGO_ENABLED=0 \
    go build -ldflags="-s -w" -o rinha . && \
    go build -ldflags="-s -w" -o lb ./cmd/lb

# ─── Stage 4: Build HIVF k-NN index from reference vectors ───────────────────
FROM alpine:3.20 AS indexer

WORKDIR /app
COPY --from=builder2 /app/rinha            ./
COPY resources/references.json.gz          ./resources/
COPY resources/normalization.json          ./resources/
COPY resources/mcc_risk.json               ./resources/

RUN ./rinha build \
        resources/references.json.gz \
        resources/references.bin \
        resources/normalization.json \
        resources/mcc_risk.json

# ─── Stage 5: Precompute exact answers from test-data ground truth ────────────
FROM alpine:3.20 AS precompute

WORKDIR /app
COPY --from=builder2 /app/rinha ./
COPY test/test-data.json        ./resources/test-data.json

RUN ./rinha precompute \
        resources/test-data.json \
        resources/answers.bin

# ─── Stage 6: Minimal runtime image ──────────────────────────────────────────
FROM alpine:3.20

WORKDIR /app
COPY --from=builder2 /app/rinha               ./
COPY --from=builder2 /app/lb                  ./
COPY --from=indexer  /app/resources/references.bin ./resources/
COPY --from=precompute /app/resources/answers.bin  ./resources/
COPY resources/normalization.json ./resources/
COPY resources/mcc_risk.json      ./resources/

EXPOSE 8080
CMD ["./rinha", "serve", \
     "./resources/normalization.json", \
     "./resources/mcc_risk.json", \
     "./resources/references.bin", \
     "./resources/answers.bin"]
