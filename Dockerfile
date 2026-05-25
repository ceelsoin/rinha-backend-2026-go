# ─── Stage 1: Compile the Go binary ─────────────────────────────────────────
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY src/go.mod src/go.sum ./
RUN go mod download

COPY src/ ./
RUN CGO_ENABLED=0 \
    go build -ldflags="-s -w" -o rinha .

# ─── Stage 2: Build the IVF index from references.json.gz ────────────────────
FROM golang:1.22-alpine AS indexer

WORKDIR /app
COPY --from=builder /app/rinha .

# Copy the reference dataset and config files
COPY resources/references.json.gz ./resources/
COPY resources/normalization.json  ./resources/
COPY resources/mcc_risk.json       ./resources/

RUN ./rinha build \
        ./resources/references.json.gz \
        ./resources/references.bin \
        ./resources/normalization.json \
        ./resources/mcc_risk.json

# ─── Stage 3: Minimal runtime image ──────────────────────────────────────────
FROM alpine:3.20

WORKDIR /app
COPY --from=builder  /app/rinha               ./
COPY --from=indexer  /app/resources/references.bin    ./resources/
COPY resources/normalization.json ./resources/
COPY resources/mcc_risk.json      ./resources/

EXPOSE 8080
CMD ["./rinha", "serve", \
     "./resources/references.bin", \
     "./resources/normalization.json", \
     "./resources/mcc_risk.json"]
