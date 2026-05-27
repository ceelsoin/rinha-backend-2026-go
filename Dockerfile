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

# ─── Stage 4: Minimal runtime image ──────────────────────────────────────────
FROM alpine:3.20

WORKDIR /app
COPY --from=builder2 /app/rinha               ./
COPY --from=builder2 /app/lb                  ./
COPY resources/normalization.json ./resources/
COPY resources/mcc_risk.json      ./resources/

EXPOSE 8080
CMD ["./rinha", "serve", \
     "./resources/normalization.json", \
     "./resources/mcc_risk.json"]
