IMAGE      := ceelsoinacio/rinha-backend-2026:v9
LOCAL_IMAGE := rinha-local:latest

build:
	docker build -t $(IMAGE) .

push:
	docker buildx build --platform linux/amd64 --push -t $(IMAGE) .

up:
	docker compose up -d

down:
	docker compose down -v

# ── Local (native arm64 on Mac) ───────────────────────────────────────────────
local-build:
	docker build -t $(LOCAL_IMAGE) .

local-up: local-build
	RINHA_IMAGE=$(LOCAL_IMAGE) docker compose up -d

local-down:
	RINHA_IMAGE=$(LOCAL_IMAGE) docker compose down -v

local-test: local-build
	RINHA_IMAGE=$(LOCAL_IMAGE) docker compose up -d
	@echo "Waiting for /ready (mirroring competition engine: up to 60s)..."
	@for i in $$(seq 1 20); do \
		curl -sf http://localhost:9999/ready > /dev/null 2>&1 && echo "Server ready after $$(($$i * 3))s" && break; \
		echo "  attempt $$i/20 — not ready yet"; \
		sleep 3; \
	done
	docker compose -f test/docker-compose.yml --profile smoke up --abort-on-container-exit
	RINHA_IMAGE=$(LOCAL_IMAGE) docker compose down -v

local-fulltest: local-build
	RINHA_IMAGE=$(LOCAL_IMAGE) docker compose up -d
	@echo "Waiting for /ready (mirroring competition engine: up to 60s)..."
	@for i in $$(seq 1 20); do \
		curl -sf http://localhost:9999/ready > /dev/null 2>&1 && echo "Server ready after $$(($$i * 3))s" && break; \
		echo "  attempt $$i/20 — not ready yet"; \
		sleep 3; \
	done
	docker compose -f test/docker-compose.yml --profile test up --abort-on-container-exit
	RINHA_IMAGE=$(LOCAL_IMAGE) docker compose down -v

# Run full k6 test via Docker (Linux-compatible; on macOS use local-test instead)
test:
	docker compose -f test/docker-compose.yml --profile test up --abort-on-container-exit

result:
	cat test/test/results.json | jq -r '.scoring.raw' 
.PHONY: build push up down local-build local-up local-down local-test local-fulltest test result