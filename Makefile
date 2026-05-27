IMAGE      := ceelsoinacio/rinha-backend-2026:v7
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
	docker compose -f test/docker-compose.yml --profile test up --abort-on-container-exit
	RINHA_IMAGE=$(LOCAL_IMAGE) docker compose down -v

test:
	docker compose -f test/docker-compose.yml --profile test up --abort-on-container-exit

result:
	cat test/test/results.json | jq -r '.scoring.raw' 
.PHONY: build push up down local-build local-up local-down local-test test result