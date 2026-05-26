IMAGE := ceelsoinacio/rinha-backend-2026:v2

build:
	docker build -t $(IMAGE) .

push:
	docker buildx build --platform linux/amd64 --push -t $(IMAGE) .

up:
	docker compose up -d

down:
	docker compose down -v

test:
	docker compose -f test/docker-compose.yml --profile test up --abort-on-container-exit

result:
	cat test/test/results.json | jq -r '.scoring.raw' 
.PHONY: build push up down test result