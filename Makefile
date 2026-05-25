test:
	bash -c "cd test && docker compose --profile test up --abort-on-container-exit" 2>&1