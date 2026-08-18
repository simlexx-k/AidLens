.PHONY: up down migrate ingest test lint

up:
	docker compose up --build

down:
	docker compose down

migrate:
	docker compose run --rm api alembic upgrade head

ingest:
	docker compose run --rm api python -m app.cli ingest --pages 1

test:
	cd apps/api && pytest

lint:
	cd apps/api && ruff check app tests
