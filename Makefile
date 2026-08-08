.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help setup qdrant qdrant-down doctor doctor-live dry-run ingest reingest wipe api ui evals clean-index clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "\nDone. Now: cp .env.example .env  and fill in your keys."

qdrant:  ## Start local Qdrant (Docker)
	docker compose up -d
	@echo "Dashboard: http://localhost:6333/dashboard"

qdrant-down:  ## Stop local Qdrant (keeps data)
	docker compose down

doctor:  ## Preflight check — no API calls
	$(PY) -m scripts.doctor

doctor-live:  ## Preflight check + probe the embedding and LLM APIs
	$(PY) -m scripts.doctor --live

dry-run:  ## Parse and chunk DATA/ without embedding or indexing
	$(PY) -m app.ingestion.processor --dry-run

ingest:  ## Ingest DATA/ (skips unchanged files)
	$(PY) -m app.ingestion.processor

reingest:  ## Re-ingest every file, keeping the collection
	$(PY) -m app.ingestion.processor --force

wipe:  ## Drop the collection and rebuild the index from scratch
	$(PY) -m app.ingestion.processor --wipe

api:  ## Run the FastAPI backend on :8000
	$(VENV)/bin/uvicorn app.main:app --reload --port 8000

ui:  ## Run the Streamlit chat UI on :8501
	$(VENV)/bin/streamlit run ui/app.py

evals:  ## Run the evaluation dashboard on :8502
	$(VENV)/bin/streamlit run evals/app.py --server.port 8502

clean-index:  ## Delete local ingestion artefacts (manifest, parsed JSON, embedding cache)
	rm -rf processed_data ingestion_manifest.json .cache
	@echo "Local artefacts removed. Qdrant is untouched — use 'make wipe' for that."

clean: clean-index  ## clean-index plus Python caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
