# Makefile per memory-system
# Usage: make <target>

.PHONY: help install start ingest chat search status clean

# Variabili
COMPOSE = docker compose
CONTAINER_NAME = mcp-rag-system-rag-1
OLLAMA_CONTAINER = mcp-rag-system-ollama-1

help: ## Mostra questa guida
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# === INSTALL & SETUP ===

install: ## Avvia i servizi (Qdrant + Ollama) e pulla i modelli LLM
	@echo "🚀 Avvio servizi..."
	$(COMPOSE) up -d qdrant ollama
	@echo "⏳ Attesa avvio Ollama..."
	@sleep 5
	@echo "📥 Pull modello LLM (qwen3.5:4b) - ~4GB RAM..."
	docker exec -it $(OLLAMA_CONTAINER) ollama pull qwen3.5:4b
	@echo "📥 Pull modello embedding (nomic-embed-text)..."
	docker exec -it $(OLLAMA_CONTAINER) ollama pull nomic-embed-text
	@echo "✅ Installazione completata!"
	@echo "   Eseguire: make ingest   (per indicizzare i documenti)"
	@echo "   Eseguire: make chat    (per avviare la chat)"
	@echo ""
	@echo "💡 Nota: Con 8GB RAM il reranking è disabilitato per risparmiare memoria."

start: ## Avvia i servizi senza pull modelli
	$(COMPOSE) up -d qdrant ollama
	@echo "✅ Servizi avviati"

stop: ## Ferma tutti i servizi
	$(COMPOSE) down

# === RAG OPERATIONS ===

ingest: ## Indicizza i documenti da ./data/
	@echo "📚 Indicizzazione documenti..."
	$(COMPOSE) run --rm rag python main.py ingest

ingest-clean: ## Re-indicizza da zero (cancella indice esistente)
	@echo "🗑️  Re-indicizzazione da zero..."
	$(COMPOSE) run --rm rag python main.py ingest --clean

chat: ## Avvia chat interattiva
	$(COMPOSE) run --rm -it rag python main.py chat

search: ## Ricerca nei documenti (usage: make search QUERY="tua query")
	@if [ -z "$(QUERY)" ]; then echo "Usage: make search QUERY=\"tua query\""; exit 1; fi
	$(COMPOSE) run --rm rag python main.py search "$(QUERY)"

status: ## Mostra statistiche sistema
	$(COMPOSE) run --rm rag python main.py status

sources: ## Lista documenti indicizzati
	$(COMPOSE) run --rm rag python main.py sources

# === UTILITY ===

clean: ## Rimuove tutti i container, volumi e cache
	$(COMPOSE) down -v --remove-orphans
	rm -rf ./output/*
	@echo "🧹 Sistema pulito"

logs: ## Mostra i log dei servizi (usage: make logs SERVICE=qdrant|ollama|rag)
	$(COMPOSE) logs -f $(SERVICE)

shell: ## Apri una shell nel container RAG
	$(COMPOSE) run --rm -it rag /bin/bash
