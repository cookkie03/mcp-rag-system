# RAG System

Sistema RAG (Retrieval-Augmented Generation) production-ready con **embedding locale** e **LLM cloud**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

- **📦 Embedding Locale**: Usa `all-mpnet-base-v2` (state-of-the-art) senza API esterne
- **🔄 Ingestione Incrementale**: Aggiunge, aggiorna ed elimina automaticamente i documenti
- **💬 Chat RAG**: Risposte contestuali tramite Google Gemini
- **🔌 MCP Server**: Compatibile con Claude Desktop, VS Code, Cursor e altri client MCP
- **🚀 Zero Rate Limits**: Nessun limite API per l'indicizzazione

## Requisiti

- Python 3.10+
- Google API Key ([ottienila qui](https://aistudio.google.com/apikey)) - Solo per la Chat

## ⚡ Quick Start

```bash
# 1. Clona e entra nella directory
git clone <repo-url>
cd file-search

# 2. Crea e attiva ambiente virtuale
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Configura API key
copy .env.example .env
# Modifica .env con la tua GOOGLE_API_KEY

# 5. Aggiungi documenti e indicizza
# Copia i tuoi file in data/, poi:
python ingest.py
```

> **Nota**: Il primo avvio scarica il modello embedding (~420MB). Una tantum.

## 📖 Utilizzo

### Ingestione Documenti

```bash
python ingest.py           # Incrementale (aggiunge nuovi, aggiorna modificati, rimuove eliminati)
python ingest.py --clean   # Ricostruzione completa da zero
```

### Chatbot Interattivo

```bash
python chat.py
```

Comandi: `/help`, `/stats`, `/exit`

### Server MCP

```bash
python mcp_server.py
```

Funziona in modalità **stdio** (per Claude Desktop, Cursor, VS Code) o come server standalone.

**Configurazione Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "python",
      "args": ["c:/path/to/file-search/mcp_server.py"],
      "cwd": "c:/path/to/file-search"
    }
  }
}
```

## ⚙️ Configurazione

Modifica `config.yaml`:

```yaml
documents_path: "./data" # Cartella documenti sorgente
vectorstore_path: "./qdrant_storage" # Database vettoriale
chunk_size: 1024 # Dimensione chunk
chunk_overlap: 200 # Sovrapposizione
top_k: 10 # Risultati per query
model: "gemini-3-pro-preview" # Modello LLM
temperature: 0.7
max_tokens: 4096
```

## 📁 Struttura Progetto

```text
file-search/
├── data/                 # Documenti da indicizzare
├── qdrant_storage/       # Database vettoriale (auto-generato)
├── ingest.py             # Script ingestione
├── chat.py               # Chatbot interattivo
├── mcp_server.py         # Server MCP
├── utils.py              # Funzioni comuni
├── config.yaml           # Configurazione
├── requirements.txt      # Dipendenze Python
├── .env.example          # Template variabili ambiente
└── README.md
```

## 🔧 Troubleshooting

| Problema                     | Soluzione                                    |
| ---------------------------- | -------------------------------------------- |
| `GOOGLE_API_KEY non trovata` | Crea `.env` con `GOOGLE_API_KEY=...`         |
| `Nessun documento trovato`   | Aggiungi file nella cartella `data/`         |
| `ModuleNotFoundError`        | Esegui `pip install -r requirements.txt`     |
| Primo avvio lento            | Normale: sta scaricando il modello embedding |

## 📄 License

MIT
