# RAG System

Sistema RAG (Retrieval-Augmented Generation) production-ready con **embedding locale** e **LLM cloud**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

- **🧠 Jina Embeddings v3**: Modello multilingue state-of-the-art (1024 dimensioni)
- **🔄 Ingestione Incrementale**: Aggiunge, aggiorna ed elimina automaticamente i documenti
- **💬 Chat RAG**: Risposte contestuali tramite Google Gemini
- **🔌 MCP Server**: Compatibile con Antigravity, Claude Desktop, VS Code, Cursor
- **⚙️ Configurazione Centralizzata**: Un solo file `config.yaml` per tutti i parametri
- **🚀 Zero Rate Limits**: Nessun limite API per l'indicizzazione

## 📄 Formati Supportati

| Categoria     | Estensioni                                                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Testo**     | `.txt`, `.md`, `.csv`, `.log`                                                                                                                      |
| **Codice**    | `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.yml`, `.xml`, `.html`, `.css`, `.java`, `.cpp`, `.c`, `.cs`, `.go`, `.rb`, `.php`, `.sh`, `.bash`, `.sql` |
| **Documenti** | `.pdf`, `.xlsx`, `.ipynb`                                                                                                                          |
| **Audio**     | `.mp3`, `.m4a`, `.wav`, `.ogg`, `.flac` _(richiede FFmpeg)_                                                                                        |

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

> **Nota**: Il primo avvio scarica il modello Jina Embeddings v3. Una tantum.

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

Funziona in modalità **stdio** (per Antigravity, Claude Desktop, Cursor, VS Code).

**Configurazione Antigravity** (`~/.gemini/antigravity/mcp_config.json`):

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "c:/path/to/file-search/.venv/Scripts/python.exe",
      "args": ["c:/path/to/file-search/mcp_server.py"]
    }
  }
}
```

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

Tutte le impostazioni sono centralizzate in `config.yaml`:

```yaml
# ===== PERCORSI =====
documents_path: "./data"
vectorstore_path: "./qdrant_storage"

# ===== EMBEDDING MODEL =====
embedding_model: "jinaai/jina-embeddings-v3"
embedding_dimension: 1024
embedding_task_passage: "retrieval.passage"
embedding_task_query: "retrieval.query"
trust_remote_code: true

# ===== CHUNKING =====
chunk_size: 1024
chunk_overlap: 200
min_text_length: 50

# ===== QDRANT =====
qdrant_collection: "documents"

# ===== RICERCA =====
top_k: 10
similarity_threshold: 0.7

# ===== LLM (per chat.py) =====
model: "gemini-3-pro-preview"
temperature: 0.2
max_tokens: 2048

# ===== SERVER MCP =====
mcp_server_name: "rag-search"

# ===== AUDIO (WHISPER) =====
whisper_model: "base" # small, medium, large

# ===== LOGGING =====
log_level: "INFO"
log_file: "rag_system.log"

# ===== FILE EXTENSIONS =====
extensions:
  text: [".txt", ".md", ".py", ...]
  pdf: [".pdf"]
  audio: [".mp3", ".m4a", ".wav", ".ogg", ".flac"]
  notebook: [".ipynb"]
  excel: [".xlsx"]
```

## 📁 Struttura Progetto

```text
file-search/
├── data/                 # Documenti da indicizzare
├── qdrant_storage/       # Database vettoriale (auto-generato)
├── ingest.py             # Script ingestione documenti
├── chat.py               # Chatbot interattivo con RAG
├── mcp_server.py         # Server MCP per IDE/assistenti
├── extractors.py         # Estrattori testo (PDF, audio, Excel, notebook)
├── utils.py              # Funzioni utility condivise
├── config.yaml           # Configurazione centralizzata
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
| Errori audio transcription   | Installa FFmpeg: `winget install ffmpeg`     |

## 📄 License

MIT
