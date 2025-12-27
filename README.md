# RAG System - Production Ready

Sistema RAG (Retrieval-Augmented Generation) production-ready con embedding locale e vector database Qdrant su Docker.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

- **Jina Embeddings v3**: Modello multilingue state-of-the-art (1024 dimensioni)
- **Qdrant Docker**: Vector database scalabile e performante
- **Ingestione Incrementale**: Rileva automaticamente file nuovi, modificati ed eliminati
- **Chunking Semantico**: Rispetta confini frasi, protegge abbreviazioni scientifiche
- **Reranking Cross-Encoder**: Riordina risultati per rilevanza semantica reale
- **Threshold Adattivo**: Soglia qualità dinamica basata su GAP analysis
- **PDF Strutturato**: Estrae tabelle in Markdown con PyMuPDF4LLM
- **Tracciabilità Fonti**: Posizione caratteri e citazioni precise
- **MCP Server**: Compatibile con Antigravity, Claude Desktop, VS Code, Cursor
- **Production-Ready**: Retry logic, input validation, logging strutturato, metriche

---

## Quick Start

### 1. Setup Docker Qdrant

⚠️ **Assicurati che Docker Desktop sia avviato prima di procedere.**

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v ./qdrant_data:/qdrant/storage \
  --name qdrant-rag \
  qdrant/qdrant:latest
```

**Windows**:

```powershell
docker run -d -p 6333:6333 -p 6334:6334 -v %cd%/qdrant_data:/qdrant/storage --name qdrant-rag qdrant/qdrant:latest
```

Verifica:

```bash
curl http://localhost:6333/health
```

### 2. Setup Python Environment

```bash
# Clone repository
git clone <repo-url>
cd file-search

# Virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Dipendenze
pip install -r requirements.txt
```

### 3. Configurazione

Crea file `.env` con le tue API keys (opzionale, solo per chat.py):

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

Verifica `config.yaml`:

```yaml
# Qdrant Docker
qdrant_mode: "http"
qdrant_host: "localhost"
qdrant_port: 6333

# Quality threshold (0.6-1.0, raccomandato 0.7 per uso scientifico)
similarity_threshold: 0.7
```

### 4. Ingestione Documenti

Copia i tuoi file nella cartella `data/`, poi:

```bash
# Prima indicizzazione (completa)
python ingest.py --clean

# Aggiornamenti incrementali (rileva modifiche)
python ingest.py
```

L'ingestione incrementale:

- ✅ Rileva file **nuovi** e li aggiunge
- ✅ Rileva file **modificati** e li aggiorna
- ✅ Rileva file **eliminati** e rimuove i vettori
- ✅ Salta file già indicizzati e non modificati

### 5. Test Sistema

```bash
# Test rapido
python -c "from mcp_server import search_knowledge_base; print(search_knowledge_base('test query', 3))"
```

---

## Formati Supportati

| Categoria     | Estensioni                                                                                                                        |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Testo**     | `.txt`, `.md`, `.csv`, `.log`                                                                                                     |
| **Codice**    | `.py`, `.js`, `.ts`, `.json`, `.yaml`, `.xml`, `.html`, `.css`, `.java`, `.cpp`, `.c`, `.cs`, `.go`, `.rb`, `.php`, `.sh`, `.sql` |
| **Documenti** | `.pdf`, `.xlsx`, `.ipynb`                                                                                                         |
| **Audio**     | `.mp3`, `.m4a`, `.wav`, `.ogg`, `.flac` (richiede FFmpeg)                                                                         |

---

## MCP Server - Integrazione IDE

Il server MCP espone il sistema RAG a IDE e assistenti AI.

### Tool Disponibili

#### 1. `search_knowledge_base(query, limit)`

Cerca nei documenti indicizzati tramite ricerca semantica vettoriale.

**Parametri**:

- `query` (string): Domanda o ricerca (3-2000 caratteri)
- `limit` (int, optional): Numero risultati (1-50, default: 10)

**Output**:

```
✅ Trovati N risultati rilevanti (soglia qualità: 0.72 (adattivo)):

[Risultato 1/N]
📄 Fonte: papers/document.pdf (chunk 12)
📍 Posizione: caratteri 4520-5480
🎯 Rilevanza: 0.887 🟢 Eccellente
📝 Citazione: "papers/document.pdf", char. 4520-5480

[contenuto del chunk...]
```

#### 2. `get_server_stats()`

Mostra statistiche del server (query totali, success rate, uptime, ecc.)

---

## Setup Antigravity (Google Gemini)

### Configurazione

**File**: `C:\Users\<username>\.gemini\antigravity\mcp_config.json`

**Linux/Mac**: `~/.gemini/antigravity/mcp_config.json`

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "C:\\path\\to\\file-search\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\file-search\\mcp_server.py"],
      "description": "RAG search engine for documents"
    }
  }
}
```

**Linux/Mac**:

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "/path/to/file-search/.venv/bin/python",
      "args": ["/path/to/file-search/mcp_server.py"],
      "description": "RAG search engine for documents"
    }
  }
}
```

### Utilizzo

1. Salva il file di configurazione
2. Riavvia Antigravity completamente
3. In Antigravity scrivi: _"Usa search_knowledge_base per cercare 'machine learning'"_

---

## Setup Claude Desktop / Claude Code

### Configurazione

**File Windows**: `%APPDATA%\Roaming\Claude\claude_desktop_config.json`

**File Linux/Mac**: `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "C:\\path\\to\\file-search\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\file-search\\mcp_server.py"]
    }
  }
}
```

**Linux/Mac**:

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "/path/to/file-search/.venv/bin/python",
      "args": ["/path/to/file-search/mcp_server.py"]
    }
  }
}
```

### Utilizzo

1. Salva il file di configurazione
2. Riavvia Claude Desktop/Code
3. Il tool `search_knowledge_base` sarà disponibile automaticamente

---

## Chatbot Interattivo (Opzionale)

Sistema di chat interattivo con RAG usando Google Gemini:

```bash
python chat.py
```

Comandi disponibili:

- `/help` - Mostra aiuto
- `/stats` - Statistiche database
- `/exit` - Esci

**Nota**: Richiede `GOOGLE_API_KEY` in `.env`

---

## Configurazione Avanzata

### config.yaml

```yaml
# ===== EMBEDDING MODEL =====
embedding_model: "jinaai/jina-embeddings-v3"
embedding_dimension: 1024
embedding_task_passage: "retrieval.passage" # Task per indicizzazione
embedding_task_query: "retrieval.query" # Task per query
trust_remote_code: true

# ===== CHUNKING =====
chunk_size: 1024 # Dimensione chunk in caratteri
chunk_overlap: 200 # Overlap tra chunks consecutivi
chunking_mode: "sentence" # "sentence" (semantico) o "character" (legacy)
min_text_length: 50 # Lunghezza minima testo per indicizzazione

# ===== PDF EXTRACTION =====
pdf_extraction_mode: "markdown" # "markdown" (tabelle/struttura) o "text" (legacy)

# ===== QDRANT =====
qdrant_mode: "http" # "http" (Docker) o "local" (embedded)
qdrant_host: "localhost"
qdrant_port: 6333
qdrant_collection: "documents"

# ===== RICERCA =====
top_k: 10 # Numero default risultati
similarity_threshold: 0.7 # Soglia base (fallback)
adaptive_threshold: true # Abilita threshold adattivo
adaptive_threshold_min: 0.5 # Minimo threshold adattivo
adaptive_threshold_max: 0.9 # Massimo threshold adattivo

# ===== RERANKING =====
rerank_enabled: true # Abilita cross-encoder reranking
rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_top_n: 30 # Candidati per reranking

# ===== LLM (per chat.py) =====
model: "gemini-3-pro-preview"
temperature: 0.2
max_tokens: 2048
```

### Similarity Threshold

Il sistema supporta **threshold adattivo** basato su GAP analysis:

- **Modalità Adattiva** (default): Analizza la distribuzione degli score e trova automaticamente il cutoff ottimale
- **Modalità Statica**: Usa `similarity_threshold` fisso come fallback

**Range consigliati**:

- **0.9-1.0**: Solo match quasi perfetti (molto restrittivo)
- **0.7-0.9**: Match rilevanti (raccomandato per uso scientifico) ✅
- **0.5-0.7**: Match moderatamente rilevanti
- **0.0-0.5**: Anche match poco rilevanti (sconsigliato)

**Default**: `adaptive_threshold: true` con range 0.5-0.9

---

## Accortezze Production-Ready

### 1. Retry Logic & Error Handling

- 3 tentativi automatici di connessione a Qdrant
- Timeout configurabili (30s Qdrant, 60s embedding)
- Logging completo di ogni errore
- Zero silent failures

### 2. Input Validation

- Query: 3-2000 caratteri
- Type checking completo
- Limit: 1-50 risultati
- Protezione DoS

### 3. Quality Threshold & Reranking

- **Threshold adattivo**: GAP analysis per cutoff ottimale automatico
- **Cross-encoder reranking**: Riordina per rilevanza semantica reale
- Score sempre visibile con indicatori qualitativi:
  - 🟢 Eccellente (≥0.9)
  - 🟡 Buona (≥0.8)
  - 🟠 Sufficiente (≥0.7)

### 4. Health Check Startup

- Verifica connessione Qdrant all'avvio
- Validazione collection esistente
- Check documenti indicizzati
- Verifica schema vettori

### 5. Logging & Metriche

- File log: `mcp_server.log`
- Metriche: total queries, success rate, avg time, uptime
- Tool `get_server_stats()` per monitoring real-time

### 6. Connection Pooling

- Modalità HTTP (Docker) per multi-processo
- Singleton client Qdrant
- Timeout espliciti
- Graceful shutdown

---

## Struttura Repository

```
file-search/
├── data/                    # Documenti da indicizzare (crea questa cartella)
├── .venv/                   # Virtual environment Python
├── chat.py                  # Chatbot interattivo (opzionale)
├── config.yaml              # Configurazione centralizzata
├── extractors.py            # Estrattori testo (PDF, audio, Excel, etc.)
├── ingest.py                # Script ingestione documenti
├── mcp_server.py            # Server MCP production-ready
├── utils.py                 # Funzioni utility
├── requirements.txt         # Dipendenze Python
├── .env.example             # Template variabili ambiente
└── README.md                # Questa documentazione
```

**File generati automaticamente**:

- `qdrant_data/` - Dati Qdrant Docker (se volume montato localmente)
- `.ingest_cache/` - Registry file indicizzati
- `mcp_server.log` - Log server MCP

---

## Troubleshooting

### Problema: "Qdrant non raggiungibile"

**Verifica Docker**:

```bash
docker ps | grep qdrant
```

**Avvia se necessario**:

```bash
docker start qdrant-rag
```

**Verifica connettività**:

```bash
curl http://localhost:6333/health
```

### Problema: "Collection documents not found"

**Soluzione**: Esegui ingestione

```bash
python ingest.py --clean
```

### Problema: "Tutti i risultati filtrati (sotto threshold)"

**Cause possibili**:

1. Query troppo generica o non correlata ai documenti
2. Threshold troppo alto

**Soluzioni**:

1. Riformula la query in modo più specifico
2. Abbassa `similarity_threshold` in `config.yaml` (es. 0.6)
3. Verifica che i documenti siano stati indicizzati correttamente

### Problema: "ModuleNotFoundError"

**Soluzione**:

```bash
pip install -r requirements.txt
```

### Problema: Server MCP non visibile in IDE

**Antigravity**:

1. Verifica path corretti in `mcp_config.json` (usa `\\` su Windows)
2. Riavvia Antigravity completamente
3. Verifica log: `tail -f mcp_server.log`

**Claude Desktop**:

1. Verifica path in `claude_desktop_config.json`
2. Riavvia Claude Desktop
3. Verifica che Qdrant sia attivo

---

## Performance

### Hardware di Test

- **CPU**: Standard (no GPU)
- **RAM**: 8GB
- **Qdrant**: Docker locale
- **Dataset**: 6400+ chunks

### Metriche

- **Avvio server**: ~45s (caricamento modello)
- **Prima query**: ~5-8s (warm-up)
- **Query successive**: ~1-2s
- **Throughput**: ~0.5 query/sec (CPU), ~10 query/sec (GPU)

### Ottimizzazioni Possibili

- **GPU**: Riduce encoding a <0.1s
- **Modello più leggero**: 3x più veloce (es. all-MiniLM-L6-v2, 384D)
- **Qdrant remoto**: Ricerca <0.05s con più RAM
- **Caching**: Redis per query frequenti

---

## Monitoring

### Log Analysis

```bash
# Ultimi errori
grep ERROR mcp_server.log | tail -20

# Statistiche query
grep "Query completata" mcp_server.log | wc -l
```

### Statistiche Server

Da Antigravity/Claude:

```
Usa get_server_stats per vedere le metriche
```

Output:

```
📊 === STATISTICHE SERVER RAG ===

🔧 Configurazione:
  • Collection: documents
  • Documenti indicizzati: 6401
  • Similarity threshold: 0.7

📈 Metriche Query:
  • Totale query: 150
  • Successi: 142 (94.7%)
  • Tempo medio: 1.8s
```

### Health Check Qdrant

```bash
# Status generale
curl http://localhost:6333/health

# Info collection
curl http://localhost:6333/collections/documents

# Numero documenti
curl -s http://localhost:6333/collections/documents | grep points_count
```

---

## Backup

### Backup Qdrant Data

```bash
# Stop container
docker stop qdrant-rag

# Backup
tar -czf qdrant_backup_$(date +%Y%m%d).tar.gz qdrant_data/

# Restart
docker start qdrant-rag
```

### Backup Registry

```bash
cp .ingest_cache/registry.json registry_backup_$(date +%Y%m%d).json
```

### Restore

```bash
# Stop Qdrant
docker stop qdrant-rag

# Restore data
tar -xzf qdrant_backup_YYYYMMDD.tar.gz

# Restart
docker start qdrant-rag

# Restore registry
cp registry_backup_YYYYMMDD.json .ingest_cache/registry.json
```

---

## Sicurezza

### Implementato

- ✅ Input validation (previene injection, overflow)
- ✅ Timeout (previene DoS)
- ✅ Logging completo (audit trail)
- ✅ Error handling sicuro (no info sensibili in output)
- ✅ Secrets in .env (git-ignored)

### Raccomandazioni

- 🔒 Firewall: Limitare accesso Qdrant porta 6333 solo a localhost
- 🔒 TLS: Usare HTTPS per Qdrant in ambienti remoti
- 🔒 Auth: Implementare autenticazione MCP se esposto
- 🔒 Rate limiting: Limite query/minuto per utente

---

## FAQ

**Q: Posso usare un modello embedding diverso?**

A: Sì, modifica `config.yaml`:

```yaml
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimension: 384
```

Poi rigenera: `python ingest.py --clean`

**Q: Posso usare Qdrant locale invece di Docker?**

A: Sì, ma non raccomandato per produzione:

```yaml
qdrant_mode: "local"
vectorstore_path: "./qdrant_storage"
```

**Q: Come aggiungo nuovi documenti?**

A: Copia i file in `data/` ed esegui `python ingest.py` (incrementale)

**Q: Posso indicizzare più collection?**

A: Modifica `qdrant_collection` in `config.yaml` e usa `ingest.py` separatamente

**Q: Il sistema funziona offline?**

A: Sì, dopo il primo download del modello embedding (Jina v3). Solo chat.py richiede internet (Google Gemini).

---

## License

MIT

---

## Credits

- **Embedding Model**: [Jina AI Embeddings v3](https://huggingface.co/jinaai/jina-embeddings-v3)
- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **MCP Protocol**: [Anthropic](https://www.anthropic.com/)

---

**Status**: ✅ Production Ready

**Version**: 2.0

**Last Update**: 2025-12-27
