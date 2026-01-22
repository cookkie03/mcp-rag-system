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

### 3. Pre-caricamento Modelli AI (Importante!)

⚠️ **Prima di usare il server MCP con un IDE**, è necessario pre-scaricare i modelli AI. Altrimenti l'IDE andrà in timeout durante il primo avvio.

```bash
# Esegui una volta per scaricare e cachare i modelli
python mcp_server.py
```

Attendi fino a vedere:

```
[MCP] Caricamento modello AI e connessione Qdrant...
[MCP] Sistema pronto.
```

Poi premi `Ctrl+C` per terminare. I modelli saranno salvati nella cache di HuggingFace (`~/.cache/huggingface/`) e i successivi avvii saranno molto più veloci.

**Nota**: Il primo download richiede ~5-10 minuti (modello ~1-2GB). Le esecuzioni successive caricano dalla cache in ~30-60 secondi.

### 4. Configurazione

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

### Modalità di Connessione

Il server può essere avviato in **3 modalità diverse**:

#### 1. **Modalità STDIO** (default - per Claude Desktop, VS Code, Cursor)

```bash
python mcp_server.py
```

Comunicazione diretta stdin/stdout - **consigliata per la maggior parte degli IDE**.

#### 2. **Modalità HTTP** (per client web o test)

```bash
python mcp_server_http.py
```

- **URL**: `http://127.0.0.1:8765/sse`
- Usa Server-Sent Events (SSE) su HTTP
- Ideale per sviluppo e testing

#### 3. **Modalità HTTPS** (per Claude Code e client che richiedono SSL)

```bash
python mcp_server_https.py
```

- **URL**: `https://127.0.0.1:8766/sse`
- Usa Server-Sent Events (SSE) su HTTPS con certificato auto-firmato
- Richiesto da alcuni client (es. Claude Code)
- Genera automaticamente certificati SSL self-signed alla prima esecuzione

#### Avvio Rapido di Entrambi (HTTP + HTTPS)

**Windows**:

```batch
start_servers.bat
```

Questo avvia **contemporaneamente**:

- Server HTTP su porta `8765`
- Server HTTPS su porta `8766`

Entrambi condividono lo stesso backend RAG e possono essere usati in parallelo senza conflitti.

**Requisiti per HTTPS**:

```bash
pip install cryptography  # Per generazione automatica certificati
```

Oppure genera manualmente i certificati:

```bash
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=localhost"
```

---

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

## Setup Claude Desktop (App)

Claude Desktop è l'applicazione standalone di Anthropic. Supporta due modalità di connessione al server MCP.

### Metodo 1: STDIO (semplice, ma con possibile timeout)

⚠️ **Nota**: Al primo avvio, il caricamento del modello può richiedere 1-2 minuti e causare timeout. Per aggirare il problema eseguire da terminale mcp_server.py, in modo tale da mantenere in memoria il modello ed aggirare il timeout di claude.

**File Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**File Linux/Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "C:\\Path\\to\\file-search\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Path\\to\\file-search\\mcp_server.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### Metodo 2: SSE Pre-avviato (consigliato, nessun timeout)

**Step 1**: Avvia il server SSE in un terminale separato:

```powershell
cd C:\Users\lucam\Desktop\file-search
.\.venv\Scripts\Activate.ps1
python mcp_server_http.py
```

**Step 2**: Configura Claude Desktop per connettersi via SSE:

```json
{
  "mcpServers": {
    "rag-search": {
      "url": "http://127.0.0.1:8765/sse"
    }
  }
}
```

### Utilizzo

1. Salva il file di configurazione
2. Riavvia Claude Desktop completamente

---

## Setup Claude Code (CLI + VS Code Extension)

⚠️ **Importante**: Il caricamento del modello AI richiede 1-2 minuti. Claude Code ha un timeout che causa errori se il modello non è ancora caricato. La soluzione è avviare il server SSE **prima** di usare Claude Code.

### Perché usare il server pre-avviato?

| Modalità              | Pro                                  | Contro                                              |
| --------------------- | ------------------------------------ | --------------------------------------------------- |
| **STDIO** (default)   | Setup semplice                       | Timeout al primo avvio se il modello non è in cache |
| **SSE (pre-avviato)** | Nessun timeout, modello già caricato | Richiede avviare il server separatamente            |

### Setup in 3 Step

#### Step 1: Avvia il server SSE (una sola volta)

Apri un terminale e lascialo in esecuzione:

```powershell
cd C:\Users\lucam\Desktop\file-search
.\.venv\Scripts\Activate.ps1
python mcp_server_http.py
```

Attendi fino a vedere:

```
[MCP] ✅ Server SSE in ascolto su http://127.0.0.1:8765/sse
```

#### Step 2: Configura Claude Code CLI

In un **altro terminale**, esegui:

```powershell
# Rimuovi eventuale configurazione precedente
claude mcp remove rag-search --scope user

# Aggiungi il server SSE
claude mcp add --transport sse --scope user rag-search http://127.0.0.1:8765/sse

# Verifica la connessione
claude mcp list
```

Dovresti vedere:

```
rag-search: http://127.0.0.1:8765/sse (SSE) - ✓ Connected
```

#### Step 3: Usa i tool RAG

Ora puoi usare Claude Code normalmente. I tool RAG saranno disponibili automaticamente:

```bash
# Esempio di utilizzo in Claude Code
> Cerca nei documenti indicizzati informazioni su "machine learning"
```

### Comandi utili

| Comando                                                                            | Descrizione                           |
| ---------------------------------------------------------------------------------- | ------------------------------------- |
| `claude mcp list`                                                                  | Mostra tutti i server MCP configurati |
| `claude mcp remove rag-search --scope user`                                        | Rimuove il server                     |
| `claude mcp add --transport sse --scope user rag-search http://127.0.0.1:8765/sse` | Aggiunge il server                    |

### Tool disponibili in Claude Code

Quando il server è connesso, Claude ha accesso a questi tool:

- **`search_knowledge_base`**: Ricerca semantica nei documenti
- **`list_sources`**: Elenco file indicizzati
- **`get_server_stats`**: Statistiche del server
- **`search_by_source`**: Ricerca filtrata per fonte
- **`get_document_by_id`**: Recupera documento specifico
- **`get_document_context`**: Ottiene contesto attorno a un chunk

### Troubleshooting

**Problema**: `Failed to connect`

- Verifica che il server SSE sia in esecuzione nel terminale
- Controlla che la porta 8765 non sia usata da altri processi

**Problema**: Tool non disponibili

- Riavvia Claude Code (chiudi e riapri VS Code)
- Verifica con `claude mcp list` che il server sia connesso

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
rerank_alpha: 0.4 # Hybrid scoring: peso vector_score (0=solo rerank, 1=solo vector)

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
- `.cache/` - Registry file indicizzati & Log server MCP

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
3. Verifica log: `tail -f .cache/mcp_server.log`

**Claude Desktop**:

1. Verifica path in `claude_desktop_config.json`
2. Riavvia Claude Desktop
3. Verifica che Qdrant sia attivo

### Problema: "Timeout" o "Modello non caricato" all'avvio MCP

**Causa**: Gli IDE (Antigravity, Claude Desktop, VS Code) hanno un timeout per l'avvio del server MCP. Il caricamento dei modelli AI (Jina Embeddings v3 + Cross-Encoder) può richiedere 1-2 minuti alla prima esecuzione, superando questo timeout.

**Sintomi**:

- L'IDE mostra errore di timeout o connessione fallita
- Il server MCP non compare tra i tool disponibili
- Messaggi tipo "failed to load model" o "connection closed"

**Soluzione**: Pre-caricare i modelli **una volta** eseguendo il server manualmente:

```bash
cd c:\path\to\file-search
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

python mcp_server.py
```

Attendi fino a vedere `[MCP] Sistema pronto.`, poi chiudi con `Ctrl+C`.

I modelli vengono salvati nella cache HuggingFace (`~/.cache/huggingface/`). Le esecuzioni successive saranno più veloci (~30-60s invece di minuti).

**Soluzioni alternative**:

1. **Disabilita reranking** per dimezzare il tempo di avvio:

   ```yaml
   # config.yaml
   rerank_enabled: false
   ```

2. **Usa un modello più leggero** (meno preciso ma più veloce):

   ```yaml
   # config.yaml
   embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
   embedding_dimension: 384
   trust_remote_code: false
   ```

   Poi rigenera: `python ingest.py --clean`

3. **Aumenta il timeout dell'IDE** (se supportato), es. per Antigravity:
   ```json
   {
     "mcpServers": {
       "rag-search": {
         "command": "...",
         "args": ["..."],
         "timeout": 120000,
         "startupTimeout": 120000
       }
     }
   }
   ```

### Problema: "SSL: CERTIFICATE_VERIFY_FAILED" su Claude Desktop

**Causa**: Claude Desktop con connettore HTTP/HTTPS richiede un certificato SSL valido. Se usi un MCP server custom su localhost, OpenSSL potrebbe non avere il file di configurazione necessario.

**Sintomi**:

- Errore SSL durante la connessione tra Claude Desktop e MCP server
- OpenSSL fallisce con `Can't open openssl.cnf`
- Message tipo "missing equal sign" nel log di OpenSSL

**Soluzione**: Genera certificati auto-firmati con OpenSSL

1. **Crea file di configurazione OpenSSL** (`openssl.cnf`):

```powershell
# PowerShell - Esegui uno per uno
Remove-Item .\openssl.cnf -Force -ErrorAction SilentlyContinue

@'
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = localhost

[v3_req]
basicConstraints = CA:TRUE
keyUsage = critical, digitalSignature, keyEncipherment, keyCertSign
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
'@ | Out-File -FilePath .\openssl.cnf -Encoding ASCII
```

2. **Genera certificato auto-firmato**:

```powershell
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -sha256 -config .\openssl.cnf
```

3. **Verifica creazione file**:

```powershell
Get-ChildItem .\key.pem, .\cert.pem -ErrorAction SilentlyContinue
```

Dovrai vedere:

```
    Directory: C:\Users\username\Desktop\file-search

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---           2025-01-03  14:30          1704 cert.pem
-a---           2025-01-03  14:30          1708 key.pem
```

4. **Configura Claude Desktop** per usare il certificato:

Modifica `~/.claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-search": {
      "command": "python",
      "args": ["c:\\path\\to\\file-search\\mcp_server.py"],
      "env": {
        "SSL_CERT_FILE": "c:\\path\\to\\file-search\\cert.pem",
        "SSL_KEY_FILE": "c:\\path\\to\\file-search\\key.pem"
      }
    }
  }
}
```

5. **Riavvia Claude Desktop** e verifica che il connettore sia visibile.

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
grep ERROR .cache/mcp_server.log | tail -20

# Statistiche query
grep "Query completata" .cache/mcp_server.log | wc -l
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
cp .cache/registry.json registry_backup_$(date +%Y%m%d).json
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
cp registry_backup_YYYYMMDD.json .cache/registry.json
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
