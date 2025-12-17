# RAG System

Sistema RAG (Retrieval-Augmented Generation) semplice e potente. Ingestione documenti, chatbot e server MCP in poche righe.

## Requisiti

- Python 3.10+
- Google API Key ([ottienila qui](https://aistudio.google.com/apikey))

## Setup

```bash
# 1. Crea ambiente virtuale
python -m venv .venv

# 2. Attiva ambiente
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Configura API key
# Copia .env.example in .env e aggiungi la tua chiave
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac
```

Modifica `.env`:
```
GOOGLE_API_KEY=la_tua_chiave_qui
```

## Utilizzo

### 1. Ingestione Documenti

Copia i tuoi documenti nella cartella `data/`, poi:

```bash
python ingest.py           # Aggiunge nuovi documenti
python ingest.py --clean   # Pulisce DB e re-ingestisce tutto
```

**Quando usare `--clean`**: Dopo aver modificato o eliminato documenti esistenti.

### 2. Chatbot Interattivo

```bash
python chat.py
```

Comandi disponibili:
- `/help` - Mostra aiuto
- `/stats` - Statistiche database
- `/exit` - Esci

### 3. Server MCP (per ChatGPT/Claude)

```bash
python mcp_server.py
```

Server disponibile su `http://127.0.0.1:8080`

**Endpoints:**
| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Statistiche database |
| POST | `/search` | Ricerca documenti |
| GET | `/tools` | Tool disponibili |

**Esempio ricerca:**
```bash
curl -X POST http://127.0.0.1:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "la tua domanda", "top_k": 5}'
```

## Configurazione MCP per ChatGPT

Aggiungi al tuo client MCP:

```json
{
  "mcpServers": {
    "rag-system": {
      "url": "http://127.0.0.1:8080",
      "tools": ["search_documents"]
    }
  }
}
```

## Formati Supportati (60+)

| Categoria | Formati |
|-----------|---------|
| **Office** | PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, ODT, ODP, ODS |
| **Web** | HTML, HTM, MHTML, XML, MD, RST, TEX |
| **Codice** | PY, IPYNB, JS, TS, JSON, YAML, JAVA, CPP, GO, e altri |
| **Audio** | MP3, WAV, M4A, FLAC, OGG, AAC (trascrizione automatica) |
| **Immagini** | PNG, JPG, GIF, BMP, TIFF, WEBP, SVG (OCR automatico) |
| **Sottotitoli** | VTT, SRT, ASS, SSA |
| **Archivi** | ZIP, TAR, GZ, RAR, 7Z (estrazione automatica) |

## Configurazione

Modifica `config.yaml` per personalizzare:

```yaml
# Percorsi
documents_path: "./data"        # Cartella documenti
vectorstore_path: "./qdrant_storage"  # Database vettoriale

# Elaborazione
chunk_size: 1024                # Dimensione chunk
chunk_overlap: 200              # Sovrapposizione chunk

# Ricerca
top_k: 5                        # Risultati per query
similarity_threshold: 0.7       # Soglia similarita (0-1)

# LLM
model: "gemini-2.0-flash-exp"   # Modello Gemini
temperature: 0.7                # Creativita (0-1)

# Server
mcp_port: 8080                  # Porta server
mcp_host: "127.0.0.1"           # Host server
```

## Struttura Progetto

```
rag-system/
├── data/               # Metti i documenti qui
├── qdrant_storage/     # Database (auto-generato)
├── ingest.py           # Script ingestione
├── chat.py             # Chatbot interattivo
├── mcp_server.py       # Server MCP
├── utils.py            # Funzioni condivise
├── config.yaml         # Configurazione
├── .env                # API key (da creare)
├── .env.example        # Template API key
└── requirements.txt    # Dipendenze
```

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| `GOOGLE_API_KEY non trovata` | Crea `.env` con `GOOGLE_API_KEY=...` |
| `Nessun documento trovato` | Copia file nella cartella `data/` |
| `Porta 8080 in uso` | Cambia `mcp_port` in `config.yaml` |
| `ModuleNotFoundError` | Esegui `pip install -r requirements.txt` |
| `Errore embedding` | Verifica che la API key sia valida |

## Licenza

MIT License
