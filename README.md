# RAG System - Local Multimodal RAG + Memory

Sistema RAG locale con embedding multimodali, memoria conversazionale (mem0) e tool calling nativo.

## Architettura

```
main.py (entry point)
  ├── chat.py       → Chat ricorsiva con tool calling (LLM sceglie i tool)
  │     ├── llm.py  → Client OpenAI-compatible (Ollama, LM Studio, vLLM, ecc.)
  │     ├── search.py → Ricerca semantica + reranking (Jina Reranker v2)
  │     ├── memory.py → mem0 (memoria conversazionale + knowledge base)
  │     └── markdown_writer.py → Generazione/editing file .md
  ├── ingest.py     → Indicizzazione documenti (RAG + mem0)
  │     └── extractors.py → dots.ocr (PDF/img), Whisper (audio), ecc.
  └── config.yaml   → Configurazione centralizzata unica
```

## Features

- **Tutto locale**: LLM, embedding, dati, memoria - zero dipendenze cloud
- **Tool Calling nativo**: il LLM decide autonomamente quando usare RAG, memoria, o creare file
- **Dual embedding multimodale**: Jina v4 (primario) + NVIDIA Nemotron (secondario)
- **mem0 Memory**: memoria persistente condivisa con la knowledge base RAG
- **dots.ocr multilingua**: supporto nativo per Italiano e Inglese
- **Whisper turbo**: trascrizione audio
- **Reranking**: Jina Reranker v2 con threshold adattivo
- **Generazione Markdown**: crea/modifica file .md su richiesta
- **Configurazione flessibile**: mix di `config.yaml` e variabili d'ambiente (.env)
- **Makefile**: gestione semplificata di tutti i processi

## Quick Start

### 1. Setup Iniziale

```bash
# Configura l'ambiente
cp .env.example .env

# Avvia i servizi (Qdrant + Ollama) e scarica i modelli LLM
make install
```

### 2. Indicizzazione e Chat

```bash
# Metti i tuoi file in ./data/ e indicizzali
make ingest

# Avvia la chat interattiva
make chat
```

## Uso

### Comandi Rapidi (Makefile)

| Comando        | Descrizione                                     |
| -------------- | ----------------------------------------------- |
| `make install` | Setup iniziale: avvia servizi e scarica modelli |
| `make start`   | Avvia i servizi (se già installati)             |
| `make ingest`  | Indicizza nuovi documenti in `./data/`          |
| `make chat`    | Avvia l'interfaccia di chat                     |
| `make status`  | Mostra statistiche del database vettoriale      |
| `make stop`    | Ferma tutti i servizi                           |
| `make clean`   | Rimuove container, volumi e file generati       |

### Ricerca Veloce da CLI

Puoi effettuare una ricerca rapida senza entrare in chat:

```bash
make search QUERY="tua domanda"
```

### Comandi Chat

All'interno della sessione di chat (`make chat`), puoi usare i seguenti comandi:

| Comando     | Descrizione                                              |
| ----------- | -------------------------------------------------------- |
| `/help`     | Mostra la lista dei comandi                              |
| `/stats`    | Statistiche del sistema                                  |
| `/sources`  | Lista dei documenti indicizzati                          |
| `/save <n>` | Salva la conversazione in un file markdown               |
| `/memory`   | Visualizza i fatti salvati nella memoria a lungo termine |
| `/clear`    | Resetta il contesto della conversazione attuale          |
| `/exit`     | Chiude la sessione                                       |

## Configurazione

Il sistema utilizza una configurazione gerarchica. Le variabili d'ambiente hanno la precedenza su `config.yaml`.

### Variabili d'Ambiente (.env)

| Variabile        | Descrizione                            | Default (config.yaml)    |
| ---------------- | -------------------------------------- | ------------------------ |
| `LLM_API_KEY`    | API Key per LLM                        | `ollama`                 |
| `LLM_BASE_URL`   | Endpoint API OpenAI-compatible         | `http://ollama:11434/v1` |
| `LLM_MODEL`      | Modello LLM da utilizzare              | `qwen3.5:4b`             |
| `QDRANT_HOST`    | Host di Qdrant                         | `qdrant`                 |
| `QDRANT_PORT`    | Port di Qdrant                         | `6333`                   |
| `QDRANT_URL`     | URL completo di Qdrant (es. per Cloud) | -                        |
| `DOCUMENTS_PATH` | Percorso cartella documenti            | `./data`                 |
| `OUTPUT_PATH`    | Percorso cartella output markdown      | `./output`               |

### File config.yaml

Tutto il resto è configurabile in `config.yaml`:

- **embeddings**: Dual model (Jina v4 / Nemotron), modalita' auto/primary/secondary
- **search**: Top-k, threshold adattivo, reranking Jina v2
- **memory**: mem0 con Qdrant + Ollama locale
- **ocr**: Configurazione engine e lingue (default: `en`, `it`)
- **extensions**: Formati file supportati per categoria

## Stack Tecnologico

| Componente | Tecnologia                                            |
| ---------- | ----------------------------------------------------- |
| LLM        | Qualsiasi OpenAI-compatible (Ollama, LM Studio, vLLM) |
| Embedding  | Jina v4 (1024D) / NVIDIA Nemotron (4096D)             |
| Reranking  | Jina Reranker v2 Multilingual                         |
| Vector DB  | Qdrant                                                |
| Memoria    | mem0                                                  |
| OCR        | dots.ocr (rednote-hilab)                              |
| Audio      | Whisper turbo                                         |
| Framework  | DataPizza                                             |

## Formati Supportati

| Categoria     | Estensioni                                                                                        | Estrazione                             |
| ------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Testo/Codice  | .txt, .md, .py, .js, .ts, .json, .yaml, .html, .css, .csv, .java, .cpp, .go, .rb, .php, .sh, .sql | Lettura diretta                        |
| PDF           | .pdf                                                                                              | dots.ocr                               |
| Immagini      | .png, .jpg, .jpeg, .gif, .bmp, .tiff, .webp                                                       | dots.ocr                               |
| Audio         | .mp3, .m4a, .opus, .wav, .ogg, .flac                                                              | Whisper turbo                          |
| Video         | .mp4, .mkv, .avi, .mov, .webm                                                                     | Whisper (audio) + dots.ocr (keyframes) |
| Notebook      | .ipynb                                                                                            | Parsing JSON celle                     |
| Excel         | .xlsx, .xls                                                                                       | openpyxl                               |
| Presentazioni | .pptx                                                                                             | python-pptx                            |
