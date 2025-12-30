"""
MCP Server - Production-Ready RAG Search Tool
Versione ottimizzata per ambienti scientifici/istituzionali
"""
import sys
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import time
from functools import wraps
import time as time_module

# === FIX WINDOWS: Forza newline Unix per MCP ===
if sys.platform == "win32":
    sys.stdout.reconfigure(newline='\n')
    sys.stderr.reconfigure(newline='\n')

# === Configurazione ambiente PRIMA di qualsiasi import ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# === Setup progetto ===
PROJECT_DIR = Path(__file__).parent.resolve()

# === Configurazione Logging Strutturato ===
LOG_FILE = PROJECT_DIR / "mcp_server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp_rag")

# === Costanti di Configurazione ===
MAX_QUERY_LENGTH = 2000  # Caratteri massimi per query
MIN_QUERY_LENGTH = 3     # Caratteri minimi per query
MAX_LIMIT = 50           # Numero massimo risultati
DEFAULT_LIMIT = 10       # Limite default
QDRANT_TIMEOUT = 30      # Timeout connessione Qdrant (secondi)
QDRANT_RETRY_ATTEMPTS = 3  # Tentativi di riconnessione
QDRANT_RETRY_DELAY = 2   # Delay tra retry (secondi)
EMBEDDING_TIMEOUT = 60   # Timeout encoding (secondi)

# === Metriche Globali ===
METRICS = {
    "total_queries": 0,
    "successful_queries": 0,
    "failed_queries": 0,
    "total_results_returned": 0,
    "low_quality_queries": 0,
    "avg_query_time": 0.0,
    "last_error": None,
    "server_start_time": datetime.now().isoformat()
}


def _load_config() -> Dict[str, Any]:
    """Carica configurazione con validazione"""
    import yaml
    config_path = PROJECT_DIR / "config.yaml"

    if not config_path.exists():
        logger.error(f"File config.yaml non trovato: {config_path}")
        raise FileNotFoundError(f"config.yaml non trovato in {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Validazione campi essenziali
        required_fields = ['embedding_model', 'qdrant_mode']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Campo obbligatorio '{field}' mancante in config.yaml")

        logger.info(f"Configurazione caricata: {config_path}")
        return config

    except Exception as e:
        logger.error(f"Errore caricamento config: {e}")
        raise


def _init_qdrant_with_retry(config: Dict[str, Any]) -> QdrantClient:
    """
    Inizializza connessione Qdrant con retry logic e health check
    """
    mode = config.get('qdrant_mode', 'local')

    for attempt in range(1, QDRANT_RETRY_ATTEMPTS + 1):
        try:
            if mode == 'http':
                host = config.get('qdrant_host', 'localhost')
                port = config.get('qdrant_port', 6333)

                logger.info(f"Tentativo {attempt}/{QDRANT_RETRY_ATTEMPTS}: Connessione a Qdrant {host}:{port}")

                client = QdrantClient(
                    host=host,
                    port=port,
                    timeout=QDRANT_TIMEOUT
                )
            else:
                # Modalità locale
                path = str(PROJECT_DIR / config['vectorstore_path'])
                logger.info(f"Tentativo {attempt}/{QDRANT_RETRY_ATTEMPTS}: Connessione a Qdrant locale {path}")
                client = QdrantClient(path=path)

            # Health check: verifica che Qdrant risponda
            collections = client.get_collections()
            logger.info(f"Health check OK - {len(collections.collections)} collections disponibili")

            # Verifica esistenza collection 'documents'
            collection_name = config.get('qdrant_collection', 'documents')
            if collection_name not in [c.name for c in collections.collections]:
                logger.warning(f"Collection '{collection_name}' non trovata. Crearla con ingest.py")
            else:
                # Verifica punti nella collection
                collection_info = client.get_collection(collection_name)
                points_count = collection_info.points_count
                logger.info(f"Collection '{collection_name}': {points_count} documenti indicizzati")

                if points_count == 0:
                    logger.warning(f"Collection '{collection_name}' vuota. Eseguire ingest.py per indicizzare documenti")

            return client

        except Exception as e:
            logger.error(f"Tentativo {attempt}/{QDRANT_RETRY_ATTEMPTS} fallito: {e}")

            if attempt < QDRANT_RETRY_ATTEMPTS:
                logger.info(f"Attesa {QDRANT_RETRY_DELAY}s prima del prossimo tentativo...")
                time.sleep(QDRANT_RETRY_DELAY)
            else:
                logger.critical("Impossibile connettersi a Qdrant dopo tutti i tentativi")
                raise ConnectionError(f"Qdrant non disponibile: {e}")


def _init_embedding_model(config: Dict[str, Any]) -> SentenceTransformer:
    """
    Inizializza modello embedding con validazione
    """
    try:
        model_name = config['embedding_model']
        trust_remote = config.get('trust_remote_code', False)

        logger.info(f"Caricamento modello embedding: {model_name}")

        model = SentenceTransformer(
            model_name,
            trust_remote_code=trust_remote
        )

        # Validazione dimensione embedding
        test_embedding = model.encode("test", show_progress_bar=False)
        actual_dim = len(test_embedding)
        expected_dim = config.get('embedding_dimension', 768)

        if actual_dim != expected_dim:
            logger.warning(
                f"Dimensione embedding diversa da attesa: {actual_dim} vs {expected_dim}. "
                f"Aggiornare 'embedding_dimension' in config.yaml"
            )

        logger.info(f"Modello caricato: {actual_dim}D embeddings")
        return model

    except Exception as e:
        logger.critical(f"Errore caricamento modello: {e}")
        raise


def _validate_query_input(query: str, limit: int) -> Optional[str]:
    """
    Valida input utente
    Returns: None se valido, altrimenti messaggio errore
    """
    # Validazione query
    if not query or not isinstance(query, str):
        return "Query non valida: deve essere una stringa non vuota"

    query_stripped = query.strip()

    if len(query_stripped) < MIN_QUERY_LENGTH:
        return f"Query troppo corta: minimo {MIN_QUERY_LENGTH} caratteri (attuale: {len(query_stripped)})"

    if len(query_stripped) > MAX_QUERY_LENGTH:
        return f"Query troppo lunga: massimo {MAX_QUERY_LENGTH} caratteri (attuale: {len(query_stripped)})"

    # Validazione limit
    if not isinstance(limit, int):
        return f"Limit deve essere un intero (attuale: {type(limit).__name__})"

    if limit <= 0:
        return f"Limit deve essere positivo (attuale: {limit})"

    if limit > MAX_LIMIT:
        return f"Limit troppo alto: massimo {MAX_LIMIT} (attuale: {limit})"

    return None


# === Caricamento Configurazione ===
try:
    CONFIG = _load_config()
    SIMILARITY_THRESHOLD = CONFIG.get('similarity_threshold', 0.7)
    COLLECTION_NAME = CONFIG.get('qdrant_collection', 'documents')
    EMBEDDING_TASK_QUERY = CONFIG.get('embedding_task_query', None)
    
    # Configurazione threshold adattivo
    ADAPTIVE_THRESHOLD_ENABLED = CONFIG.get('adaptive_threshold', False)
    ADAPTIVE_THRESHOLD_MIN = CONFIG.get('adaptive_threshold_min', 0.5)
    ADAPTIVE_THRESHOLD_MAX = CONFIG.get('adaptive_threshold_max', 0.9)

    logger.info("=== MCP RAG Server - Inizializzazione ===")
    logger.info(f"Similarity threshold: {SIMILARITY_THRESHOLD} (adaptive: {ADAPTIVE_THRESHOLD_ENABLED})")
    logger.info(f"Collection: {COLLECTION_NAME}")

except Exception as e:
    logger.critical(f"Errore critico durante inizializzazione: {e}")
    sys.exit(1)

# === Pre-caricamento modello e Qdrant (EAGER LOADING) ===
sys.stderr.write("[MCP] Caricamento modello AI e connessione Qdrant...\n")
sys.stderr.flush()

try:
    MODEL = _init_embedding_model(CONFIG)
    QDRANT = _init_qdrant_with_retry(CONFIG)
    
    # === Caricamento Cross-Encoder per Reranking (opzionale) ===
    RERANK_ENABLED = CONFIG.get('rerank_enabled', False)
    RERANK_MODEL = None
    RERANK_TOP_N = CONFIG.get('rerank_top_n', 30)
    RERANK_ALPHA = CONFIG.get('rerank_alpha', 0.4)  # Peso vector_score in hybrid
    
    if RERANK_ENABLED:
        from sentence_transformers import CrossEncoder
        rerank_model_name = CONFIG.get('rerank_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info(f"Caricamento cross-encoder: {rerank_model_name}")
        RERANK_MODEL = CrossEncoder(rerank_model_name)
        logger.info("Cross-encoder caricato per reranking")
    else:
        logger.info("Reranking disabilitato")

    sys.stderr.write("[MCP] Sistema pronto.\n")
    sys.stderr.flush()
    logger.info("=== Inizializzazione completata con successo ===")

except Exception as e:
    sys.stderr.write(f"[MCP] ERRORE CRITICO: {e}\n")
    sys.stderr.flush()
    logger.critical(f"Impossibile avviare il server: {e}")
    sys.exit(1)

# === Server MCP ===
mcp = FastMCP("rag-search")


# === Retry Decorator per Query Robuste ===
def retry_on_failure(max_attempts=3, delay=1):
    """Decorator per retry automatico su errori transitori"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        logger.warning(f"Query tentativo {attempt}/{max_attempts} fallito: {e}")
                        time_module.sleep(delay)
            raise last_error
        return wrapper
    return decorator


@retry_on_failure(max_attempts=2, delay=0.5)
def _execute_search(client, collection, vector, limit, timeout):
    """Esegue ricerca Qdrant con retry automatico"""
    return client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=True,
        timeout=timeout
    )


def _deduplicate_results(results, get_score_fn, threshold=0.85):
    """
    Rimuove risultati con testo quasi identico.
    Mantiene quello con score migliore.
    """
    if len(results) <= 1:
        return results
    
    def jaccard(t1, t2, n=3):
        # N-gram Jaccard similarity
        s1 = set(t1.lower()[i:i+n] for i in range(max(0, len(t1)-n+1)))
        s2 = set(t2.lower()[i:i+n] for i in range(max(0, len(t2)-n+1)))
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)
    
    # Ordina per score decrescente (highest quality first)
    sorted_results = sorted(results, key=get_score_fn, reverse=True)
    
    unique = []
    for candidate in sorted_results:
        cand_text = candidate.payload.get('text', '')
        # Se troppo simile a uno già preso, scartalo
        is_dup = any(
            jaccard(cand_text, acc.payload.get('text', '')) >= threshold
            for acc in unique
        )
        if not is_dup:
            unique.append(candidate)
    
    return unique


@mcp.tool()
def search_knowledge_base(query: str, limit: int = DEFAULT_LIMIT) -> str:
    """Semantic search in indexed documents. Returns results with score, source path, and char position for citation."""
    start_time = time.time()
    METRICS["total_queries"] += 1

    logger.info(f"Query ricevuta: '{query[:100]}...' (limit={limit})")

    try:
        # === 1. VALIDAZIONE INPUT ===
        validation_error = _validate_query_input(query, limit)
        if validation_error:
            METRICS["failed_queries"] += 1
            logger.warning(f"Validazione fallita: {validation_error}")
            return f"❌ Errore validazione: {validation_error}"

        query_stripped = query.strip()

        # === 2. ENCODING QUERY ===
        try:
            encode_kwargs = {"show_progress_bar": False}
            if EMBEDDING_TASK_QUERY:
                encode_kwargs["task"] = EMBEDDING_TASK_QUERY

            logger.debug("Encoding query...")
            vector = MODEL.encode(query_stripped, **encode_kwargs)

            # Converti in lista Python
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            else:
                vector = list(vector)

        except Exception as e:
            METRICS["failed_queries"] += 1
            METRICS["last_error"] = str(e)
            logger.error(f"Errore encoding: {e}", exc_info=True)
            return f"❌ Errore durante encoding della query: {e}"

        # === 3. RICERCA VETTORIALE (con retry automatico) ===
        try:
            logger.debug(f"Ricerca in Qdrant (collection: {COLLECTION_NAME})...")
            
            # Se reranking attivo, recupera più candidati
            search_limit = RERANK_TOP_N if RERANK_ENABLED and RERANK_MODEL else limit

            # Usa wrapper con retry per robustezza
            results = _execute_search(
                QDRANT, COLLECTION_NAME, vector, search_limit, QDRANT_TIMEOUT
            )

        except Exception as e:
            METRICS["failed_queries"] += 1
            METRICS["last_error"] = str(e)
            logger.error(f"Errore ricerca Qdrant: {e}", exc_info=True)
            return f"❌ Errore durante ricerca nel database: {e}"
        
        # === 3.5 RERANKING (se abilitato) ===
        if RERANK_ENABLED and RERANK_MODEL and results.points:
            try:
                # Prepara coppie (query, testo) per cross-encoder
                pairs = [(query_stripped, r.payload.get('text', '')) for r in results.points]
                
                # Calcola score di reranking (logits)
                rerank_logits = RERANK_MODEL.predict(pairs, show_progress_bar=False)
                
                # Normalizza logits → probabilità [0,1] con sigmoid
                import math
                def sigmoid(x):
                    try:
                        return 1 / (1 + math.exp(-x))
                    except OverflowError:
                        return 0.0 if x < 0 else 1.0
                
                # Calcola hybrid score: α * vector_score + (1-α) * rerank_score
                alpha = RERANK_ALPHA
                for i, result in enumerate(results.points):
                    normalized_rerank = sigmoid(float(rerank_logits[i]))
                    vector_score = result.score
                    
                    # Hybrid scoring
                    result.rerank_score = alpha * vector_score + (1 - alpha) * normalized_rerank
                
                results.points.sort(key=lambda x: x.rerank_score, reverse=True)
                results.points = results.points[:limit]  # Tronca a limit richiesto
                
                logger.debug(f"Reranking hybrid (α={alpha}): {len(pairs)} candidati -> {len(results.points)} risultati")
            except Exception as e:
                logger.warning(f"Reranking fallito, uso ranking originale: {e}")

        # === 4. FILTRAGGIO PER QUALITÀ (con threshold adattivo) ===
        if not results.points:
            logger.info("Nessun risultato trovato")
            METRICS["successful_queries"] += 1
            return "ℹ️ Nessun documento rilevante trovato per questa query."
        
        # Determina quale score usare: rerank_score se disponibile, altrimenti score vettoriale
        use_rerank_score = RERANK_ENABLED and RERANK_MODEL and hasattr(results.points[0], 'rerank_score')
        
        def get_score(r):
            """Restituisce lo score appropriato (rerank o vettoriale)"""
            if use_rerank_score and hasattr(r, 'rerank_score'):
                return r.rerank_score
            return r.score
        
        # Calcola threshold (adattivo o statico) usando lo score appropriato
        scores = [get_score(r) for r in results.points]
        
        if ADAPTIVE_THRESHOLD_ENABLED and len(scores) >= 3:
            # GAP Analysis: trova il gap naturale tra risultati rilevanti e non
            sorted_scores = sorted(scores, reverse=True)
            
            # Calcola i gap tra score consecutivi
            best_gap = 0
            threshold_at_gap = SIMILARITY_THRESHOLD
            
            for i in range(len(sorted_scores) - 1):
                gap = sorted_scores[i] - sorted_scores[i + 1]
                if gap >= 0.08 and gap > best_gap:  # Gap significativo (>= 0.08)
                    best_gap = gap
                    # Threshold = valore sotto il gap + piccolo margine
                    threshold_at_gap = sorted_scores[i + 1] + 0.01
            
            if best_gap >= 0.08:
                # Trovato gap significativo
                effective_threshold = max(ADAPTIVE_THRESHOLD_MIN, 
                                         min(ADAPTIVE_THRESHOLD_MAX, threshold_at_gap))
                logger.debug(f"Threshold adattivo: {effective_threshold:.3f} (gap: {best_gap:.3f})")
            else:
                # Nessun gap → usa percentile-based
                top_half_avg = sum(sorted_scores[:len(sorted_scores)//2 + 1]) / (len(sorted_scores)//2 + 1)
                effective_threshold = max(ADAPTIVE_THRESHOLD_MIN, 
                                         min(ADAPTIVE_THRESHOLD_MAX, top_half_avg - 0.12))
                logger.debug(f"Threshold percentile: {effective_threshold:.3f}")
        else:
            effective_threshold = SIMILARITY_THRESHOLD

        # Filtra usando lo score appropriato (rerank o vettoriale)

        # Filtra usando lo score appropriato (rerank o vettoriale)
        filtered_results = [
            r for r in results.points
            if get_score(r) >= effective_threshold
        ]
        
        # Deduplicazione semantica (rimuove chunk quasi identici)
        filtered_results = _deduplicate_results(filtered_results, get_score)

        if not filtered_results:
            METRICS["low_quality_queries"] += 1
            logger.info(
                f"Risultati filtrati (max score: {get_score(results.points[0]):.3f} "
                f"< threshold: {effective_threshold:.3f})"
            )
            return (
                f"0 risultati sopra soglia {effective_threshold:.2f}. "
                f"Max score: {get_score(results.points[0]):.3f}. Riformula query."
            )

        # === 5. FORMATTAZIONE RISULTATI (compatta ma esplicativa) ===
        output = [f"Trovati {len(filtered_results)} risultati (soglia:{effective_threshold:.2f})"]

        for idx, result in enumerate(filtered_results, 1):
            source_path = result.payload.get('source_path', result.payload.get('source', '?'))
            text = result.payload.get('text', '').strip()
            chunk_id = result.payload.get('chunk_id', '?')
            char_start = result.payload.get('char_start', -1)
            char_end = result.payload.get('char_end', -1)
            score = get_score(result)
            
            # Citazione esplicita
            if char_start >= 0:
                cite = f"char:{char_start}-{char_end}"
            else:
                cite = f"chunk:{chunk_id}"

            output.append(
                f"\n[{idx}] score:{score:.3f} src:{source_path} ({cite})\n{text}"
            )

        formatted_output = "\n".join(output)

        # === 6. METRICHE ===
        elapsed = time.time() - start_time
        METRICS["successful_queries"] += 1
        METRICS["total_results_returned"] += len(filtered_results)

        # Aggiorna media tempo query (moving average)
        alpha = 0.1  # smoothing factor
        METRICS["avg_query_time"] = (
            alpha * elapsed + (1 - alpha) * METRICS["avg_query_time"]
        )

        logger.info(
            f"Query completata: {len(filtered_results)} risultati in {elapsed:.2f}s "
            f"(score range: {get_score(filtered_results[0]):.3f} - {get_score(filtered_results[-1]):.3f})"
        )

        return formatted_output

    except Exception as e:
        # Catch-all per errori imprevisti
        METRICS["failed_queries"] += 1
        METRICS["last_error"] = str(e)
        logger.error(f"Errore imprevisto: {e}", exc_info=True)
        return f"❌ Errore imprevisto: {e}"


@mcp.tool()
def get_server_stats() -> str:
    """Returns server stats: document count, query metrics, errors."""
    try:
        # Info collection
        collection_info = QDRANT.get_collection(COLLECTION_NAME)

        uptime = datetime.now() - datetime.fromisoformat(METRICS["server_start_time"])
        success_rate = (
            (METRICS["successful_queries"] / METRICS["total_queries"] * 100)
            if METRICS["total_queries"] > 0 else 0
        )

        stats = (
            f"CONFIG: documents={collection_info.points_count} embedding_dim={collection_info.config.params.vectors.size} "
            f"threshold={SIMILARITY_THRESHOLD} uptime={uptime}\n"
            f"METRICS: total_queries={METRICS['total_queries']} success={METRICS['successful_queries']}({success_rate:.0f}%) "
            f"failed={METRICS['failed_queries']} low_quality={METRICS['low_quality_queries']} "
            f"avg_time={METRICS['avg_query_time']:.2f}s\n"
            f"LAST_ERROR: {METRICS['last_error'] or 'None'}"
        )

        logger.info("Statistiche richieste")
        return stats.strip()

    except Exception as e:
        logger.error(f"Errore recupero statistiche: {e}")
        return f"Err: {e}"



@mcp.tool()
def list_collections() -> str:
    """Lists all Qdrant collections with document counts."""
    try:
        collections = QDRANT.get_collections()
        output = ["COLLECTIONS:"]
        for col in collections.collections:
            info = QDRANT.get_collection(col.name)
            output.append(f"  {col.name}: {info.points_count} docs, {info.status.name}")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Errore list_collections: {e}")
        return f"❌ Errore: {e}"


@mcp.tool()
def get_document_by_id(doc_id: str) -> str:
    """Retrieves a specific chunk by its Qdrant ID. Returns source, chunk_id, char position, and content."""
    try:
        # Prova a convertire in int se necessario, Qdrant usa entrambi
        point_id = doc_id
        if doc_id.isdigit():
            point_id = int(doc_id)

        results = QDRANT.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True
        )

        if not results:
            return f"❌ Nessun documento trovato con ID: {doc_id}"

        r = results[0]
        payload = r.payload or {}
        
        pos = f"char:{payload.get('char_start', '?')}-{payload.get('char_end', '?')}"
        return (
            f"id:{r.id} src:{payload.get('source_path', payload.get('source', '?'))} "
            f"chunk:{payload.get('chunk_id', '?')} {pos}\n"
            f"content:\n{payload.get('text', '')}"
        )
    except Exception as e:
        logger.error(f"Errore get_document_by_id: {e}")
        return f"❌ Errore: {e}"


@mcp.tool()
def list_sources() -> str:
    """Lists all indexed source files in tree format."""
    try:
        # Scroll per trovare fonti uniche (limitato per performance)
        limit_points = 2000
        sources = set()
        
        # Usa scroll API
        points, _ = QDRANT.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit_points,
            with_payload=True,
            with_vectors=False
        )
        
        for p in points:
            payload = p.payload or {}
            if 'source' in payload:
                sources.add(payload['source'])
            elif 'source_path' in payload:
                sources.add(payload['source_path'])
        
        if not sources:
            return f"SOURCES: 0 files (scanned {len(points)} points)"
        
        # Costruisci struttura ad albero
        tree = {}
        for source in sources:
            # Normalizza separatori path (Windows/Unix)
            normalized = source.replace("\\", "/")
            parts = normalized.split("/")
            
            # Naviga/crea la struttura
            current = tree
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # È un file (foglia)
                    if "__files__" not in current:
                        current["__files__"] = []
                    current["__files__"].append(part)
                else:
                    # È una directory
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        
        # Genera output (compatto: senza emoji)
        output = [f"SOURCES ({len(sources)} files, scan:{len(points)})"]
        
        def render_tree(node, prefix="", is_root=True):
            lines = []
            dirs = sorted([k for k in node.keys() if k != "__files__"])
            files = sorted(node.get("__files__", []))
            items = [(d, True) for d in dirs] + [(f, False) for f in files]
            for i, (name, is_dir) in enumerate(items):
                is_last = (i == len(items) - 1)
                conn = "" if is_root and i == 0 else ("└─" if is_last else "├─")
                npfx = prefix + ("  " if is_last else "│ ") if not (is_root and i == 0) else ""
                if is_dir:
                    lines.append(f"{prefix}{conn}{name}/")
                    lines.extend(render_tree(node[name], npfx, False))
                else:
                    lines.append(f"{prefix}{conn}{name}")
            return lines
        
        output.extend(render_tree(tree))
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Errore list_sources: {e}")
        return f"❌ Errore: {e}"


@mcp.tool()
def search_by_source(query: str, source_query: str, limit: int = 10) -> str:
    """Semantic search filtered by source file name (substring match)."""
    try:
        # Encoding query
        vector = MODEL.encode(query, show_progress_bar=False).tolist()
        
        # Crea filtro Qdrant
        # Cerchiamo substring nel campo 'source' o 'source_path'
        # Nota: Qdrant 'match' è exact, 'like' o 'text' dipende dalla config schema.
        # Per sicurezza, visto che non sappiamo lo schema, usiamo MatchValue su 
        # scroll e filtro client-side SE la collection non ha full-text index su payload.
        # TUTTAVIA: per efficienza proviamo un filtro MatchText/MatchValue se possibile
        # ma Qdrant base senza config JSON specifica potrebbe fallire su 'MatchText'.
        # Approccio ibrido chirurgico: recupero più risultati e filtro in python.
        
        search_limit = limit * 5  # Prendi di più per filtrare dopo
        
        hit = QDRANT.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=search_limit,
            with_payload=True
        )
        
        filtered = []
        source_q_lower = source_query.lower()
        
        for point in hit.points:
            payload = point.payload or {}
            p_source = payload.get('source', '').lower()
            p_path = payload.get('source_path', '').lower()
            
            if source_q_lower in p_source or source_q_lower in p_path:
                filtered.append(point)
                
        filtered = filtered[:limit]
        
        if not filtered:
            return f"0 risultati in '{source_query}'."
            
        # Formattazione compatta ma esplicativa
        output = [f"Risultati in fonte='{source_query}': {len(filtered)}"]
        for idx, r in enumerate(filtered, 1):
            payload = r.payload or {}
            txt = payload.get('text', '').strip()
            output.append(f"[{idx}] score:{r.score:.3f} src:{payload.get('source', '?')}\n{txt[:200]}...")
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Errore search_by_source: {e}")
        return f"Err: {e}"


@mcp.tool()
def multi_query_search(queries: str, limit: int = 5) -> str:
    """Runs multiple queries (separated by ';') and merges deduplicated results."""
    try:
        # Split queries
        q_list = [q.strip() for q in queries.replace('\n', ';').split(';') if q.strip()]
        
        if not q_list:
            return "Err: no queries. Use ';' or newline."
        
        all_results = {}
        for q in q_list:
            vector = MODEL.encode(q, show_progress_bar=False).tolist()
            hits = QDRANT.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=limit,
                with_payload=True
            )
            for p in hits.points:
                if (p.payload or {}) and (p.id not in all_results or p.score > all_results[p.id].score):
                    all_results[p.id] = p
                    
        final_points = sorted(all_results.values(), key=lambda x: x.score, reverse=True)[:limit*2]
        
        output = [f"MultiQuery ({len(q_list)} queries): {len(final_points)} unique_results"]
        for idx, r in enumerate(final_points, 1):
            payload = r.payload or {}
            txt = payload.get('text', '').strip()
            output.append(f"[{idx}] score:{r.score:.3f} src:{payload.get('source', '?')}\n{txt[:250]}...")
        return "\n".join(output)

    except Exception as e:
        logger.error(f"Errore multi_query_search: {e}")
        return f"Err: {e}"


@mcp.tool()
def hybrid_search(query: str, limit: int = 10, keyword_boost: float = 0.2) -> str:
    """Semantic search with keyword boost. Increases score if exact query words are found."""
    try:
        # 1. Ricerca vettoriale standard
        vector = MODEL.encode(query, show_progress_bar=False).tolist()
        
        # Chiediamo più risultati per riordinarli
        hits = QDRANT.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit * 3,
            with_payload=True
        )
        
        # 2. Reranking basato su keyword (minimo 2 caratteri per evitare rumore)
        keywords = [w.lower() for w in query.split() if len(w) > 2]
        
        for p in hits.points:
            payload = p.payload or {}
            txt = payload.get('text', '').lower()
            matches = sum(1 for k in keywords if k in txt)
            if matches > 0:
                # Boost logaritmico base
                boost = keyword_boost * matches
                p.score += boost
                
        # 3. Riordina
        hits.points.sort(key=lambda x: x.score, reverse=True)
        top_results = hits.points[:limit]
        
        if not top_results:
            return "0 results."
        
        output = [f"HybridSearch (keyword_boost:{keyword_boost}): {len(top_results)} results"]
        for idx, r in enumerate(top_results, 1):
            payload = r.payload or {}
            txt = payload.get('text', '').strip()
            output.append(f"[{idx}] score:{r.score:.3f} src:{payload.get('source', '')}\n{txt[:200]}...")
        return "\n".join(output)
        
    except Exception as e:
        logger.error(f"Errore hybrid_search: {e}")
        return f"Err: {e}"


@mcp.tool()
def get_document_context(source_name: str, chunk_id: int, chunks_before: int = 2, chunks_after: int = 2) -> str:
    """Retrieves a chunk with surrounding context (adjacent chunks). Use when a single chunk lacks context."""
    try:
        # Validazione input
        chunks_before = max(0, min(10, chunks_before))
        chunks_after = max(0, min(10, chunks_after))
        
        # Recupera tutti i chunks della fonte specificata
        limit_scan = 500  # Limite ragionevole per un singolo documento
        
        points, _ = QDRANT.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit_scan,
            with_payload=True,
            with_vectors=False
        )
        
        # Filtra per source_name (case-insensitive, substring match)
        source_lower = source_name.lower()
        matching_chunks = []
        
        for p in points:
            payload = p.payload or {}
            p_source = payload.get('source', '').lower()
            p_path = payload.get('source_path', '').lower()
            
            if source_lower in p_source or source_lower in p_path:
                matching_chunks.append(p)
        
        if not matching_chunks:
            return f"❌ Nessun chunk trovato per fonte: '{source_name}'"
        
        # Ordina per posizione nel documento (char_start o chunk_id)
        def get_position(p):
            payload = p.payload or {}
            # Preferisci char_start se disponibile, altrimenti chunk_id
            char_start = payload.get('char_start', -1)
            if char_start >= 0:
                return char_start
            return payload.get('chunk_id', 0)
        
        matching_chunks.sort(key=get_position)
        
        # Trova l'indice del chunk richiesto
        target_idx = None
        for idx, p in enumerate(matching_chunks):
            p_chunk_id = (p.payload or {}).get('chunk_id', -1)
            if p_chunk_id == chunk_id:
                target_idx = idx
                break
        
        if target_idx is None:
            # Prova a cercare per ID Qdrant
            for idx, p in enumerate(matching_chunks):
                if str(p.id) == str(chunk_id):
                    target_idx = idx
                    break
        
        if target_idx is None:
            available_ids = [str((p.payload or {}).get('chunk_id', p.id)) for p in matching_chunks[:20]]
            return (
                f"❌ Chunk ID {chunk_id} non trovato in '{source_name}'.\n"
                f"Chunk disponibili (primi 20): {', '.join(available_ids)}"
            )
        
        # Calcola range di chunks da restituire
        start_idx = max(0, target_idx - chunks_before)
        end_idx = min(len(matching_chunks), target_idx + chunks_after + 1)
        
        context_chunks = matching_chunks[start_idx:end_idx]
        
        # Formatta output compatto ma esplicativo
        source_full = (matching_chunks[0].payload or {}).get('source_path', source_name)
        output = [f"Context src:{source_full} target_chunk:{chunk_id}(index:{target_idx}) range:{start_idx}-{end_idx-1} ({len(context_chunks)} chunks)"]
        
        for idx, p in enumerate(context_chunks):
            payload = p.payload or {}
            text = payload.get('text', '').strip()
            is_target = (start_idx + idx) == target_idx
            marker = "[TARGET]" if is_target else ""
            output.append(f"chunk:{payload.get('chunk_id', '?')} char:{payload.get('char_start', '?')}-{payload.get('char_end', '?')} {marker}\n{text}")
        
        return "\n---\n".join(output)
        
    except Exception as e:
        logger.error(f"Errore get_document_context: {e}", exc_info=True)
        return f"❌ Errore: {e}"


# === Entry point ===
if __name__ == "__main__":
    try:
        logger.info("Avvio MCP server in modalità stdio...")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server interrotto dall'utente")
    except Exception as e:
        logger.critical(f"Errore fatale: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server terminato")
