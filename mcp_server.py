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


@mcp.tool()
def search_knowledge_base(query: str, limit: int = DEFAULT_LIMIT) -> str:
    """
    Cerca nei documenti indicizzati tramite ricerca semantica vettoriale.

    Args:
        query: Testo della domanda o ricerca (3-2000 caratteri)
        limit: Numero massimo di risultati (1-50, default: 10)

    Returns:
        Risultati formattati con score di similarità e fonte
    """
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

        # === 3. RICERCA VETTORIALE ===
        try:
            logger.debug(f"Ricerca in Qdrant (collection: {COLLECTION_NAME})...")
            
            # Se reranking attivo, recupera più candidati
            search_limit = RERANK_TOP_N if RERANK_ENABLED and RERANK_MODEL else limit

            results = QDRANT.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=search_limit,
                with_payload=True,
                timeout=QDRANT_TIMEOUT
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
        filtered_results = [
            r for r in results.points
            if get_score(r) >= effective_threshold
        ]

        if not filtered_results:
            METRICS["low_quality_queries"] += 1
            logger.info(
                f"Risultati filtrati (max score: {results.points[0].score:.3f} "
                f"< threshold: {effective_threshold:.3f})"
            )
            return (
                f"ℹ️ Trovati {len(results.points)} risultati ma tutti sotto la soglia "
                f"di qualità ({effective_threshold:.2f}).\n\n"
                f"Miglior match: {results.points[0].score:.3f} - "
                f"Prova a riformulare la query in modo più specifico."
            )

        # === 5. FORMATTAZIONE RISULTATI ===
        threshold_info = f"{effective_threshold:.2f}"
        if ADAPTIVE_THRESHOLD_ENABLED:
            threshold_info += " (adattivo)"
        
        output = [
            f"✅ Trovati {len(filtered_results)} risultati rilevanti "
            f"(soglia qualità: {threshold_info}):\n"
        ]

        for idx, result in enumerate(filtered_results, 1):
            source = result.payload.get('source', 'Fonte sconosciuta')
            source_path = result.payload.get('source_path', source)
            text = result.payload.get('text', '').strip()
            chunk_id = result.payload.get('chunk_id', '?')
            char_start = result.payload.get('char_start', -1)
            char_end = result.payload.get('char_end', -1)
            
            # Usa lo score appropriato (rerank se disponibile)
            score = get_score(result)

            # Indicatore qualità basato su score
            if score >= 0.9:
                quality = "🟢 Eccellente"
            elif score >= 0.8:
                quality = "🟡 Buona"
            else:
                quality = "🟠 Sufficiente"
            
            # Formatta posizione e citazione se disponibile
            if char_start >= 0 and char_end >= 0:
                position_info = f"📍 Posizione: caratteri {char_start}-{char_end}\n"
                citation = f"📝 Citazione: \"{source_path}\", char. {char_start}-{char_end}"
            else:
                position_info = ""
                citation = f"📝 Citazione: \"{source_path}\", chunk {chunk_id}"

            output.append(
                f"[Risultato {idx}/{len(filtered_results)}]\n"
                f"📄 Fonte: {source_path} (chunk {chunk_id})\n"
                f"{position_info}"
                f"🎯 Rilevanza: {score:.3f} {quality}\n"
                f"{citation}\n\n"
                f"{text}\n"
            )

        formatted_output = "\n---\n\n".join(output)

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
            f"(score range: {filtered_results[0].score:.3f} - {filtered_results[-1].score:.3f})"
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
    """
    Restituisce statistiche e metriche del server RAG.
    Utile per monitoring e debugging in ambienti di produzione.
    """
    try:
        # Info collection
        collection_info = QDRANT.get_collection(COLLECTION_NAME)

        uptime = datetime.now() - datetime.fromisoformat(METRICS["server_start_time"])
        success_rate = (
            (METRICS["successful_queries"] / METRICS["total_queries"] * 100)
            if METRICS["total_queries"] > 0 else 0
        )

        stats = f"""
📊 === STATISTICHE SERVER RAG ===

🔧 Configurazione:
  • Collection: {COLLECTION_NAME}
  • Documenti indicizzati: {collection_info.points_count}
  • Dimensione embedding: {collection_info.config.params.vectors.size}
  • Similarity threshold: {SIMILARITY_THRESHOLD}
  • Uptime: {uptime}

📈 Metriche Query:
  • Totale query: {METRICS["total_queries"]}
  • Successi: {METRICS["successful_queries"]} ({success_rate:.1f}%)
  • Fallimenti: {METRICS["failed_queries"]}
  • Query bassa qualità: {METRICS["low_quality_queries"]}
  • Risultati totali: {METRICS["total_results_returned"]}
  • Tempo medio: {METRICS["avg_query_time"]:.2f}s

⚠️ Ultimo errore: {METRICS["last_error"] or "Nessuno"}

🕐 Timestamp: {datetime.now().isoformat()}
        """

        logger.info("Statistiche richieste")
        return stats.strip()

    except Exception as e:
        logger.error(f"Errore recupero statistiche: {e}")
        return f"❌ Errore recupero statistiche: {e}"


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
