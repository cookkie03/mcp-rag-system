"""MCP Server - RAG Search Tool"""
import sys
import os

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

from pathlib import Path
from mcp.server.fastmcp import FastMCP

# === Setup progetto ===
PROJECT_DIR = Path(__file__).parent.resolve()

def _load_config():
    import yaml
    config_path = PROJECT_DIR / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG = _load_config()

# === Pre-caricamento modello e Qdrant ===
sys.stderr.write("[MCP] Caricamento modello AI...\n")
sys.stderr.flush()

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model_kwargs = {"trust_remote_code": CONFIG.get('trust_remote_code', False)}
MODEL = SentenceTransformer(CONFIG['embedding_model'], **model_kwargs)
EMBEDDING_TASK_QUERY = CONFIG.get('embedding_task_query', None)
QDRANT = QdrantClient(
    host=CONFIG.get('qdrant_host', 'localhost'),
    port=CONFIG.get('qdrant_port', 6333)
)

sys.stderr.write("[MCP] Modello pronto.\n")
sys.stderr.flush()

# === Server MCP ===
mcp = FastMCP("rag-search")

# === Tool ===
@mcp.tool()
def search_knowledge_base(query: str, limit: int = 10) -> str:
    """Cerca nei documenti indicizzati utile per rispondere a domande su contenuti locali."""
    try:
        encode_kwargs = {}
        if EMBEDDING_TASK_QUERY:
            encode_kwargs["task"] = EMBEDDING_TASK_QUERY
        vector = MODEL.encode(query, **encode_kwargs).tolist()
        results = QDRANT.query_points(
            collection_name=CONFIG.get('qdrant_collection', 'documents'),
            query=vector,
            limit=limit,
            with_payload=True
        )
        
        if not results.points:
            return "Nessun risultato trovato."
        
        output = []
        for r in results.points:
            source = r.payload.get('source', '?')
            text = r.payload.get('text', '').strip()
            output.append(f"[{source}]\n{text}")
        
        return "\n\n---\n\n".join(output)
        
    except Exception as e:
        return f"Errore: {e}"

# === Entry point ===
if __name__ == "__main__":
    mcp.run(transport="stdio")
