"""MCP Server - Server per integrazioni Claude/ChatGPT/Gemini"""

import logging
from typing import Optional
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from utils import load_config, load_environment, get_api_key

try:
    from datapizza.clients.google import GoogleClient
    from datapizza.embedders import ClientEmbedder
except ImportError as e:
    raise ImportError(f"Installa dipendenze: pip install -r requirements.txt\nErrore: {e}")

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carica configurazione
load_environment()
config = load_config("config.yaml")

# Crea Server MCP
mcp = FastMCP("RAG Search Server")

# Inizializza componenti globali
api_key = get_api_key("GOOGLE_API_KEY")

# Client Google per embedding
google_client = GoogleClient(
    api_key=api_key,
    model=config['embedding_model']
)
embedder = ClientEmbedder(client=google_client)

# Vector store (Qdrant locale)
# Usa la stessa logica di ingest.py per il path
store_path = config['vectorstore_path']
qdrant_client = QdrantClient(path=str(store_path))
collection_name = "documents"


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> str:
    """
    Ricerca documenti nella knowledge base aziendale.
    Usa questo tool quando l'utente fa domande su documenti, procedure o dati interni indicizzati.

    Args:
        query: La frase o domanda da cercare.
        top_k: Numero di risultati da ritornare (default 5).
    """
    try:
        # Genera embedding query
        query_embedding = embedder.embed(query)

        # Cerca in Qdrant
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        if not results:
            return "Nessun risultato trovato."

        # Formatta output
        output = [f"Trovati {len(results)} risultati per '{query}':\n"]
        for i, res in enumerate(results, 1):
            source = res.payload.get('source', 'Sconosciuto')
            text = res.payload.get('text', '')
            score = res.score
            output.append(f"[{i}] Fonte: {source} (Score: {score:.2f})")
            output.append(f"Contenuto: {text}\n")

        return "\n".join(output)

    except Exception as e:
        return f"Errore durante la ricerca: {str(e)}"


@mcp.resource("config://status")
def get_status() -> str:
    """Restituisce lo stato del server e della collezione Qdrant"""
    try:
        col = qdrant_client.get_collection(collection_name)
        return f"Status: {col.status}, Vectors: {col.points_count}"
    except Exception as e:
        return f"Error checking status: {e}"


if __name__ == "__main__":
    # Avvia server MCP su stdio (default per FastMCP)
    mcp.run()
