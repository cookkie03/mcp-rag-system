"""MCP Server - RAG Search Tool"""

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from utils import load_config, load_environment

load_environment()
config = load_config("config.yaml")

mcp = FastMCP("RAG Search")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
qdrant = QdrantClient(path=str(config['vectorstore_path']))

@mcp.tool()
def search_knowledge_base(query: str, limit: int = 5) -> str:
    """Cerca nei documenti indicizzati. Usa per domande su contenuti locali."""
    try:
        results = qdrant.search("documents", query_vector=model.encode(query).tolist(), limit=limit)
        if not results:
            return "Nessun risultato."
        return "\n\n".join([f"[{r.payload.get('source','?')}] {r.payload.get('text','')}" for r in results])
    except Exception as e:
        return f"Errore: {e}"

if __name__ == "__main__":
    mcp.run()
