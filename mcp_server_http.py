"""
MCP Server SSE - Server RAG per Claude Code

Avvia il server RAG in modalità SSE per connettersi a Claude Code,
evitando il timeout durante il caricamento del modello AI pesante.

Uso:
    python mcp_server_http.py
    python mcp_server_http.py --host 0.0.0.0 --port 9000

Endpoint: http://127.0.0.1:8765/sse

Configurazione Claude Code:
    claude mcp add --transport sse --scope user rag-search http://127.0.0.1:8765/sse
"""
import sys
from pathlib import Path

# Aggiungi directory progetto al path
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

# Configura porta PRIMA di importare mcp
import os
os.environ["FASTMCP_PORT"] = "8765"

# Importa il server MCP già configurato
from mcp_server import mcp, logger

# Configurazione
HOST = "127.0.0.1"
PORT = 8765

if __name__ == "__main__":
    # Supporta argomenti opzionali per host/port
    import argparse
    parser = argparse.ArgumentParser(description="MCP Server SSE")
    parser.add_argument("--host", default=HOST, help="Host address")
    parser.add_argument("--port", type=int, default=PORT, help="Port number")
    args = parser.parse_args()
    
    try:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        
        endpoint = f"http://{args.host}:{args.port}/sse"
        
        logger.info(f"Avvio MCP server SSE su {args.host}:{args.port}...")
        print(f"[MCP] ✅ Server SSE in ascolto su {endpoint}")
        print(f"[MCP] Config: claude mcp add --transport sse --scope user rag-search {endpoint}")

        mcp.run(transport="sse")

    except KeyboardInterrupt:
        logger.info("Server interrotto dall'utente")
    except Exception as e:
        logger.critical(f"Errore fatale: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server terminato")
