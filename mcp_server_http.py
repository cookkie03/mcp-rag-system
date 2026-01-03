"""
C:\\Users\\lucam\\Desktop\\file-search\\.venv\\Scripts\\python.exe C:\\Users\\lucam\\Desktop\\file-search\\mcp_server_http.py
MCP Server HTTP Wrapper
Avvia il server RAG in modalità HTTP (SSE) invece di stdio.

Questo permette a Claude Code e altri client di connettersi via HTTP,
evitando il timeout durante il caricamento del modello AI pesante.

Uso:
    1. Avvia questo script in un terminale separato
    2. Configura il client MCP per connettersi a http://127.0.0.1:8765/sse
"""
import sys
from pathlib import Path

# Aggiungi directory progetto al path
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

# Configura host e port PRIMA di importare mcp
# FastMCP legge da variabili ambiente o settings
import os
os.environ["FASTMCP_PORT"] = "8765"

# Importa il server MCP già configurato
from mcp_server import mcp, logger

if __name__ == "__main__":
    try:
        # Configura le settings del server
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = 8765
        
        logger.info(f"Avvio MCP server in modalità SSE su {mcp.settings.host}:{mcp.settings.port}...")
        print(f"[MCP HTTP] Server SSE in ascolto su http://{mcp.settings.host}:{mcp.settings.port}/sse")
        print(f"[MCP HTTP] Configura il client con URL: http://127.0.0.1:8765/sse")

        # Avvia server con trasporto SSE (HTTP-based)
        mcp.run(transport="sse")

    except KeyboardInterrupt:
        logger.info("Server HTTP interrotto dall'utente")
    except Exception as e:
        logger.critical(f"Errore fatale HTTP: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server HTTP terminato")
