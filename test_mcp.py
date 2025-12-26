"""Test MCP Server - Verifica funzionamento RAG"""
import asyncio
import sys
from pathlib import Path

# Aggiungi il path del progetto
sys.path.insert(0, str(Path(__file__).parent))

async def test_mcp_server():
    """Testa il server MCP con una query di esempio"""
    print("Test MCP RAG Server")
    print("=" * 50)

    # Importa il modulo MCP server
    from mcp_server import search_knowledge_base

    # Test 1: Query generica
    print("\nTest 1: Query 'embedding model'")
    print("-" * 50)
    result = search_knowledge_base("embedding model", limit=3)
    print(result)

    # Test 2: Query specifica
    print("\n\nTest 2: Query 'RAG system'")
    print("-" * 50)
    result = search_knowledge_base("RAG system", limit=3)
    print(result)

    # Test 3: Query che non dovrebbe trovare nulla
    print("\n\nTest 3: Query 'zxcvbnmasdfghjkl'")
    print("-" * 50)
    result = search_knowledge_base("zxcvbnmasdfghjkl", limit=3)
    print(result)

    print("\n\nTest completato!")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
