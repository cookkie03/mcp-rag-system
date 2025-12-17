"""Interactive chatbot - Chatbot interattivo"""

from rich.console import Console
from utils import load_config, load_environment, get_api_key

try:
    from datapizza.clients.google import GoogleClient
    from datapizza.embedders import ClientEmbedder
    from datapizza.vectorstores.qdrant import QdrantVectorstore
except ImportError as e:
    raise ImportError(f"Installa dipendenze: pip install -r requirements.txt\nErrore: {e}")

console = Console()


class Chatbot:
    """Chatbot RAG semplificato"""

    def __init__(self, config):
        self.config = config
        api_key = get_api_key("GOOGLE_API_KEY")

        # LLM per generazione risposte
        self.llm = GoogleClient(
            api_key=api_key,
            model=config['model'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

        # Client per embedding
        self.embed_client = GoogleClient(
            api_key=api_key,
            model=config['embedding_model']
        )
        self.embedder = ClientEmbedder(client=self.embed_client)

        # Vector store
        self.vectorstore = QdrantVectorstore(
            host=str(config['vectorstore_path']),
            port=None
        )

        self.collection_name = "documents"
        self.system_prompt = """Sei un assistente AI con accesso a una knowledge base.
Rispondi basandoti sul contesto fornito. Cita sempre le fonti (documento e chunk ID).
Se il contesto non contiene informazioni rilevanti, dillo chiaramente."""

    def search(self, query, top_k=None):
        """Ricerca semantica"""
        try:
            if top_k is None:
                top_k = self.config['top_k']

            query_embedding = self.embedder.embed(query)
            results = self.vectorstore.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                k=top_k
            )
            return results
        except Exception as e:
            console.print(f"[red]Errore ricerca: {e}[/red]")
            return []

    def build_context(self, results):
        """Costruisci contesto da risultati"""
        if not results:
            return "", []

        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            # Result potrebbe essere un Chunk object o dict
            if hasattr(result, 'text'):
                content = result.text
                metadata = result.metadata if hasattr(result, 'metadata') else {}
            else:
                content = result.get('text', result.get('content', ''))
                metadata = result.get('metadata', {})

            doc_name = metadata.get('source', 'Unknown')
            chunk_id = metadata.get('chunk_id', 'N/A')

            context_parts.append(f"[Fonte {i}: {doc_name} (Chunk {chunk_id})]\n{content}")
            sources.append(f"{doc_name} (Chunk {chunk_id})")

        return "\n\n".join(context_parts), sources

    def generate_response(self, query, context):
        """Genera risposta con LLM"""
        try:
            if context:
                prompt = f"""{self.system_prompt}

CONTESTO:
{context}

DOMANDA: {query}

Rispondi basandoti sul contesto."""
            else:
                prompt = f"""{self.system_prompt}

DOMANDA: {query}

Nota: Nessun contesto trovato nella knowledge base."""

            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Errore generazione: {str(e)}"

    def process_query(self, query):
        """Processa query attraverso RAG"""
        results = self.search(query)
        context, sources = self.build_context(results)
        response = self.generate_response(query, context)
        return response, sources

    def get_stats(self):
        """Statistiche vector store"""
        try:
            info = self.vectorstore.client.get_collection(self.collection_name)
            return {
                'vectors': info.points_count,
                'status': str(info.status)
            }
        except Exception as e:
            return {'error': str(e)}

    def run(self):
        """Loop interattivo"""
        console.print("[cyan]RAG Chatbot - Digita /help per comandi[/cyan]\n")

        while True:
            try:
                query = console.input("[bold cyan]> [/bold cyan]").strip()

                if not query:
                    continue

                if query.lower() in ['/exit', '/quit']:
                    console.print("[yellow]Arrivederci![/yellow]")
                    break

                elif query.lower() == '/stats':
                    stats = self.get_stats()
                    console.print(f"\n[cyan]Statistiche:[/cyan]")
                    for k, v in stats.items():
                        console.print(f"  {k}: {v}")
                    console.print()
                    continue

                elif query.lower() == '/help':
                    console.print("""
[cyan]Comandi:[/cyan]
  /exit   - Esci
  /stats  - Statistiche
  /help   - Questo messaggio
""")
                    continue

                # Processa query
                console.print("[cyan]Ricerca in corso...[/cyan]")
                response, sources = self.process_query(query)

                console.print(f"\n[bold green]Risposta:[/bold green]\n{response}")

                if sources:
                    console.print(f"\n[bold cyan]Fonti:[/bold cyan]")
                    for source in sources:
                        console.print(f"  - {source}")

                console.print()

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrotto[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Errore: {str(e)}[/red]")


if __name__ == "__main__":
    load_environment()
    config = load_config("config.yaml")
    chatbot = Chatbot(config)
    chatbot.run()
