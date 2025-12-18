"""RAG Chatbot - Local Embeddings + Gemini LLM"""

from rich.console import Console
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from utils import load_config, load_environment, get_api_key

try:
    from datapizza.clients.google import GoogleClient
except ImportError:
    raise ImportError("pip install datapizza-ai-clients-google")

console = Console()

class Chatbot:
    def __init__(self, config):
        self.config = config
        self.llm = GoogleClient(
            api_key=get_api_key("GOOGLE_API_KEY"),
            model=config['model'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
        console.print("[dim]Caricamento modello embedding...[/dim]")
        model_kwargs = {"trust_remote_code": config.get('trust_remote_code', False)}
        self.model = SentenceTransformer(config['embedding_model'], **model_kwargs)
        self.embedding_task = config.get('embedding_task_query', None)
        self.qdrant = QdrantClient(path=str(config['vectorstore_path']))

    def search(self, query, top_k=None):
        encode_kwargs = {}
        if self.embedding_task:
            encode_kwargs["task"] = self.embedding_task
        vector = self.model.encode(query, **encode_kwargs).tolist()
        results = self.qdrant.search("documents", query_vector=vector, limit=top_k or self.config['top_k'])
        return [(r.payload.get('text', ''), r.payload.get('source', '?')) for r in results]

    def ask(self, query):
        results = self.search(query)
        if not results:
            return "Nessun documento rilevante trovato.", []
        
        context = "\n\n".join([f"[{src}]: {txt}" for txt, src in results])
        prompt = f"""Rispondi basandoti sul contesto. Cita le fonti.

CONTESTO:
{context}

DOMANDA: {query}"""
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response), [s for _, s in results]

    def run(self):
        console.print("[cyan]RAG Chat - /help per comandi[/cyan]\n")
        while True:
            try:
                q = console.input("[bold]> [/bold]").strip()
                if not q: continue
                if q.lower() in ['/exit', '/quit']: break
                
                if q.lower() == '/help':
                    console.print("[dim]/exit - Esci | /stats - Statistiche[/dim]\n")
                    continue
                
                if q.lower() == '/stats':
                    try:
                        info = self.qdrant.get_collection("documents")
                        console.print(f"[dim]Vectors: {info.points_count} | Status: {info.status}[/dim]\n")
                    except Exception as e:
                        console.print(f"[red]{e}[/red]")
                    continue
                
                response, sources = self.ask(q)
                console.print(f"\n[green]{response}[/green]")
                if sources:
                    console.print(f"[dim]Fonti: {', '.join(set(sources))}[/dim]\n")
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Errore: {e}[/red]")

if __name__ == "__main__":
    load_environment()
    Chatbot(load_config("config.yaml")).run()
