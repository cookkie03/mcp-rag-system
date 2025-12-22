"""
Document Ingestion - Local Embeddings + Qdrant
Indicizza documenti testuali in un database vettoriale per ricerca semantica.
"""

import os, sys, uuid, time, json, hashlib
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

from utils import load_config, load_environment, ensure_directory, get_supported_extensions
from extractors import extract_text

console = Console()


class Ingester:
    def __init__(self, config):
        self.config = config
        self.docs_path = ensure_directory(config['documents_path'])
        self.store_path = ensure_directory(config['vectorstore_path'])
        self.stats = {'processed': 0, 'chunks': 0, 'deleted': 0, 'skipped': 0, 'errors': []}

        # Parametri da config
        self.collection_name = config.get('qdrant_collection', 'documents')
        self.min_text_length = config.get('min_text_length', 50)
        self.hash_buffer_size = config.get('hash_buffer_size', 8192)
        self.all_extensions = get_supported_extensions(config)

        console.print("[yellow]Caricamento modello embedding...[/yellow]")
        model_kwargs = {"trust_remote_code": config.get('trust_remote_code', False)}
        self.model = SentenceTransformer(config['embedding_model'], **model_kwargs)
        self.embedding_dim = config.get('embedding_dimension', 768)
        self.embedding_task = config.get('embedding_task_passage', None)
        
        # Qdrant connection with lock handling
        self.qdrant = self._connect_qdrant()
        if self.collection_name not in [c.name for c in self.qdrant.get_collections().collections]:
            self.qdrant.create_collection(self.collection_name, VectorParams(size=self.embedding_dim, distance=Distance.COSINE))
        
        # File registry per tracciare i file già indicizzati
        self.registry_file = Path(self.store_path) / ".registry.json"
        self.registry = json.loads(self.registry_file.read_text()) if self.registry_file.exists() else {}

    def _connect_qdrant(self):
        """Connect to Qdrant with automatic lock handling."""
        lock_file = Path(self.store_path) / ".lock"
        
        # First attempt
        try:
            return QdrantClient(path=str(self.store_path))
        except RuntimeError as e:
            if "already accessed" not in str(e).lower():
                raise  # Re-raise if it's a different error
            
            console.print("[yellow]Database locked, attempting to remove lock...[/yellow]")
            
            # Try removing the lock file
            try:
                if lock_file.exists():
                    lock_file.unlink()
                    console.print("[green]Lock file removed, retrying connection...[/green]")
            except PermissionError:
                pass  # Lock file is held by another process
            
            # Second attempt after removing lock
            try:
                return QdrantClient(path=str(self.store_path))
            except RuntimeError as e2:
                console.print(f"\n[bold red]Error:[/bold red] {e2}")
                console.print("\n[bold yellow]The Qdrant database is locked by another process (likely the MCP server).[/bold yellow]")
                console.print("[cyan]To fix this issue:[/cyan]")
                console.print("  1. Run [bold]python process_manager.py[/bold] to view and terminate blocking processes")
                console.print("  2. Restart your IDE (Antigravity/VS Code) to release the lock")
                console.print("  3. Manually delete the lock file: [dim]qdrant_storage/.lock[/dim]")
                console.print("  4. Then run this script again\n")
                sys.exit(1)

    def _hash(self, path):
        """Calcola hash MD5 del file"""
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(self.hash_buffer_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _delete_vectors(self, filename):
        """Elimina vettori di un file dal database"""
        try:
            self.qdrant.delete(self.collection_name, Filter(must=[FieldCondition(key="source", match=MatchValue(value=filename))]))
        except: 
            pass

    def _chunk_text(self, text, size=None, overlap=None):
        """Divide il testo in chunks con overlap"""
        size = size or self.config.get('chunk_size', 1024)
        overlap = overlap or self.config.get('chunk_overlap', 200)
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += size - overlap
        return chunks

    def _ingest(self, path):
        """Processa un singolo file"""
        # Verifica formato supportato
        if path.suffix.lower() not in self.all_extensions:
            return 0, f"Formato non supportato: {path.suffix}"
        
        # Elimina vecchi vettori
        self._delete_vectors(path.name)
        
        # Estrai testo (gestisce PDF, audio e testo normale)
        text, err = extract_text(path, self.config)
        if err:
            return 0, f"Errore estrazione: {err}"
        
        if not text or len(text) < self.min_text_length:
            return 0, "File troppo corto"
        
        # Chunking
        chunks = self._chunk_text(text)
        if not chunks:
            return 0, "Nessun chunk generato"
        
        # Embedding
        try:
            encode_kwargs = {"show_progress_bar": False}
            if self.embedding_task:
                encode_kwargs["task"] = self.embedding_task
            raw_embeddings = self.model.encode(chunks, **encode_kwargs)
            
            # Converti in liste Python pure (niente numpy)
            vectors = []
            for emb in raw_embeddings:
                if hasattr(emb, 'tolist'):
                    vectors.append(emb.tolist())
                else:
                    vectors.append(list(emb))
        except Exception as e:
            return 0, f"Errore embedding: {e}"
        
        # Verifica dimensioni
        if len(vectors) != len(chunks):
            return 0, f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
        
        # Crea punti
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            if len(vector) != self.embedding_dim:
                continue  # Skip vettori malformati
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={'text': chunk, 'source': path.name, 'chunk_id': i}
            ))
        
        if not points:
            return 0, "Nessun punto valido"
        
        # Upsert
        try:
            self.qdrant.upsert(self.collection_name, points)
        except Exception as e:
            return 0, f"Errore upsert: {e}"
        
        return len(points), None

    def run(self, clean=False):
        start = time.time()
        
        if clean:
            try: 
                self.qdrant.delete_collection(self.collection_name)
            except: 
                pass
            self.qdrant.create_collection(self.collection_name, VectorParams(size=self.embedding_dim, distance=Distance.COSINE))
            self.registry = {}
            console.print("[yellow]Storage pulito[/yellow]")

        # Scan files
        files = {}
        for p in Path(self.docs_path).rglob("*"):
            if p.is_file() and p.suffix.lower() in self.all_extensions:
                key = str(p.relative_to(self.docs_path))
                files[key] = p
        
        # Detect changes
        new, mod, deleted = [], [], []
        for k in self.registry:
            if k not in files:
                deleted.append(k)
        for key, path in files.items():
            h = self._hash(path)
            if key not in self.registry:
                new.append((key, path, h))
            elif self.registry[key] != h:
                mod.append((key, path, h))

        console.print(Panel(f"Trovati: {len(files)} file | Nuovi: {len(new)} | Mod: {len(mod)} | Del: {len(deleted)}", title="Analisi"))
        
        if not (new or mod or deleted):
            console.print("[green]Tutto aggiornato![/green]")
            return

        with Progress() as prog:
            # Delete
            if deleted:
                task = prog.add_task("[red]Eliminazione...", total=len(deleted))
                for key in deleted:
                    self._delete_vectors(Path(key).name)
                    del self.registry[key]
                    self.stats['deleted'] += 1
                    prog.advance(task)

            # Ingest
            to_do = new + mod
            if to_do:
                task = prog.add_task("[green]Ingestione...", total=len(to_do))
                for key, path, h in to_do:
                    n, err = self._ingest(path)
                    if err:
                        self.stats['errors'].append(f"{path.name}: {err}")
                        self.stats['skipped'] += 1
                    else:
                        self.registry[key] = h
                        self.stats['processed'] += 1
                        self.stats['chunks'] += n
                    prog.advance(task)

        self.registry_file.write_text(json.dumps(self.registry))
        
        elapsed = int(time.time() - start)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        tempo_str = f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")
        console.print(Panel(
            f"Processati: {self.stats['processed']} | Chunks: {self.stats['chunks']} | Skipped: {self.stats['skipped']} | Tempo: {tempo_str}",
            title="Completato", border_style="green"
        ))
        
        if self.stats['errors']:
            console.print("\n[bold red]Errori:[/bold red]")
            for e in self.stats['errors']:
                console.print(f"  • {e}")


if __name__ == "__main__":
    load_environment()
    Ingester(load_config("config.yaml")).run(clean="--clean" in sys.argv)
