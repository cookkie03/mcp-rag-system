"""Document Ingestion - Local Embeddings + Qdrant"""

import os, sys, uuid, time, json, hashlib
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

from utils import load_config, load_environment, ensure_directory, is_supported_file

console = Console()

class Ingester:
    def __init__(self, config):
        self.config = config
        self.docs_path = ensure_directory(config['documents_path'])
        self.store_path = ensure_directory(config['vectorstore_path'])
        self.stats = {'processed': 0, 'chunks': 0, 'deleted': 0, 'errors': []}

        console.print("[yellow]Caricamento modello...[/yellow]")
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
        self.qdrant = QdrantClient(path=str(self.store_path))
        if "documents" not in [c.name for c in self.qdrant.get_collections().collections]:
            self.qdrant.create_collection("documents", VectorParams(size=768, distance=Distance.COSINE))

        self.registry_file = Path(self.store_path) / ".registry.json"
        self.registry = json.loads(self.registry_file.read_text()) if self.registry_file.exists() else {}

    def _hash(self, path):
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _delete_vectors(self, filename):
        try:
            self.qdrant.delete("documents", Filter(must=[FieldCondition(key="source", match=MatchValue(value=filename))]))
        except: pass

    def _ingest(self, path):
        self._delete_vectors(path.name)
        text = path.read_text(encoding='utf-8', errors='ignore').strip()
        if not text:
            return 0
        
        # Simple chunking with safety truncation
        size, overlap, max_chunk = self.config['chunk_size'], self.config['chunk_overlap'], 2000
        raw_chunks = [text[i:i+size] for i in range(0, len(text), size - overlap) if text[i:i+size].strip()]
        
        # Truncate oversized chunks
        chunks = []
        for c in raw_chunks:
            if len(c) > max_chunk:
                chunks.extend([c[i:i+max_chunk] for i in range(0, len(c), max_chunk) if c[i:i+max_chunk].strip()])
            else:
                chunks.append(c)
        
        if not chunks:
            return 0
        
        vectors = self.model.encode(chunks, show_progress_bar=False).tolist()
        points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload={'text': c, 'source': path.name}) 
                  for c, v in zip(chunks, vectors)]
        self.qdrant.upsert("documents", points)
        return len(points)

    def run(self, clean=False):
        start = time.time()
        
        if clean:
            try: self.qdrant.delete_collection("documents")
            except: pass
            self.qdrant.create_collection("documents", VectorParams(size=768, distance=Distance.COSINE))
            self.registry = {}
            console.print("[yellow]Storage pulito[/yellow]")

        # Scan files
        files = {str(p.relative_to(self.docs_path)): p 
                 for p in Path(self.docs_path).rglob("*") if p.is_file() and is_supported_file(str(p))}
        
        # Detect changes
        new, mod, deleted = [], [], [k for k in self.registry if k not in files]
        for key, path in files.items():
            h = self._hash(path)
            if key not in self.registry:
                new.append((key, path, h))
            elif self.registry[key] != h:
                mod.append((key, path, h))

        console.print(Panel(f"Nuovi: {len(new)} | Modificati: {len(mod)} | Eliminati: {len(deleted)}", title="Analisi"))
        
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
                    try:
                        n = self._ingest(path)
                        self.registry[key] = h
                        self.stats['processed'] += 1
                        self.stats['chunks'] += n
                    except Exception as e:
                        self.stats['errors'].append(f"{path.name}: {e}")
                    prog.advance(task)

        self.registry_file.write_text(json.dumps(self.registry))
        console.print(Panel(f"Chunks: {self.stats['chunks']} | Tempo: {int(time.time()-start)}s", title="Done", border_style="green"))
        
        if self.stats['errors']:
            for e in self.stats['errors']:
                console.print(f"[red]{e}[/red]")

if __name__ == "__main__":
    load_environment()
    Ingester(load_config("config.yaml")).run(clean="--clean" in sys.argv)
