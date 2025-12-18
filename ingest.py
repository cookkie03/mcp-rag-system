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

from utils import load_config, load_environment, ensure_directory, is_supported_file
from extractors import extract_text, PDF_EXTENSIONS, AUDIO_EXTENSIONS, NOTEBOOK_EXTENSIONS, EXCEL_EXTENSIONS

console = Console()

# Estensioni supportate
TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.ts', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.csv'}
ALL_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | AUDIO_EXTENSIONS | NOTEBOOK_EXTENSIONS | EXCEL_EXTENSIONS


class Ingester:
    def __init__(self, config):
        self.config = config
        self.docs_path = ensure_directory(config['documents_path'])
        self.store_path = ensure_directory(config['vectorstore_path'])
        self.stats = {'processed': 0, 'chunks': 0, 'deleted': 0, 'skipped': 0, 'errors': []}

        console.print("[yellow]Caricamento modello embedding...[/yellow]")
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        
        self.qdrant = QdrantClient(path=str(self.store_path))
        if "documents" not in [c.name for c in self.qdrant.get_collections().collections]:
            self.qdrant.create_collection("documents", VectorParams(size=768, distance=Distance.COSINE))

        self.registry_file = Path(self.store_path) / ".registry.json"
        self.registry = json.loads(self.registry_file.read_text()) if self.registry_file.exists() else {}

    def _hash(self, path):
        """Calcola hash MD5 del file"""
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _delete_vectors(self, filename):
        """Elimina vettori di un file dal database"""
        try:
            self.qdrant.delete("documents", Filter(must=[FieldCondition(key="source", match=MatchValue(value=filename))]))
        except: 
            pass

    def _chunk_text(self, text, size=1000, overlap=200):
        """Divide il testo in chunks con overlap"""
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
        if path.suffix.lower() not in ALL_EXTENSIONS:
            return 0, f"Formato non supportato: {path.suffix}"
        
        # Elimina vecchi vettori
        self._delete_vectors(path.name)
        
        # Estrai testo (gestisce PDF, audio e testo normale)
        text, err = extract_text(path)
        if err:
            return 0, f"Errore estrazione: {err}"
        
        if not text or len(text) < 50:
            return 0, "File troppo corto"
        
        # Chunking
        chunks = self._chunk_text(text, self.config['chunk_size'], self.config['chunk_overlap'])
        if not chunks:
            return 0, "Nessun chunk generato"
        
        # Embedding
        try:
            # encode() con convert_to_numpy=False restituisce tensor, usiamo convert_to_tensor=False
            raw_embeddings = self.model.encode(chunks, show_progress_bar=False)
            
            # Converti in liste Python pure (niente numpy)
            vectors = []
            for emb in raw_embeddings:
                # Ogni emb potrebbe essere numpy.ndarray o list
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
            if len(vector) != 768:
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
            self.qdrant.upsert("documents", points)
        except Exception as e:
            return 0, f"Errore upsert: {e}"
        
        return len(points), None

    def run(self, clean=False):
        start = time.time()
        
        if clean:
            try: 
                self.qdrant.delete_collection("documents")
            except: 
                pass
            self.qdrant.create_collection("documents", VectorParams(size=768, distance=Distance.COSINE))
            self.registry = {}
            console.print("[yellow]Storage pulito[/yellow]")

        # Scan files
        files = {}
        for p in Path(self.docs_path).rglob("*"):
            if p.is_file() and p.suffix.lower() in ALL_EXTENSIONS:
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
        console.print(Panel(
            f"Processati: {self.stats['processed']} | Chunks: {self.stats['chunks']} | Skipped: {self.stats['skipped']} | Tempo: {elapsed}s",
            title="Completato", border_style="green"
        ))
        
        if self.stats['errors']:
            console.print("\n[bold red]Errori:[/bold red]")
            for e in self.stats['errors']:
                console.print(f"  • {e}")


if __name__ == "__main__":
    load_environment()
    Ingester(load_config("config.yaml")).run(clean="--clean" in sys.argv)
