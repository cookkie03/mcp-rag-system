"""Document Ingestion - Nuovo SDK google-genai + qdrant_client"""

import os
import sys
import uuid
import time
import json
import hashlib
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel

# Nuovo SDK Google GenAI
from google import genai

from utils import load_config, load_environment, get_api_key, ensure_directory, is_supported_file

try:
    from datapizza.modules.splitters import TextSplitter  
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
except ImportError as e:
    raise ImportError(f"Installa dipendenze: pip install -r requirements.txt\nErrore: {e}")

console = Console()


class Ingester:
    def __init__(self, config):
        self.config = config
        self.docs_path = ensure_directory(config['documents_path'])
        self.store_path = ensure_directory(config['vectorstore_path'])
        self.collection_name = "documents"
        self.embedding_dim = config.get('embedding_dimensions', 768)
        self.stats = {'processed': 0, 'failed': 0, 'chunks': 0, 'skipped': 0, 'updated': 0, 'errors': []}

        # Nuovo client Google GenAI
        api_key = get_api_key("GOOGLE_API_KEY")
        self.genai_client = genai.Client(api_key=api_key)

        # Splitter datapizza
        self.splitter = TextSplitter(
            max_char=config['chunk_size'],
            overlap=config['chunk_overlap']
        )

        # Qdrant diretto
        self.qdrant = QdrantClient(path=str(self.store_path))
        self._ensure_collection()

        # Registry
        self.tracking_file = Path(self.store_path) / ".ingested_files.json"
        self.registry = self._load_registry()

    def _ensure_collection(self):
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )

    def _load_registry(self):
        if self.tracking_file.exists():
            try:
                return json.loads(self.tracking_file.read_text(encoding='utf-8'))
            except:
                return {}
        return {}

    def _save_registry(self):
        try:
            self.tracking_file.write_text(json.dumps(self.registry, indent=2, ensure_ascii=False), encoding='utf-8')
        except:
            pass

    def _compute_file_hash(self, file_path):
        h = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except:
            return None

    def _get_file_key(self, file_path):
        try:
            return str(file_path.relative_to(self.docs_path))
        except:
            return str(file_path)

    def _register_file(self, file_path, file_hash):
        self.registry[self._get_file_key(file_path)] = {
            'hash': file_hash,
            'mtime': file_path.stat().st_mtime,
            'ingested_at': datetime.now().isoformat()
        }

    def _check_file_status(self, file_path):
        key = self._get_file_key(file_path)
        h = self._compute_file_hash(file_path)
        if key not in self.registry:
            return 'new', h
        if self.registry[key].get('hash') == h:
            return 'unchanged', h
        return 'modified', h

    def clean(self):
        try:
            self.qdrant.delete_collection(self.collection_name)
        except:
            pass
        self._ensure_collection()
        self.registry = {}
        self._save_registry()
        console.print(Panel("[yellow]Database pulito[/yellow]", title="Clean"))

    def get_files(self):
        files = []
        for root, _, names in os.walk(self.docs_path):
            for name in names:
                p = Path(root) / name
                if is_supported_file(str(p)):
                    files.append(p)
        return sorted(files)

    def embed(self, text, retries=5):
        """Embed singolo testo"""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts, retries=5):
        """Embed batch di testi (max 250) con rate limiting intelligente"""
        import re
        
        for attempt in range(retries):
            try:
                response = self.genai_client.models.embed_content(
                    model='gemini-embedding-001',
                    contents=texts  # Lista di testi
                )
                # Converti tutti gli embedding a liste Python pure
                return [[float(x) for x in emb.values] for emb in response.embeddings]
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                    match = re.search(r'retryDelay.*?(\d+)', error_str)
                    wait_time = int(match.group(1)) + 5 if match else 65
                    console.print(f"[yellow]Rate limit, attendo {wait_time}s...[/yellow]")
                    time.sleep(wait_time)
                elif attempt == retries - 1:
                    raise e
                else:
                    time.sleep(2 ** (attempt + 1))

    def ingest_file(self, file_path):
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            if not content.strip():
                return True, f"{file_path.name} (vuoto)"

            # Split con datapizza (restituisce Chunk objects)
            chunk_objects = self.splitter.split(content)
            if not chunk_objects:
                return True, f"{file_path.name} (no chunks)"

            # Estrai testo dai Chunk
            texts = [c.text if hasattr(c, 'text') else str(c) for c in chunk_objects]

            # Dividi chunk lunghi (nessuna perdita dati)
            MAX = 6000
            chunks = []
            for t in texts:
                if len(t) > MAX:
                    for i in range(0, len(t), MAX):
                        s = t[i:i+MAX].strip()
                        if s:
                            chunks.append(s)
                elif t.strip():
                    chunks.append(t)

            if not chunks:
                return True, f"{file_path.name} (empty after split)"

            # Embed in BATCH (max 100 per batch per sicurezza, limite è 250)
            # Con ~10 req/min, aspettiamo 7s tra batch
            BATCH_SIZE = 100
            all_vectors = []
            
            for batch_start in range(0, len(chunks), BATCH_SIZE):
                batch_texts = chunks[batch_start:batch_start + BATCH_SIZE]
                batch_vectors = self.embed_batch(batch_texts)
                all_vectors.extend(batch_vectors)
                
                # Rate limiting: 7s tra batch per stare sotto 10 req/min
                if batch_start + BATCH_SIZE < len(chunks):
                    console.print(f"[dim]Batch {batch_start//BATCH_SIZE + 1} completato, attendo 7s...[/dim]")
                    time.sleep(7)

            # Crea punti Qdrant
            points = []
            for i, (text, vec) in enumerate(zip(chunks, all_vectors)):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={'text': text, 'source': file_path.name, 'chunk_id': str(i), 'path': str(file_path)}
                ))

            if points:
                self.qdrant.upsert(collection_name=self.collection_name, points=points)

            self.stats['processed'] += 1
            self.stats['chunks'] += len(points)
            return True, f"{file_path.name} ({len(points)} chunks)"

        except Exception as e:
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{file_path.name}: {e}")
            return False, f"{file_path.name}: {e}"

    def run(self, clean=False):
        start = time.time()
        if clean:
            self.clean()

        files = self.get_files()
        if not files:
            console.print(Panel("[yellow]Nessun documento[/yellow]", title="Info"))
            return

        new, mod, unch = [], [], []
        info = {}
        for f in files:
            s, h = self._check_file_status(f)
            info[f] = {'hash': h}
            (new if s == 'new' else mod if s == 'modified' else unch).append(f)

        console.print(Panel(f"[cyan]{len(files)} file[/cyan]\n  Nuovi: {len(new)}\n  Mod: {len(mod)}\n  Inv: {len(unch)}", title="Analisi"))

        to_do = new + mod
        if not to_do:
            console.print(Panel("[green]Nulla da fare[/green]"))
            return

        self.stats['skipped'] = len(unch)

        with Progress() as prog:
            task = prog.add_task("[cyan]...", total=len(to_do))
            for f in to_do:
                ok, msg = self.ingest_file(f)
                if ok:
                    self._register_file(f, info[f]['hash'])
                    if f in mod:
                        self.stats['updated'] += 1
                prog.update(task, advance=1, description=f"{'[green]OK' if ok else '[red]ERR'}[/] {f.name}")

        self._save_registry()

        t = Table(title="Risultati")
        t.add_column("Metrica", style="cyan")
        t.add_column("Valore", style="green")
        t.add_row("Nuovi", str(self.stats['processed'] - self.stats['updated']))
        t.add_row("Aggiornati", str(self.stats['updated']))
        t.add_row("Saltati", str(self.stats['skipped']))
        t.add_row("Falliti", str(self.stats['failed']))
        t.add_row("Chunks", str(self.stats['chunks']))
        console.print(t)

        if self.stats['errors']:
            console.print(f"\n[bold red]═══ {len(self.stats['errors'])} ERRORI ═══[/bold red]")
            for err in self.stats['errors']:
                console.print(f"[red]  • {err}[/red]")

        e = int(time.time() - start)
        console.print(Panel(f"[green]{e//3600}h {(e%3600)//60}m {e%60}s[/green]", title="Tempo"))


if __name__ == "__main__":
    load_environment()
    Ingester(load_config("config.yaml")).run(clean="--clean" in sys.argv)
