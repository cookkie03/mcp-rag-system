"""Document ingestion - Ingestione documenti"""

import os
import sys
import uuid
import time
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel

from utils import load_config, load_environment, get_api_key, ensure_directory, is_supported_file

try:
    from datapizza.modules.splitters import TextSplitter
    from datapizza.embedders import ClientEmbedder
    from datapizza.clients.google import GoogleClient
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
except ImportError as e:
    raise ImportError(f"Installa dipendenze: pip install -r requirements.txt\nErrore: {e}")

console = Console()


class Ingester:
    """Ingestione documenti semplificata"""

    def __init__(self, config):
        self.config = config
        self.docs_path = ensure_directory(config['documents_path'])
        self.store_path = ensure_directory(config['vectorstore_path'])

        api_key = get_api_key("GOOGLE_API_KEY")

        # Client Google per embedding
        self.google_client = GoogleClient(
            api_key=api_key,
            model=config['embedding_model']
        )
        self.embedder = ClientEmbedder(client=self.google_client)

        # Vector store (Qdrant locale su disco)
        self.qdrant_client = QdrantClient(path=str(self.store_path))

        # Splitter
        self.splitter = TextSplitter(
            max_char=config['chunk_size'],
            overlap=config['chunk_overlap']
        )

        self.collection_name = "documents"
        self.embedding_dim = config.get('embedding_dimensions', 768)
        self.stats = {'processed': 0, 'failed': 0, 'chunks': 0, 'errors': []}

        # Crea collection se non esiste
        self._ensure_collection()

    def _ensure_collection(self):
        """Assicura che la collection esista"""
        collections = self.qdrant_client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )

    def clean(self):
        """Pulisce il database prima di re-ingestire"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._ensure_collection()
            console.print(Panel("[yellow]Database pulito[/yellow]", title="Clean"))
            return True
        except Exception:
            return True

    def get_files(self):
        """Trova file supportati"""
        files = []
        for root, dirs, filenames in os.walk(self.docs_path):
            for filename in filenames:
                file_path = Path(root) / filename
                if is_supported_file(str(file_path)):
                    files.append(file_path)
        return sorted(files)

    def embed_with_retry(self, text, max_retries=3):
        """Embed con retry automatico per errori di rete"""
        for attempt in range(max_retries):
            try:
                return self.embedder.embed(text)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)  # 2, 4, 8 secondi
                    time.sleep(wait_time)
                else:
                    raise e

    def ingest_file(self, file_path):
        """Ingestisci un file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return True, f"{file_path.name} (vuoto, saltato)"

            text_chunks = self.splitter.split(content)
            points = []

            for i, text in enumerate(text_chunks):
                embedding_vector = self.embed_with_retry(text)
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding_vector,
                    payload={
                        'text': text,
                        'source': file_path.name,
                        'chunk_id': str(i),
                        'path': str(file_path)
                    }
                )
                points.append(point)

            # Batch upsert
            if points:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

            self.stats['processed'] += 1
            self.stats['chunks'] += len(text_chunks)
            return True, f"{file_path.name} ({len(text_chunks)} chunks)"

        except Exception as e:
            self.stats['failed'] += 1
            error_msg = f"{file_path.name}: {str(e)}"
            self.stats['errors'].append(error_msg)
            return False, error_msg

    def run(self, clean=False):
        """Esegui ingestione"""
        if clean:
            self.clean()

        files = self.get_files()

        if not files:
            console.print(Panel("[yellow]Nessun documento trovato in ./data/[/yellow]", title="Info"))
            return

        console.print(Panel(f"[cyan]Trovati {len(files)} documenti[/cyan]", title="Ingestione"))

        with Progress() as progress:
            task = progress.add_task("[cyan]Elaborazione...", total=len(files))
            for file_path in files:
                success, msg = self.ingest_file(file_path)
                status = "[green]OK[/green]" if success else "[red]ERR[/red]"
                progress.update(task, advance=1, description=f"{status} {msg}")

        # Statistiche a schermo
        table = Table(title="Risultati", show_header=True)
        table.add_column("Metrica", style="cyan")
        table.add_column("Valore", style="green")
        table.add_row("Elaborati", str(self.stats['processed']))
        table.add_row("Falliti", str(self.stats['failed']))
        table.add_row("Chunk totali", str(self.stats['chunks']))
        console.print(table)

        # Scrittura Log Errori su File
        if self.stats['errors']:
            log_file = "ingestion_errors.log"
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"=== Report Errori Ingestione - {datetime.now()} ===\n\n")
                    f.write(f"Totale File Falliti: {self.stats['failed']}\n")
                    f.write("-" * 50 + "\n\n")
                    for error in self.stats['errors']:
                        f.write(f"{error}\n")
                        f.write("-" * 30 + "\n")
                
                console.print(f"\n[red]Rilevati {len(self.stats['errors'])} errori. Dettagli salvati in: {log_file}[/red]")
            except Exception as e:
                console.print(f"[red]Impossibile scrivere file di log: {e}[/red]")

        console.print(Panel(f"[green]Completato: {datetime.now().strftime('%H:%M:%S')}[/green]", title="Status"))


if __name__ == "__main__":
    load_environment()
    config = load_config("config.yaml")

    # Check --clean flag
    clean = "--clean" in sys.argv

    ingester = Ingester(config)
    ingester.run(clean=clean)

