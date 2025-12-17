"""Document ingestion - Ingestione documenti"""

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

from utils import load_config, load_environment, get_api_key, ensure_directory, is_supported_file

try:
    from datapizza.modules.splitters import TextSplitter
    from datapizza.embedders import ClientEmbedder
    from datapizza.clients.google import GoogleClient
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
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
        self.stats = {'processed': 0, 'failed': 0, 'chunks': 0, 'skipped': 0, 'updated': 0, 'errors': []}

        # File registry per tracking
        self.tracking_file = Path(self.store_path) / ".ingested_files.json"
        self.registry = self._load_registry()

        # Crea collection se non esiste
        self._ensure_collection()

    def _load_registry(self):
        """Carica il registry dei file già processati"""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_registry(self):
        """Salva il registry dei file processati"""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except IOError as e:
            console.print(f"[yellow]Attenzione: impossibile salvare registry: {e}[/yellow]")

    def _compute_file_hash(self, file_path):
        """Calcola hash MD5 del contenuto del file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except IOError:
            return None

    def _get_file_key(self, file_path):
        """Genera chiave univoca per il file (path relativo alla cartella data)"""
        try:
            return str(file_path.relative_to(self.docs_path))
        except ValueError:
            return str(file_path)

    def _register_file(self, file_path, file_hash):
        """Registra un file come processato con successo"""
        key = self._get_file_key(file_path)
        self.registry[key] = {
            'hash': file_hash,
            'mtime': file_path.stat().st_mtime,
            'ingested_at': datetime.now().isoformat(),
            'full_path': str(file_path)
        }

    def _check_file_status(self, file_path):
        """
        Controlla lo stato di un file.
        Ritorna: (status, hash) dove status è 'new', 'unchanged', 'modified'
        """
        key = self._get_file_key(file_path)
        current_hash = self._compute_file_hash(file_path)
        
        if key not in self.registry:
            return 'new', current_hash
        
        stored_hash = self.registry[key].get('hash')
        
        if current_hash == stored_hash:
            return 'unchanged', current_hash
        else:
            return 'modified', current_hash

    def _delete_file_chunks(self, file_path):
        """
        Elimina tutti i chunk associati a un file specifico da Qdrant.
        Usa il campo 'source' nel payload per identificare i chunk.
        """
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=file_path.name)
                        )
                    ]
                )
            )
            return True
        except Exception as e:
            console.print(f"[yellow]Attenzione: impossibile eliminare vecchi chunk di {file_path.name}: {e}[/yellow]")
            return False

    def _ensure_collection(self):
        """Assicura che la collection esista"""
        collections = self.qdrant_client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )

    def clean(self):
        """Pulisce il database, registry e log errori prima di re-ingestire"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._ensure_collection()
            # Reset anche il registry
            self.registry = {}
            self._save_registry()
            # Elimina anche il log degli errori
            error_log = Path(self.store_path) / "ingestion_errors.log"
            if error_log.exists():
                error_log.unlink()
            console.print(Panel("[yellow]Database, registry e log errori puliti[/yellow]", title="Clean"))
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

                # Handle nested list from datapizza (may return [[...]] or [...])
                # Flatten if needed
                if embedding_vector and isinstance(embedding_vector[0], (list, tuple)):
                    embedding_vector = embedding_vector[0]
                
                # Convert to list of native Python floats
                if hasattr(embedding_vector, 'tolist'):
                    embedding_vector = embedding_vector.tolist()
                embedding_vector = [float(x) for x in embedding_vector]
                
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
        """Esegui ingestione con supporto incrementale"""
        start_time = time.time()  # Tempo di inizio (float)
        
        if clean:
            self.clean()

        files = self.get_files()

        if not files:
            console.print(Panel("[yellow]Nessun documento trovato in ./data/[/yellow]", title="Info"))
            return

        # Classifica i file e salva gli hash
        new_files = []
        modified_files = []
        unchanged_files = []
        file_info = {}  # Dizionario per salvare hash pre-calcolati
        
        for file_path in files:
            status, file_hash = self._check_file_status(file_path)
            file_info[file_path] = {'hash': file_hash, 'status': status}
            if status == 'new':
                new_files.append(file_path)
            elif status == 'modified':
                modified_files.append(file_path)
            else:
                unchanged_files.append(file_path)

        # Report classificazione
        console.print(Panel(
            f"[cyan]Trovati {len(files)} documenti[/cyan]\n"
            f"  [green]• Nuovi: {len(new_files)}[/green]\n"
            f"  [yellow]• Modificati: {len(modified_files)}[/yellow]\n"
            f"  [dim]• Invariati (saltati): {len(unchanged_files)}[/dim]",
            title="Analisi File"
        ))

        # File da processare (nuovi + modificati)
        files_to_process = new_files + modified_files
        
        if not files_to_process:
            console.print(Panel("[green]Nessun file nuovo o modificato da processare[/green]", title="Info"))
            return

        self.stats['skipped'] = len(unchanged_files)

        with Progress() as progress:
            task = progress.add_task("[cyan]Elaborazione...", total=len(files_to_process))
            for file_path in files_to_process:
                is_modified = file_path in modified_files
                file_hash = file_info[file_path]['hash']
                
                # Se il file è stato modificato, elimina prima i vecchi chunk
                if is_modified:
                    self._delete_file_chunks(file_path)
                
                # Processa il file
                success, msg = self.ingest_file(file_path)
                
                if success:
                    # Registra solo se processato con successo
                    self._register_file(file_path, file_hash)
                    if is_modified:
                        self.stats['updated'] += 1
                        status_icon = "[yellow]UPD[/yellow]"
                    else:
                        status_icon = "[green]NEW[/green]"
                else:
                    status_icon = "[red]ERR[/red]"
                
                progress.update(task, advance=1, description=f"{status_icon} {msg}")

        # Salva registry alla fine
        self._save_registry()

        # Statistiche a schermo
        table = Table(title="Risultati", show_header=True)
        table.add_column("Metrica", style="cyan")
        table.add_column("Valore", style="green")
        table.add_row("Nuovi processati", str(self.stats['processed'] - self.stats['updated']))
        table.add_row("Aggiornati", str(self.stats['updated']))
        table.add_row("Saltati (invariati)", str(self.stats['skipped']))
        table.add_row("Falliti", str(self.stats['failed']))
        table.add_row("Chunk totali", str(self.stats['chunks']))
        console.print(table)

        # Scrittura Log Errori su File (in qdrant_storage)
        if self.stats['errors']:
            log_file = Path(self.store_path) / "ingestion_errors.log"
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

        # Calcola durata totale
        elapsed = int(time.time() - start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            duration_str = f"{hours} ore {minutes} min {seconds} sec"
        elif minutes > 0:
            duration_str = f"{minutes} min {seconds} sec"
        else:
            duration_str = f"{seconds} sec"
        
        console.print(Panel(f"[green]Completato in {duration_str}[/green]", title="Status"))


if __name__ == "__main__":
    load_environment()
    config = load_config("config.yaml")

    # Check --clean flag
    clean = "--clean" in sys.argv

    ingester = Ingester(config)
    ingester.run(clean=clean)

