"""
Document Ingestion - Local Embeddings + Qdrant (Docker HTTP)
Indicizza documenti testuali in un database vettoriale per ricerca semantica.
"""

import sys, uuid, time, json, hashlib
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

# Directory per il registry locale (traccia file già indicizzati)
REGISTRY_DIR = Path(__file__).parent / ".ingest_cache"


class Ingester:
    def __init__(self, config):
        self.config = config
        self.docs_path = ensure_directory(config['documents_path'])
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
        
        # Connessione Qdrant HTTP (Docker)
        self.qdrant = QdrantClient(
            host=config.get('qdrant_host', 'localhost'),
            port=config.get('qdrant_port', 6333)
        )
        
        # Crea collection se non esiste
        if self.collection_name not in [c.name for c in self.qdrant.get_collections().collections]:
            self.qdrant.create_collection(self.collection_name, VectorParams(size=self.embedding_dim, distance=Distance.COSINE))
        
        # File registry per tracciare i file già indicizzati
        REGISTRY_DIR.mkdir(exist_ok=True)
        self.registry_file = REGISTRY_DIR / "registry.json"
        self.registry = json.loads(self.registry_file.read_text()) if self.registry_file.exists() else {}

    def _hash(self, path):
        """Calcola hash MD5 del file"""
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(self.hash_buffer_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _delete_vectors(self, source_path):
        """Elimina vettori di un file dal database usando source_path"""
        try:
            self.qdrant.delete(
                self.collection_name, 
                points_selector=Filter(must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))])
            )
        except: 
            pass

    def _chunk_text(self, text, size=None, overlap=None):
        """
        Divide il testo in chunks semantici rispettando i confini delle frasi.
        
        Returns:
            Lista di tuple: (chunk_text, char_start, char_end)
        
        Features:
        - Protezione abbreviazioni scientifiche (Dr., Fig., et al., i.e., etc.)
        - Protezione URL, email, numeri decimali
        - Fallback gerarchico per frasi troppo lunghe (stile LangChain)
        - Overlap semantico a livello di frasi
        - Tracciabilità posizione caratteri
        """
        import re
        
        size = size or self.config.get('chunk_size', 1024)
        overlap = overlap or self.config.get('chunk_overlap', 200)
        mode = self.config.get('chunking_mode', 'sentence')
        
        # === MODALITÀ LEGACY ===
        if mode == 'character':
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    # Trova posizione reale nel testo originale
                    real_start = text.find(chunk, start)
                    real_end = real_start + len(chunk)
                    chunks.append((chunk, real_start, real_end))
                start += size - overlap
            return chunks
        
        # === PROTEZIONE PATTERN SENSIBILI ===
        # Placeholder unico per evitare conflitti
        DOT = '\x00DOT\x00'
        
        protected = text
        
        # 1. Proteggi URL e email (prima di tutto)
        protected = re.sub(r'https?://[^\s]+', lambda m: m.group(0).replace('.', DOT), protected)
        protected = re.sub(r'[\w.-]+@[\w.-]+\.\w+', lambda m: m.group(0).replace('.', DOT), protected)
        
        # 2. Proteggi numeri decimali (0.05, 3.14, -2.5)
        protected = re.sub(r'(\d)\.(\d)', rf'\1{DOT}\2', protected)
        
        # 3. Proteggi abbreviazioni titoli
        protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Dott|Ing|Avv|Sig|Sig\.ra)\.', rf'\1{DOT}', protected, flags=re.IGNORECASE)
        
        # 4. Proteggi abbreviazioni scientifiche
        protected = re.sub(r'\b(Fig|Tab|Eq|Vol|No|Ch|Sec|App|Ref|Rev)\.', rf'\1{DOT}', protected, flags=re.IGNORECASE)
        
        # 5. Proteggi abbreviazioni latine/accademiche
        protected = re.sub(r'\b(vs|etc|al|Jr|Sr|Inc|Ltd|Corp|Co)\.', rf'\1{DOT}', protected, flags=re.IGNORECASE)
        protected = re.sub(r'\b(i\.e|e\.g|et al|cf|ibid|op\.cit|viz|approx|ca)\.?', 
                          lambda m: m.group(0).replace('.', DOT), protected, flags=re.IGNORECASE)
        
        # 6. Proteggi numerazioni (1. 2. 3.) solo a inizio riga o dopo newline
        protected = re.sub(r'(^|\n)(\d+)\.(\s)', rf'\1\2{DOT}\3', protected)
        
        # 7. Proteggi acronimi con punti (U.S.A., Ph.D., M.Sc.)
        protected = re.sub(r'\b([A-Z]\.){2,}', lambda m: m.group(0).replace('.', DOT), protected)
        
        # === SPLIT IN FRASI ===
        # Split su: . ! ? seguiti da spazio, oppure doppio newline
        sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', protected)
        
        # Ripristina i punti e pulisci
        sentences = [s.strip().replace(DOT, '.') for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # === FUNZIONE HELPER: SPLIT GERARCHICO (fallback per frasi lunghe) ===
        def split_long_segment(segment, max_size):
            """Split gerarchico stile LangChain: ; -> , -> spazi -> caratteri"""
            if len(segment) <= max_size:
                return [segment]
            
            result = []
            
            # Livello 1: Prova a splittare su punto e virgola
            parts = re.split(r'(?<=[;])\s*', segment)
            if len(parts) > 1:
                for part in parts:
                    result.extend(split_long_segment(part.strip(), max_size))
                return [r for r in result if r]
            
            # Livello 2: Prova a splittare su virgola
            parts = re.split(r'(?<=[,])\s*', segment)
            if len(parts) > 1:
                current = ""
                for part in parts:
                    if len(current) + len(part) + 1 <= max_size:
                        current = (current + " " + part).strip() if current else part
                    else:
                        if current:
                            result.append(current)
                        current = part
                if current:
                    result.append(current)
                return [r for r in result if r]
            
            # Livello 3: Split su spazi (parole)
            words = segment.split()
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= max_size:
                    current = (current + " " + word).strip() if current else word
                else:
                    if current:
                        result.append(current)
                    current = word
            if current:
                result.append(current)
            
            # Livello 4: Se ancora troppo lungo, split brutale a caratteri
            final_result = []
            for r in result:
                if len(r) > max_size:
                    for i in range(0, len(r), max_size):
                        final_result.append(r[i:i+max_size])
                else:
                    final_result.append(r)
            
            return [r for r in final_result if r]
        
        # === AGGREGAZIONE CHUNKS CON POSIZIONI ===
        chunks = []  # Lista di (chunk_text, char_start, char_end)
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            sent_len = len(sentence)
            
            # Frase troppo lunga: applica fallback gerarchico
            if sent_len > size:
                # Prima salva il chunk corrente
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    # Trova posizione nel testo originale
                    start_pos = text.find(current_chunk[0])
                    end_pos = text.find(current_chunk[-1]) + len(current_chunk[-1])
                    chunks.append((chunk_text, start_pos, end_pos))
                    current_chunk = []
                    current_len = 0
                
                # Split gerarchico della frase lunga
                sub_parts = split_long_segment(sentence, size)
                for part in sub_parts:
                    start_pos = text.find(part)
                    end_pos = start_pos + len(part) if start_pos >= 0 else -1
                    chunks.append((part, start_pos, end_pos))
                continue
            
            # Se aggiungere la frase supera size, chiudi chunk
            if current_len + sent_len + 1 > size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                start_pos = text.find(current_chunk[0])
                end_pos = text.find(current_chunk[-1]) + len(current_chunk[-1])
                chunks.append((chunk_text, start_pos, end_pos))
                
                # Overlap semantico: mantieni ultime frasi che rientrano in 'overlap'
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) + 1 <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_len = overlap_len
            
            current_chunk.append(sentence)
            current_len += sent_len + 1
        
        # Ultimo chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            start_pos = text.find(current_chunk[0])
            end_pos = text.find(current_chunk[-1]) + len(current_chunk[-1])
            chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks

    def _ingest(self, path):
        """Processa un singolo file"""
        if path.suffix.lower() not in self.all_extensions:
            return 0, f"Formato non supportato: {path.suffix}"
        
        # Calcola source_path subito per usarlo in delete e payload
        try:
            source_path = str(path.relative_to(self.docs_path))
        except ValueError:
            source_path = path.name
        
        # Elimina vecchi vettori di questo file (se esistono)
        self._delete_vectors(source_path)
        
        text, err = extract_text(path, self.config)
        if err:
            return 0, f"Errore estrazione: {err}"
        
        if not text or len(text) < self.min_text_length:
            return 0, "File troppo corto"
        
        chunks = self._chunk_text(text)  # Lista di (text, start, end)
        if not chunks:
            return 0, "Nessun chunk generato"
        
        # Estrai solo i testi per l'embedding
        chunk_texts = [c[0] for c in chunks]
        
        try:
            encode_kwargs = {"show_progress_bar": False}
            if self.embedding_task:
                encode_kwargs["task"] = self.embedding_task
            raw_embeddings = self.model.encode(chunk_texts, **encode_kwargs)
            
            vectors = []
            for emb in raw_embeddings:
                if hasattr(emb, 'tolist'):
                    vectors.append(emb.tolist())
                else:
                    vectors.append(list(emb))
        except Exception as e:
            return 0, f"Errore embedding: {e}"
        
        if len(vectors) != len(chunks):
            return 0, f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
        
        points = []
        for i, (chunk_data, vector) in enumerate(zip(chunks, vectors)):
            if len(vector) != self.embedding_dim:
                continue
            
            chunk_text, char_start, char_end = chunk_data
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    'text': chunk_text,
                    'source': path.name,
                    'source_path': source_path,
                    'chunk_id': i,
                    'char_start': char_start,
                    'char_end': char_end
                }
            ))
        
        if not points:
            return 0, "Nessun punto valido"
        
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
            if deleted:
                task = prog.add_task("[red]Eliminazione...", total=len(deleted))
                for key in deleted:
                    self._delete_vectors(key)  # key è source_path completo
                    del self.registry[key]
                    self.stats['deleted'] += 1
                    prog.advance(task)

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
