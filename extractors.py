"""Estrattori di testo per formati non-testuali (PDF, Audio)"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Estensioni speciali (richiedono estrazione)
PDF_EXTENSIONS = {'.pdf'}
AUDIO_EXTENSIONS = {'.mp3', '.m4a', '.wav', '.ogg', '.flac'}


def extract_pdf_text(file_path: Path) -> str:
    """Estrae testo da PDF usando PyMuPDF"""
    import fitz  # pymupdf
    
    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    
    return "\n\n".join(text_parts).strip()


def extract_audio_text(file_path: Path, model_name: str = "base") -> str:
    """Trascrive audio usando Whisper (richiede FFmpeg)"""
    import whisper
    
    logger.info(f"Caricamento modello Whisper '{model_name}'...")
    model = whisper.load_model(model_name)
    
    logger.info(f"Trascrizione {file_path.name}...")
    result = model.transcribe(str(file_path))
    
    return result["text"].strip()


def extract_text(file_path: Path) -> tuple[str, str | None]:
    """Estrae testo da qualsiasi formato supportato.
    
    Args:
        file_path: Path del file da processare
        
    Returns:
        (testo, errore) - se errore è None, l'estrazione è riuscita
    """
    ext = file_path.suffix.lower()
    
    try:
        if ext in PDF_EXTENSIONS:
            return extract_pdf_text(file_path), None
        elif ext in AUDIO_EXTENSIONS:
            return extract_audio_text(file_path), None
        else:
            # File di testo normale
            return file_path.read_text(encoding='utf-8', errors='ignore').strip(), None
    except ImportError as e:
        return "", f"Dipendenza mancante: {e}"
    except Exception as e:
        logger.error(f"Errore estrazione {file_path}: {e}")
        return "", str(e)
