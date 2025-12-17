"""Utility functions - Funzioni condivise"""

import logging
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configura logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("rag_system.log")
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Carica configurazione YAML"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config non trovata: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_environment():
    """Carica variabili ambiente da .env"""
    load_dotenv()


def get_api_key(key_name: str = "GOOGLE_API_KEY") -> str:
    """Ottiene API key da ambiente"""
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"{key_name} non trovata. Aggiungi in .env")
    return api_key


def ensure_directory(path: str) -> Path:
    """Crea directory se non esiste"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_supported_file_extensions() -> set:
    """Estensioni supportate (60+)"""
    return {
        # Office
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        '.odt', '.odp', '.ods',
        # Web & Markup
        '.html', '.htm', '.mhtml', '.xml', '.md', '.rst', '.tex', '.latex',
        # Code
        '.py', '.ipynb', '.js', '.ts', '.json', '.yaml', '.yml', '.toml',
        '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.swift',
        '.kt', '.scala', '.r', '.m', '.h', '.hpp', '.sh', '.bash',
        # Audio
        '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.opus', '.wma',
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg',
        '.ico', '.heic', '.heif',
        # Subtitles
        '.vtt', '.srt', '.ass', '.ssa',
        # Archives
        '.zip', '.tar', '.gz', '.rar', '.7z'
    }


def is_supported_file(file_path: str) -> bool:
    """Verifica se file è supportato"""
    ext = Path(file_path).suffix.lower()
    return ext in get_supported_file_extensions()
