"""LLM Locale - Wrapper Ollama per generazione testo"""

import logging
from utils import setup_logging

logger = setup_logging("rag.llm")


class LocalLLM:
    """Wrapper per LLM locale via Ollama"""

    def __init__(self, config: dict):
        llm_config = config.get('llm', {})
        self.model = llm_config.get('model', 'llama3.1:8b')
        self.base_url = llm_config.get('base_url', 'http://localhost:11434')
        self.temperature = llm_config.get('temperature', 0.3)
        self.max_tokens = llm_config.get('max_tokens', 4096)
        self.system_prompt = llm_config.get('system_prompt', '')

        try:
            import ollama
            self.client = ollama.Client(host=self.base_url)
            # Verifica connessione
            self.client.list()
            logger.info(f"Connesso a Ollama: {self.base_url} (modello: {self.model})")
        except ImportError:
            raise ImportError("pip install ollama")
        except Exception as e:
            raise ConnectionError(f"Ollama non raggiungibile su {self.base_url}: {e}")

    def generate(self, prompt: str, system_prompt: str = None, context: str = None) -> str:
        """Genera risposta con contesto opzionale (RAG)"""
        sys_prompt = system_prompt or self.system_prompt

        full_prompt = prompt
        if context:
            full_prompt = f"CONTESTO:\n{context}\n\nDOMANDA: {prompt}"

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": full_prompt})

        return self._chat(messages)

    def chat(self, messages: list[dict]) -> str:
        """Chat multi-turn con lista messaggi [{'role': ..., 'content': ...}]"""
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(messages)
        return self._chat(full_messages)

    def _chat(self, messages: list[dict]) -> str:
        """Chiamata interna a Ollama"""
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Errore generazione LLM: {e}")
            raise

    def route_intent(self, query: str) -> str:
        """Analizza il prompt e decide: 'rag', 'memory', 'both', 'markdown', 'general'.

        Il LLM locale decide autonomamente quale sistema usare.
        """
        routing_prompt = f"""Analizza questa richiesta utente e rispondi con UNA SOLA parola tra:
- rag: se l'utente chiede informazioni su documenti, contenuti, dati caricati
- memory: se l'utente chiede di ricordare qualcosa, preferenze, conversazioni passate
- both: se serve sia cercare nei documenti sia usare la memoria
- markdown: se l'utente chiede di creare, scrivere o modificare un file
- general: per domande generiche che non richiedono documenti o memoria

Richiesta: {query}

Rispondi con UNA SOLA parola:"""

        try:
            response = self._chat([{"role": "user", "content": routing_prompt}])
            intent = response.strip().lower().split()[0]
            if intent in ('rag', 'memory', 'both', 'markdown', 'general'):
                return intent
            return 'both'  # Default sicuro
        except Exception:
            return 'both'
