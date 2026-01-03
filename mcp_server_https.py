"""
MCP Server HTTPS Wrapper
Avvia il server RAG in modalità HTTPS (SSE) per Claude Code.

Questo wrapper usa SSL/TLS per connessioni sicure richieste da alcuni client.
Può essere eseguito CONTEMPORANEAMENTE a mcp_server_http.py - condividono lo stesso backend.

Uso:
    python mcp_server_https.py

Collegamento:
    - HTTPS: https://127.0.0.1:8766/sse
"""
import sys
import os
import ssl
import threading
import time
from pathlib import Path

# Aggiungi directory progetto al path
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

# Controlla dipendenze PRIMA di importare qualsiasi cosa
try:
    import uvicorn
    import httpx
except ImportError as e:
    print(f"[ERRORE] Dipendenza mancante: {e}")
    print("[ERRORE] Installa le dipendenze: pip install uvicorn httpx")
    sys.exit(1)

# Configura porta HTTPS PRIMA di importare mcp
os.environ["FASTMCP_PORT"] = "8766"

# Importa il server MCP già configurato
from mcp_server import mcp, logger

def genera_certificati_self_signed():
    """Genera certificati SSL auto-firmati se non esistono"""
    cert_file = PROJECT_DIR / "cert.pem"
    key_file = PROJECT_DIR / "key.pem"

    if cert_file.exists() and key_file.exists():
        logger.info(f"Certificati SSL trovati: {cert_file}")
        return str(cert_file), str(key_file)

    logger.info("Generazione certificati SSL auto-firmati...")
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        import ipaddress

        # Genera chiave privata RSA 2048-bit
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Crea certificato auto-firmato valido 1 anno
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"MCP Server"),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.IPAddress(ipaddress.IPv4Address(u"127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())

        # Salva certificato pubblico
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        # Salva chiave privata (senza password)
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        logger.info(f"✓ Certificati generati: {cert_file}, {key_file}")
        return str(cert_file), str(key_file)

    except ImportError:
        logger.error("'cryptography' non installato!")
        print("\n[ERRORE] Installa la libreria: pip install cryptography")
        print("\nOppure genera manualmente i certificati con OpenSSL:")
        print("  openssl req -x509 -newkey rsa:2048 -nodes \\")
        print("    -keyout key.pem -out cert.pem -days 365 \\")
        print('    -subj "/CN=localhost"')
        return None, None
    except Exception as e:
        logger.error(f"Errore generazione certificati: {e}", exc_info=True)
        return None, None

def avvia_server_http_interno(porta_interna=9766):
    """Avvia il server HTTP interno in un thread separato"""
    try:
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = porta_interna
        logger.info(f"Avvio server HTTP interno su porta {porta_interna}...")
        mcp.run(transport="sse")
    except Exception as e:
        logger.error(f"Errore server HTTP interno: {e}", exc_info=True)


async def app_https(scope, receive, send):
    """
    App ASGI wrapper che crea un reverse proxy interno
    Riceve richieste HTTPS e le invia al server HTTP interno
    """
    if scope["type"] != "http":
        await send({"type": "http.response.start", "status": 500, "headers": []})
        await send({"type": "http.response.body", "body": b"WebSocket non supportato"})
        return

    # Invia le richieste al server HTTP interno tramite httpx
    import httpx

    try:
        # Costruisci URL interno
        path = scope.get("path", "/")
        query_string = scope.get("query_string", b"").decode()
        internal_url = f"http://127.0.0.1:9766{path}"
        if query_string:
            internal_url += f"?{query_string}"

        # Estrai il body se presente
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        # Effettua la richiesta al server interno
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=scope["method"],
                url=internal_url,
                content=body,
                headers=[(name, value) for name, value in scope.get("headers", [])],
                follow_redirects=True
            )

        # Invia la risposta
        await send({
            "type": "http.response.start",
            "status": response.status_code,
            "headers": response.headers.raw,
        })
        await send({
            "type": "http.response.body",
            "body": response.content,
        })

    except Exception as e:
        logger.error(f"Errore reverse proxy: {e}", exc_info=True)
        await send({
            "type": "http.response.start",
            "status": 500,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({
            "type": "http.response.body",
            "body": f"Errore interno: {str(e)}".encode(),
        })


if __name__ == "__main__":
    try:
        # Genera o carica certificati SSL
        cert_file, key_file = genera_certificati_self_signed()

        if not cert_file or not key_file:
            print("\n[ERRORE] Impossibile avviare server HTTPS senza certificati SSL")
            sys.exit(1)

        HTTPS_HOST = "127.0.0.1"
        HTTPS_PORT = 8766
        HTTP_INTERNO_PORT = 9766

        # Messaggio di avvio
        print("\n" + "="*70)
        print("  MCP Server RAG - HTTPS Wrapper")
        print("="*70)
        print(f"  URL: https://{HTTPS_HOST}:{HTTPS_PORT}/sse")
        print(f"  Certificato: {cert_file}")
        print()
        print("  NOTA: Il certificato è auto-firmato. Il browser mostrerà un avviso")
        print("        di sicurezza - è normale per ambienti di sviluppo locali.")
        print("="*70 + "\n")

        # Avvia server HTTP interno in un thread
        http_thread = threading.Thread(
            target=avvia_server_http_interno,
            args=(HTTP_INTERNO_PORT,),
            daemon=True
        )
        http_thread.start()

        # Attendi che il server interno sia pronto
        time.sleep(2)

        logger.info(f"Avvio server HTTPS su {HTTPS_HOST}:{HTTPS_PORT}...")

        # Avvia server HTTPS tramite uvicorn
        uvicorn.run(
            app_https,
            host=HTTPS_HOST,
            port=HTTPS_PORT,
            ssl_certfile=cert_file,
            ssl_keyfile=key_file,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("Server HTTPS interrotto dall'utente")
    except Exception as e:
        logger.critical(f"Errore fatale HTTPS: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Server HTTPS terminato")
