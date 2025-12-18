"""Wrapper MCP - Filtra output per Antigravity"""
import sys
import subprocess
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PYTHON = SCRIPT_DIR / ".venv" / "Scripts" / "python.exe"
SERVER = SCRIPT_DIR / "mcp_server.py"

def main():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    
    proc = subprocess.Popen(
        [str(PYTHON), str(SERVER)],
        stdin=sys.stdin,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=env,
        bufsize=0
    )

    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            
            try:
                text = line.decode('utf-8', errors='ignore').strip()
            except:
                continue

            # Solo JSON passa
            if text.startswith('{'):
                sys.stdout.buffer.write(text.encode('utf-8') + b'\n')
                sys.stdout.buffer.flush()
                
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        if proc.poll() is None:
            proc.terminate()

if __name__ == "__main__":
    main()
