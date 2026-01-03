@echo off
REM Avvia entrambi i server MCP (HTTP + HTTPS) in finestre separate

echo ========================================
echo  Avvio MCP Server - HTTP e HTTPS
echo ========================================
echo.

REM Avvia server HTTP in una nuova finestra
start "MCP Server HTTP (porta 8765)" cmd /k ".venv\Scripts\python.exe mcp_server_http.py"

REM Attendi 2 secondi
timeout /t 2 /nobreak >nul

REM Avvia server HTTPS in una nuova finestra
start "MCP Server HTTPS (porta 8766)" cmd /k ".venv\Scripts\python.exe mcp_server_https.py"

echo.
echo [OK] Server avviati in finestre separate:
echo   - HTTP:  http://127.0.0.1:8765/sse
echo   - HTTPS: https://127.0.0.1:8766/sse
echo.
echo Premi un tasto per chiudere questo messaggio...
pause >nul
