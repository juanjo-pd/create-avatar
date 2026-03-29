"""Server that injects OpenAI API key and serves the conversational app.

Reads API key from:
1. OPENAI_API_KEY env var (Railway/production)
2. .env file (local dev)
"""

import http.server
import json
import os
from pathlib import Path

PORT = int(os.environ.get("PORT", 8080))


def load_api_key():
    # 1. Environment variable (Railway)
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    # 2. .env file (local dev)
    for env_path in [
        Path(__file__).parent.parent.parent / ".env",
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
    ]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and "openai" in line.lower():
                        return line.split("=", 1)[1].strip()
    return None


API_KEY = load_api_key()

# Serve from the app directory
APP_DIR = Path(__file__).parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/key":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"key": API_KEY or ""}).encode())
            return
        return super().do_GET()

    def log_message(self, format, *args):
        if "/api/" in str(args[0]) if args else False:
            return  # Don't log API key requests
        super().log_message(format, *args)


if __name__ == "__main__":
    print(f"API key: {'loaded' if API_KEY else 'NOT FOUND - set OPENAI_API_KEY'}")
    print(f"Server: http://localhost:{PORT}")
    http.server.HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
