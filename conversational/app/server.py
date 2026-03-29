"""Local dev server that injects OpenAI API key from .env into the app."""

import http.server
import os
from pathlib import Path

PORT = 8080
ENV_PATH = Path(__file__).parent.parent.parent / ".env"


def load_api_key():
    if not ENV_PATH.exists():
        return None
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith("openai_key"):
                return line.split("=", 1)[1].strip()
    return None


API_KEY = load_api_key()


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/key":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            import json
            self.wfile.write(json.dumps({"key": API_KEY or ""}).encode())
            return
        return super().do_GET()


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    print(f"API key: {'loaded' if API_KEY else 'NOT FOUND'}")
    print(f"Server: http://localhost:{PORT}")
    http.server.HTTPServer(("", PORT), Handler).serve_forever()
