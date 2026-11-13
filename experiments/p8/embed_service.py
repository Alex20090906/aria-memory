"""Tiny embedding service for the semantic label finder (E37).

Runs on the main PC (CPU) because the AI PC's research Python has no torch. The AI PC reaches it through a reverse
SSH tunnel (ssh -N -R 8095:127.0.0.1:8095 ...). Listens on 127.0.0.1 only.

    POST /embed  {"texts": [str, ...]}  ->  {"vectors": [[float, ...], ...]}   (L2-normalised)

The caller formats queries and keys itself (same strings as p8/label_finder_embed.py).
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from sentence_transformers import SentenceTransformer  # noqa: E402

PORT = int(os.environ.get("EMBED_PORT", "8095"))
ENC = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cpu")


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/embed":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        texts = body.get("texts") or []
        vecs = ENC.encode(texts, normalize_embeddings=True, batch_size=32).tolist() if texts else []
        out = json.dumps({"vectors": vecs}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"embed service on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
