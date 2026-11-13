"""Memory service on the AI PC: the three-model loop (main PC) reaches the LLM server in-weights memory through this.
Local only (127.0.0.1:8094); the main PC connects over an SSH tunnel (ssh -L 8094:127.0.0.1:8094).

POST /write   {"owner": "Leo", "object": "passport", "value": "under the doormat"}      -> {"positions": n}
POST /recall  {"owner": "Leo", "object": "passport", "context": "<text before the anchor>"} -> {"text": "..."}
              context may be empty: the anchor then stands alone after the fixed preamble
POST /think   {"question": "<what the person said>"}  -> {"answer","anchor","theta","final",...} (inline recall)
POST /write2  {"e": "護照", "s": "location", "v": "書房抽屜"}  -> {"positions": n}   (label as given, E41)
POST /familiar {"e": "護照", "s": "location"}  -> {"fam": f, "units": [...]}       (familiarity gate, E41)
POST /reset   {}                                                                        -> forget everything
POST /off     {}                                                                        -> no injection
GET  /health
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import fmem_anchor as fa
import fmem_tool as f
import inline_recall as ir
import re
from colloquial_test import canon

_SETUP = {}

class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/write":
                o, obj = canon(req["owner"], req["object"])
                k = fa.write(o, obj, req["value"], dl=float(req.get("dl", 14.0)), tau_w=float(req.get("tau_w", 1e9)))
                self._send(200, {"positions": k})
            elif self.path == "/recall":
                ctx = req.get("context", "").strip()
                prompt = f.PRE + (ctx + "\n" if ctx else "") + f"[{req['owner']}|{req['object']}] ->"
                text = fa.recall(prompt, req["owner"], req["object"], max_tokens=int(req.get("max_tokens", 8)))
                fa.off()
                self._send(200, {"text": text.strip().split("\n")[0]})
            elif self.path == "/think":
                # inline recall: the thinker writes the anchor in its own stream, the memory completes it (theta gated),
                # and the same text continues with the answer
                q = req["question"].strip()
                m = re.match(r"^([A-Z][a-z]+):\s*(.+)$", q, re.S)
                if m:   # "Name: text" made the thinker end its turn at once
                    q = f'{m.group(1)} says: "{m.group(2)}"'
                r = ir.think_inline(q, max_tokens=int(req.get("max_tokens", 80)), canon=canon, resolve=True)
                fa.off()
                self._send(200, r)
            elif self.path == "/write2":
                # E41: the label exactly as the listener wrote it (no English owner/object clean-up), injection off after
                k = fa.write(req["e"], req["s"], req["v"])
                fa.off()
                self._send(200, {"positions": k})
            elif self.path == "/familiar":
                # E41: familiarity gate of one label (echo when FMEM_ECHO=1) and its anchor units, injection off after
                n_mem, D, k, n, layers = f.read_header()
                if (D, k) not in _SETUP:
                    _SETUP[(D, k)] = fa.setup(D, k)
                E, P, st = _SETUP[(D, k)]
                toks, *_ = f.capture_request(f"[{req['e']}|{req['s']}] ->")
                anc = fa.anchor_units(E, P, st, toks, req["e"], req["s"], k)
                fam = float(fa.familiarity(anc, D))
                fa.off()
                self._send(200, {"fam": fam, "units": [int(u) for u in anc]})
            elif self.path == "/reset":
                f.reset(None)
                fa.theta_reset()        # familiarity and key echo go with the content (as zh_v3 does)
                fa.off()
                self._send(200, {"ok": True})
            elif self.path == "/off":
                fa.off()
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "no such endpoint"})
        except Exception as e:  # report, keep serving
            self._send(500, {"error": str(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("memory service on 127.0.0.1:8094", flush=True)
    HTTPServer(("127.0.0.1", 8094), H).serve_forever()
