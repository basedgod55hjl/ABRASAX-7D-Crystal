# ============================================================================
# ABRASAX — Unified Xenoliving System
# PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
# CREATOR: Sir Charles Spikes (BASEDGOD)
# ============================================================================

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from noncuda_fullstack.runtime.python.cbm_runtime import CbmRuntime


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = CbmRuntime(ROOT)


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True, "service": "noncuda_fullstack"})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self):  # noqa: N802
        if self.path == "/think":
            size = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(size).decode("utf-8") if size > 0 else "{}"
            data = json.loads(raw)
            probe = data.get("probe", "")
            result = RUNTIME.think(probe)
            RUNTIME.write_status({"last_probe": probe, "last_result": result})
            self._json(200, result)
            return
        self._json(404, {"error": "not_found"})


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8099), Handler)
    print("[NONCUDA_SERVICE] listening on http://127.0.0.1:8099")
    server.serve_forever()


if __name__ == "__main__":
    main()
