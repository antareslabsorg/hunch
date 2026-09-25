"""Minimal HTTP server for the decision API (stdlib only; hunch/schema.py schema, single process, one model).

    python -m hunch.serve --checkpoint <dir with model.safetensors and hunch_config.json> --model Qwen/Qwen3-1.7B --port 8080

--device defaults to cuda when it is available, else cpu. --temperature overrides the release temperature (1.0 for
out-of-family inputs, as every benchmark here uses). The port opens before the model loads: /healthz answers at once,
/readyz is 503 until the model is loaded and a warm-up request has passed, and a failed load exits non-zero.

Endpoints: POST /v1/decide (JSON {states:[...]}), GET /healthz (liveness), GET /readyz (readiness: model loaded and
warm-up request succeeded). A request either returns a distribution for every question or fails as a whole with a
JSON error and a non-2xx status. No silent partial results.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .infer import Hunch

STATE = {"ready": False, "model": None, "requests": 0, "errors": 0, "started": time.time()}
LOCK = threading.Lock()  # one GPU, one forward at a time


class Handler(BaseHTTPRequestHandler):
    server_version = "hunch/0.0.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            return self._send(200, {"status": "alive", "uptime_s": round(time.time() - STATE["started"])})
        if self.path == "/readyz":
            return self._send(200 if STATE["ready"] else 503, {"ready": STATE["ready"], "requests": STATE["requests"], "errors": STATE["errors"]})
        return self._send(404, {"error": "not_found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/decide":
            return self._send(404, {"error": "not_found"})
        if not STATE["ready"]:
            return self._send(503, {"error": "not_ready"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 50 * 1024 * 1024:
                return self._send(413, {"error": "payload_too_large"})
            req = json.loads(self.rfile.read(length))
            states = req["states"]
            if not isinstance(states, list) or not states:
                raise ValueError("states must be a non-empty list")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:   # TypeError: a body that is not a JSON object
            STATE["errors"] += 1
            return self._send(400, {"error": "validation", "message": str(exc)})
        try:
            with LOCK:
                out = STATE["model"].decide(states)
            STATE["requests"] += 1
            return self._send(200, out)
        except ValueError as exc:  # schema/limit violations found during serialization
            STATE["errors"] += 1
            return self._send(400, {"error": "validation", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            STATE["errors"] += 1
            print(f"ERROR decide: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return self._send(500, {"error": "internal", "message": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):  # quiet access log; errors are printed explicitly
        return


WARMUP = [{"id": "warmup", "state": {"ticket": "Where is my card? It has been two weeks."},
           "questions": [{"id": "q", "type": "choice", "instruction": "Which intent best describes the message?",
                          "candidates": [{"id": "card_arrival", "label": "card arrival"}, {"id": "lost_card", "label": "lost or stolen card"}]},
                         {"id": "b", "type": "boolean", "instruction": "Is the customer asking about a physical card?"}]}]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", default="auto")
    p.add_argument("--temperature", type=float, help="override the release temperature from hunch_config.json; 1.0 for inputs unlike the training families")
    args = p.parse_args()
    # listen before loading: /healthz answers during a slow load and /readyz says 503 until the warm-up has passed
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    t0 = time.time()
    try:
        model = Hunch.load(args.checkpoint, model=args.model, device=args.device, amp=args.amp)
        if args.temperature is not None:
            model.T = args.temperature
        out = model.decide(WARMUP)
        total = sum(out["results"][0]["answers"]["q"]["probabilities"].values())
        if abs(total - 1) > 1e-4:
            raise RuntimeError(f"warm-up probabilities sum to {total}, not 1")
    except Exception as exc:  # noqa: BLE001 - a model that cannot load or warm up must not look alive
        print(f"ERROR load: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        server.shutdown()
        sys.exit(1)
    STATE.update(model=model, ready=True)
    print(json.dumps({"ready": True, "device": args.device, "temperature": model.T, "load_s": round(time.time() - t0, 1), "warmup": out["results"][0]["answers"]}), flush=True)
    threading.Event().wait()  # serve until the process is stopped


if __name__ == "__main__":
    main()
