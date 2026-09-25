#!/usr/bin/env python3
"""Latency of one checkpoint under stated conditions: one request at a time, on one GPU, every condition recorded.

    python scripts/bench_latency.py --checkpoint <dir> --model Qwen/Qwen3-0.6B --out results/latency.json
    python scripts/bench_latency.py --mlx hunch-0.6b-preview-mlx-f16 --model Qwen/Qwen3-0.6B --out results/latency_mlx.json

The requests are the 400 typed-decisions test states with their questions (`bench_typed_decisions.to_request`; gold labels
are not read), each sent alone to `Hunch.decide`. The first 20 are warm-up and not timed. Reported: the median and 90th
percentile of wall-clock milliseconds per request and per decision, and the conditions (GPU or chip, dtype, token budget,
library versions). This is latency for single requests, not throughput. `--mlx` times an MLX artifact on Apple silicon
(fp32 activations, the runtime's default) instead of the PyTorch checkpoint on CUDA.
"""
import argparse, json, os, platform, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint"); ap.add_argument("--mlx"); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--amp", default="bf16"); ap.add_argument("--token-budget", type=int, default=16384); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fast", action="store_true", help="shared-prefix execution (hunch/fast.py): the same answers, each state read once")
    a = ap.parse_args()
    import pandas as pd, torch
    from huggingface_hub import hf_hub_download
    from bench_typed_decisions import REVISION, to_request
    from hunch.infer import Hunch
    df = pd.read_parquet(hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION))
    reqs = [to_request(r) for _, r in df.iterrows()]
    if bool(a.checkpoint) == bool(a.mlx):
        raise SystemExit("give exactly one of --checkpoint (PyTorch on CUDA) or --mlx (an MLX artifact on Apple silicon)")
    if a.mlx:   # MLX materialises its arrays before the readout returns, so wall-clock needs no extra synchronisation
        import subprocess, mlx.core as mx
        from hunch.formats.hunch_mlx import load
        h, sync = load(a.mlx, temperature=1.0, token_budget=a.token_budget, fast=a.fast), (lambda: None)
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        cond = {"device": chip, "runtime": f"MLX {mx.__version__}", "weights": "f16", "activations": "float32",
                "load_average_1min_at_start": round(os.getloadavg()[0], 2),   # a laptop in use: say how busy
                "engine": "shared-prefix (hunch/fast.py)" if a.fast else "reference (hunch/infer.py)"}
    else:
        from hunch.fast import FastHunch
        h, sync = (FastHunch if a.fast else Hunch).load(a.checkpoint, model=a.model, amp=a.amp, device="cuda"), torch.cuda.synchronize
        h.T = 1.0; h.token_budget = a.token_budget
        cond = {"gpu": torch.cuda.get_device_name(0), "amp": a.amp, "torch": torch.__version__,
                "engine": "shared-prefix (hunch/fast.py)" if a.fast else "reference (hunch/infer.py)"}
    per_req, per_dec = [], []
    for i, req in enumerate(reqs):
        sync(); t = time.perf_counter()
        h.decide([req])
        sync(); ms = (time.perf_counter() - t) * 1000
        if i >= a.warmup:
            per_req.append(ms); per_dec.append(ms / len(req["questions"]))
    q = lambda v, p: float(np.percentile(v, p))
    out = {"checkpoint": Path(a.checkpoint or a.mlx).name, "model": a.model, "requests_timed": len(per_req), "warmup": a.warmup,
           "ms_per_request": {"p50": q(per_req, 50), "p90": q(per_req, 90)}, "ms_per_decision": {"p50": q(per_dec, 50), "p90": q(per_dec, 90)},
           "conditions": {**cond, "token_budget": a.token_budget, "python": platform.python_version(), "batch": "one request at a time",
                          "suite": f"LocalLLaMA/typed-decisions test @{REVISION}"}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
