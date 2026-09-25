#!/usr/bin/env python3
"""Equivalence of shared-prefix execution (hunch/fast.py) with the reference (hunch/infer.py), declared in the project log (not published).

    python scripts/check_fast.py --checkpoint <dir> --model Qwen/Qwen3-0.6B --heldout data/sprint_v3d/heldout.jsonl --out out.json \
        [--ref-amp fp32 --fast-amp bf16 --floor-changed 28 --floor-tv 7.68e-02]

Both engines share one set of weights in memory; each runs at its own precision (bf16 by default). Compared per question: whether the argmax answer changes, and the
total-variation distance between the two distributions, on the first 2,000 held-out questions per family and on the 2,000
typed-decisions decisions. Temperature 1 for both, since only their difference is measured.
"""
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hunch.fast import FastHunch
from hunch.infer import Hunch
from hunch.schema import read_jsonl


def dist(ans: dict) -> list[float]:
    if ans["type"] == "choice": return list(ans["probabilities"].values())
    if ans["type"] == "boolean": return [1 - ans["probability"], ans["probability"]]
    return list(ans["probabilities"])


def request_of(r) -> dict:
    q = r.question
    return {"id": r.id, "state": r.state, "questions": [{"id": "q", "type": q.type, "instruction": q.instruction,
            "candidates": [{"id": c.id, "label": c.label, "description": c.description} for c in q.candidates],
            "levels": list(q.levels), "criteria": dict(q.criteria)}]}


def compare(ref, fast, reqs, batch):
    changed, tvs, n = 0, [], 0
    for s in range(0, len(reqs), batch):
        chunk = reqs[s:s + batch]
        a, b = ref.decide(chunk), fast.decide(chunk)
        for ra, rb in zip(a["results"], b["results"]):
            for qid, x in ra["answers"].items():
                p, q = dist(x), dist(rb["answers"][qid])
                tvs.append(0.5 * sum(abs(u - v) for u, v in zip(p, q)))
                changed += int(max(range(len(p)), key=p.__getitem__) != max(range(len(q)), key=q.__getitem__)); n += 1
    return {"n": n, "answers_changed": changed, "tv_max": max(tvs), "tv_mean": sum(tvs) / len(tvs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--heldout", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-family", type=int, default=2000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ref-amp", default="bf16", choices=["bf16", "fp32"]); ap.add_argument("--fast-amp", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--floor-changed", type=int, help="the checkpoint's bf16-vs-fp32 floor: answers changed of the held-out questions")
    ap.add_argument("--floor-tv", type=float, help="the floor's largest total-variation distance")
    a = ap.parse_args()
    amp = {"bf16": torch.bfloat16, "fp32": None}
    ref = Hunch.load(a.checkpoint, model=a.model, amp=a.ref_amp, device="cuda"); ref.T = 1.0
    fast = FastHunch(ref.scorer, ref.tok, 1.0, ref.device, amp[a.fast_amp], ref.token_budget)
    seen, held = defaultdict(int), []
    for r in read_jsonl(Path(a.heldout)):
        if seen[r.family] < a.per_family:
            seen[r.family] += 1; held.append(request_of(r))
    import pandas as pd
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_typed_decisions import REVISION, to_request
    td = [to_request(r) for _, r in pd.read_parquet(hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet",
                                                                    repo_type="dataset", revision=REVISION)).iterrows()]
    t0 = time.time()
    out = {"checkpoint": Path(a.checkpoint).name, "model": a.model, "ref_amp": a.ref_amp, "fast_amp": a.fast_amp, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "heldout": {**compare(ref, fast, held, a.batch), "families": dict(seen)}, "typed_decisions": compare(ref, fast, td, 1)}
    out["seconds"] = round(time.time() - t0, 1)
    if a.floor_changed is not None and a.floor_tv is not None:   # the on-device builds' floor rule (FORMATS.md)
        import math
        h = out["heldout"]; n = h["n"]; p1, p0 = h["answers_changed"] / n, a.floor_changed / n; pb = (p1 + p0) / 2
        z = (p1 - p0) / math.sqrt(2 * pb * (1 - pb) / n) if 0 < pb < 1 else 0.0
        out["floor_gate"] = {"floor_changed": a.floor_changed, "floor_tv": a.floor_tv, "z": z,
                             "verdict": "pass" if h["tv_max"] <= a.floor_tv and z < 1.6449 else "fail"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out, indent=1) + "\n"); print(json.dumps(out, indent=1))
