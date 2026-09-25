#!/usr/bin/env python3
"""Equivalence of shared-prefix execution on an MLX build (`hunch_mlx.load(..., fast=True)`), as declared in the project log (not published).

    python scripts/check_fast_mlx.py --mlx hunch/formats/out/hunch-0.6b-preview-mlx-f16 \
        --reference hunch/formats/out/preds-0.6b/mlx-f16.jsonl --fp32 hunch/formats/out/preds-0.6b/pytorch-fp32-cpu.jsonl \
        --floor hunch/formats/out/equivalence-0.6b.json --out out.json

The reference is the MLX read the formats gate recorded: per question, at T = 1, on the 6,000 held-out questions. The fast
engine re-scores exactly those questions, by id and in that order, and is compared per question with it (answer changed,
total-variation distance) and with the fp32 PyTorch read under FORMATS.md's floor rule (`floor_gate`).
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from check_fast import dist, request_of
from hunch.formats.equivalence_test import floor_gate
from hunch.formats.hunch_mlx import load
from hunch.schema import read_jsonl


def probs(path: str) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line); out[r["id"]] = r["probs"]
    return out


def compare(got: dict, ref: dict) -> dict:
    tv, changed = [], 0
    for i, p in got.items():
        q = ref[i]
        if len(p) != len(q):
            raise SystemExit(f"{i}: {len(p)} probabilities against {len(q)} in the reference")
        tv.append(0.5 * sum(abs(u - v) for u, v in zip(p, q)))
        changed += max(range(len(p)), key=p.__getitem__) != max(range(len(q)), key=q.__getitem__)
    return {"n": len(tv), "answers_changed": changed, "tv_max": max(tv), "tv_mean": sum(tv) / len(tv)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlx", required=True); ap.add_argument("--reference", required=True); ap.add_argument("--fp32", required=True)
    ap.add_argument("--floor", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--heldout", default=str(ROOT / "data/sprint_v3d/heldout.jsonl")); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    ref, fp32 = probs(a.reference), probs(a.fp32)
    if set(ref) != set(fp32):
        raise SystemExit("the MLX and fp32 reads cover different questions")
    recs = {r.id: r for r in read_jsonl(Path(a.heldout)) if r.id in ref}
    if len(recs) != len(ref):
        raise SystemExit(f"{len(ref) - len(recs)} reference questions are missing from {a.heldout}")
    h, ids, got, tokens, t0 = load(a.mlx, temperature=1.0, fast=True), list(ref), {}, 0, time.time()
    for s in range(0, len(ids), a.batch):
        chunk = ids[s:s + a.batch]
        out = h.decide([request_of(recs[i]) for i in chunk])
        tokens += out["usage"]["path_tokens"]
        for i, res in zip(chunk, out["results"]):
            if res["state_id"] != i:
                raise SystemExit(f"result order: {res['state_id']} where {i} was sent")
            got[i] = dist(res["answers"]["q"])
        if (s // a.batch) % 50 == 0:
            print(f"{len(got)}/{len(ids)} scored, {time.time() - t0:.0f} s", flush=True)
    nf = json.loads(Path(a.floor).read_text())["noise_floor"]
    (key,) = [k for k in nf if k.startswith("pytorch-fp32")]   # the size's one fp32 read against its bf16 archive
    vs_ref, vs_fp32 = compare(got, ref), compare(got, fp32)
    res = {"artifact": Path(a.mlx).name, "settings": h.scorer.settings, "engine": "shared-prefix (hunch/fast.py)", "path_tokens": tokens,
           "seconds": round(time.time() - t0, 1), "vs_mlx_reference": vs_ref,
           "gate_exact": {"rule": "no answer changed and largest TV below 1e-4",
                          "verdict": "pass" if vs_ref["answers_changed"] == 0 and vs_ref["tv_max"] < 1e-4 else "fail"},
           "vs_fp32": vs_fp32, "fp32_floor_gate": floor_gate({"n": vs_fp32["n"], "n_nonfinite": 0, "n_argmax_diff": vs_fp32["answers_changed"],
                                                               "tv_max": vs_fp32["tv_max"]}, nf[key])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
