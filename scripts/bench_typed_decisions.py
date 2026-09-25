#!/usr/bin/env python3
"""Score a Hunch checkpoint on the public `typed-decisions` suite and report its leaderboard columns.

    python scripts/bench_typed_decisions.py --checkpoint <dir> --model Qwen/Qwen3-1.7B --out results.json

Declared in the project log (not published) 2026-09-23 10:40 before any prediction was generated. Revision pinned to
c76749ec58bd, whose parquet files are byte-identical to ea9306458d6e where the published Jev 1.13.0
row was measured. Temperature is fixed at 1.0 because MODEL_CARD.md pre-declares that for
out-of-family use; it is NOT fitted here, which would be selection on the split being reported.

Hunch enters as general/zero-shot: it has never seen these four workflows. Gold is the mean of three
samples from a ~4B-class teacher, so every number here measures agreement with that teacher and not
correctness, and teacher self-agreement (0.735) is the reference ceiling.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c76749ec58bd"


def to_request(row) -> dict:
    """One benchmark case -> one hunch/schema.py request. `criteria` is a label->description map for
    choice/noul and an ordered list of level descriptions for score, which is exactly the
    authored-description shape the v3d recipe trains on."""
    qs = []
    for name, q in json.loads(row["questions"]).items():
        # 200 of the 2,000 questions are bare `noul` with no `criteria` at all, so this must not
        # be unpacked before the type is known.
        t = q["type"]
        crit = q.get("criteria")
        if t == "choice":
            qs.append({"id": name, "type": "choice", "instruction": q["instructions"],
                       "candidates": [{"id": k, "label": k, "description": v} for k, v in sorted(crit.items())]})
        elif t == "noul":
            qs.append({"id": name, "type": "boolean", "instruction": q["instructions"]})
        elif t == "score" and crit is None:
            raise SystemExit(f"score question {name!r} has no criteria; cannot build levels")
        elif t == "score":
            qs.append({"id": name, "type": "score", "instruction": q["instructions"],
                       "levels": list(crit)})
        else:
            raise SystemExit(f"unhandled question type {t!r}")
    return {"id": row["id"], "state": json.loads(row["state"]), "questions": qs}


def gold_vector(g: dict, order: list[str]) -> np.ndarray:
    v = np.array([float(g["probabilities"].get(k, 0.0)) for k in order])
    s = v.sum()
    return v / s if s > 0 else v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token-budget", type=int, default=16384)
    args = ap.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from hunch.infer import Hunch

    path = hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet",
                           repo_type="dataset", revision=REVISION)
    df = pd.read_parquet(path)
    if args.limit:
        df = df.head(args.limit)

    hunch = Hunch.load(args.checkpoint, model=args.model, amp=args.amp, device=args.device)
    # T = 1.0: out-of-family, per MODEL_CARD.md. Not fitted on this split.
    hunch.T = 1.0
    hunch.token_budget = args.token_budget

    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        req = to_request(r)
        t1 = time.time()
        resp = hunch.decide([req])
        dt_ms = (time.time() - t1) * 1000.0
        gold = json.loads(r["gold"])
        answers = resp["results"][0]["answers"]
        for name, g in gold.items():
            a = answers.get(name)
            if a is None:
                rows.append({"case": r["id"], "q": name, "type": g["type"], "missing": True}); continue
            # Each response type carries its distribution differently (hunch/infer.py:110-118),
            # and the order must be the one the gold uses or KL compares different labels.
            if g["type"] == "choice":
                order = sorted(g["probabilities"])
                p = np.array([float(a["probabilities"][k]) for k in order])
            elif g["type"] == "noul":
                order = ["false", "true"]
                p = np.array([1.0 - float(a["probability"]), float(a["probability"])])
            else:
                order = sorted(g["probabilities"], key=int)
                p = np.array([float(x) for x in a["probabilities"]])
            q = gold_vector(g, order)
            if len(p) != len(q):
                rows.append({"case": r["id"], "q": name, "type": g["type"], "shape_mismatch": [len(p), len(q)]}); continue
            pred = order[int(np.argmax(p))]
            rows.append({"case": r["id"], "q": name, "type": g["type"],
                         "correct": float(pred == str(g["label"])),
                         "kl": float((q * (np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum()),
                         "brier": float(((p - q) ** 2).sum()),
                         "ms": dt_ms / max(1, len(gold)), "p": [round(float(x), 6) for x in p]})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)} cases  {time.time()-t0:.0f}s", flush=True)

    ok = [r for r in rows if "correct" in r]
    def agg(sel):
        s = [r for r in ok if sel(r)]
        if not s: return None
        return {"n": len(s), "accuracy": float(np.mean([r["correct"] for r in s])),
                "kl": float(np.mean([r["kl"] for r in s])), "brier": float(np.mean([r["brier"] for r in s]))}
    out = {"dataset": "LocalLLaMA/typed-decisions", "revision": REVISION, "split": "all/test",
           "checkpoint": args.checkpoint, "model": args.model, "temperature": 1.0,
           "kind": "general, zero-shot", "n_cases": len(df), "n_decisions": len(rows),
           "skipped": [r for r in rows if "correct" not in r][:20],
           "overall": agg(lambda r: True),
           "by_type": {t: agg(lambda r, t=t: r["type"] == t) for t in ("choice", "noul", "score")},
           "latency_ms_p50_per_decision": float(np.median([r["ms"] for r in ok])),
           "seconds": time.time() - t0,
           "teacher_self_agreement_reference": 0.735}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    # per-decision rows beside the summary: case-level bootstrap intervals and paired comparisons need them
    Path(args.out).with_suffix(".items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({k: out[k] for k in ("overall", "by_type", "latency_ms_p50_per_decision", "n_decisions")}, indent=1))


if __name__ == "__main__":
    main()
