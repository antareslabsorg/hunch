#!/usr/bin/env python3
"""Time and score TypeSafe's Jev 1.13 on the public `typed-decisions` suite through OpenRouter (declared in the project log (not published)).

    OPENROUTER_API_KEY=... python scripts/bench_jev.py --out docs/results/launch/jev/jev_typed_decisions.json

The same 400 test cases (revision c76749ec58bd) as the Hunch reads, one request at a time in file order. The first 20 are
warm-up; the remaining 380 are timed as wall-clock round trip from the calling machine, network and OpenRouter included. Hunch
runs locally, so its latency and this one are different kinds of number. Answers are scored against the gold with the rules of
`bench_typed_decisions.py`: argmax against the gold label, KL and Brier over the gold's label order.
"""
import argparse, json, os, platform, random, time, urllib.error, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REVISION, MODEL, URL, WARMUP = "c76749ec58bd", "typesafe/jev-1.13", "https://openrouter.ai/api/alpha/decisions", 20


def call(key, body, retries=3):
    """One decision request; returns (status, json or error text, round-trip ms, retries). Retries only on 429/5xx/network."""
    for attempt in range(retries + 1):
        req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read()), (time.perf_counter() - t) * 1000, attempt
        except urllib.error.HTTPError as e:
            status, text = e.code, e.read().decode()[:500]
        except (urllib.error.URLError, TimeoutError) as e:
            status, text = 0, repr(e)[:200]
        if status not in (0, 429) and status < 500:
            return status, text, (time.perf_counter() - t) * 1000, attempt
        time.sleep(min(8, 2 ** attempt) * (0.5 + random.random()))   # jittered, so parallel callers do not retry in step
    return status, text, (time.perf_counter() - t) * 1000, attempt


def dist(g, a):
    """Jev's answer as a vector over the gold's label order (the bench's rules), or None if the shapes differ."""
    if g["type"] == "choice":
        order = sorted(g["probabilities"]); p = [float(a.get("probabilities", {}).get(k, 0.0)) for k in order]
    elif g["type"] == "noul":
        order = ["false", "true"]; p = [1.0 - float(a["noul"]), float(a["noul"])]
    else:
        order = sorted(g["probabilities"], key=int); p = [float(a.get("probabilities", {}).get(k, 0.0)) for k in order]
    return order, np.array(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or SystemExit("set OPENROUTER_API_KEY")
    if isinstance(key, SystemExit): raise key
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION))
    if a.limit: df = df.head(a.limit)
    reqs, rows, served, cost, t0 = [], [], set(), 0.0, time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        state = r["state"]
        try: state = json.loads(state)
        except (TypeError, ValueError): pass
        status, resp, ms, retries = call(key, {"model": MODEL, "state": state, "questions": json.loads(r["questions"])})
        reqs.append({"case": r["id"], "status": status, "ms": ms, "retries": retries, "warmup": i < WARMUP,
                     "input_tokens": resp.get("usage", {}).get("input_tokens") if isinstance(resp, dict) else None})
        if status != 200 or not isinstance(resp, dict):
            reqs[-1]["error"] = str(resp)[:300]; continue
        served.add(resp.get("model")); cost += float(resp.get("usage", {}).get("cost") or 0)
        for name, g in json.loads(r["gold"]).items():
            ans = resp.get("answers", {}).get(name)
            if ans is None:
                rows.append({"case": r["id"], "q": name, "type": g["type"], "missing": True}); continue
            order, p = dist(g, ans)
            q = np.array([float(g["probabilities"].get(k, 0.0)) for k in order]); q = q / q.sum() if q.sum() > 0 else q
            rows.append({"case": r["id"], "q": name, "type": g["type"], "correct": float(order[int(np.argmax(p))] == str(g["label"])),
                         "kl": float((q * (np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum()),
                         "brier": float(((p - q) ** 2).sum()), "p": [round(float(x), 6) for x in p]})
        if (i + 1) % 50 == 0: print(f"  {i + 1}/{len(df)} cases  {time.time() - t0:.0f}s", flush=True)
    ok = [x for x in rows if "correct" in x]
    agg = lambda s: {"n": len(s), "accuracy": float(np.mean([x["correct"] for x in s])), "kl": float(np.mean([x["kl"] for x in s])),
                     "brier": float(np.mean([x["brier"] for x in s]))} if s else None
    timed = [x["ms"] for x in reqs if not x["warmup"] and x["status"] == 200]
    out = {"dataset": "LocalLLaMA/typed-decisions", "revision": REVISION, "split": "all/test", "model": MODEL,
           "served_as": sorted(m for m in served if m), "endpoint": URL, "measured_at": time.strftime("%Y-%m-%d %H:%M %Z"),
           "caller": f"{platform.system()} {platform.machine()}, one request at a time, round trip over the public internet via OpenRouter",
           "n_cases": len(df), "n_decisions": len(rows), "failed_requests": sum(x["status"] != 200 for x in reqs),
           "overall": agg(ok), "by_type": {t: agg([x for x in ok if x["type"] == t]) for t in ("choice", "noul", "score")},
           "latency_ms": {"warmup": WARMUP, "requests_timed": len(timed), "p50": float(np.median(timed)) if timed else None,
                          "p90": float(np.percentile(timed, 90)) if timed else None, "mean": float(np.mean(timed)) if timed else None,
                          "min": float(np.min(timed)) if timed else None},
           "cost_usd": round(cost, 6), "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    Path(a.out).with_suffix(".requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reqs))
    Path(a.out).with_suffix(".items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(json.dumps({k: out[k] for k in ("served_as", "failed_requests", "overall", "latency_ms", "cost_usd")}, indent=1))
