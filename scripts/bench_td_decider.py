#!/usr/bin/env python3
"""The open Decider on typed-decisions and v2-system-one, scored by the same code as Hunch (the project log (not published) 2026-09-23 22:54).

    python scripts/bench_td_decider.py --model Mapika/decider-2b --revision 9839cc9d908be16c5988c0d041034b5fdf82c7a2 --out td_decider-2b.json [--suite v2so]

Scores the open Decider through its own Jev-shaped interface (`Decider.system_one`, shipped temperature, its own
per-question isolation) on LocalLLaMA/typed-decisions test (rev c76749ec58bd; 400 cases, 2,000 decisions) with the same
gold alignment and metrics as scripts/bench_typed_decisions.py, so the two models are compared by identical code.
`--suite v2so` scores pngwn/typed-decisions-v2-system-one test (5,214 choice rows) through `Decider.decide`.
Zero-shot for Decider: its public data registry (decider/data/mixture.py) contains no typed-decisions data.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

TD_REV = "c76749ec58bd"
V2SO_REV = "4af44cdb3cb6e7a4374ef6a2e86bbc656e93d29e"


def load_decider(model, device, revision=None):
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, revision=revision)                # model repo ships its own `decider/` package
    sys.path.insert(0, path)
    from decider.infer import Decider
    from huggingface_hub import HfApi
    return Decider(path, device=device), HfApi().model_info(model, revision=revision).sha


def gold_vector(g, order):
    v = np.array([float(g["probabilities"].get(k, 0.0)) for k in order]); s = v.sum()
    return v / s if s > 0 else v


def td(d, a):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("LocalLLaMA/typed-decisions", "all/test-00000-of-00001.parquet", repo_type="dataset", revision=TD_REV))
    if a.limit: df = df.head(a.limit)
    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        state, questions, gold = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["gold"])
        t1 = time.time(); ans = d.system_one(state, questions)["answers"]; dt = (time.time() - t1) * 1000.0
        for name, g in gold.items():
            x = ans.get(name)
            if x is None: rows.append({"case": r["id"], "q": name, "type": g["type"], "missing": True}); continue
            if g["type"] == "choice":
                order = sorted(g["probabilities"]); p = np.array([float(x["probabilities"].get(k, 0.0)) for k in order])
            elif g["type"] == "noul":
                order = ["false", "true"]; p = np.array([1.0 - float(x["noul"]), float(x["noul"])])
            else:
                order = sorted(g["probabilities"], key=int); p = np.array([float(x["probabilities"].get(k, 0.0)) for k in order])
            s = p.sum(); p = p / s if s > 0 else p
            q = gold_vector(g, order)
            rows.append({"case": r["id"], "q": name, "type": g["type"], "correct": float(order[int(np.argmax(p))] == str(g["label"])),
                         "kl": float((q * (np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum()),
                         "brier": float(((p - q) ** 2).sum()), "ms": dt / max(1, len(gold)), "p": [round(float(v), 6) for v in p]})
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(df)} cases {time.time()-t0:.0f}s", flush=True)
    ok = [x for x in rows if "correct" in x]
    def agg(sel):
        s = [x for x in ok if sel(x)]
        return None if not s else {"n": len(s), "accuracy": float(np.mean([x["correct"] for x in s])),
                                   "kl": float(np.mean([x["kl"] for x in s])), "brier": float(np.mean([x["brier"] for x in s]))}
    Path(a.out).with_suffix(".items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))   # per-decision rows for paired tests
    return {"dataset": "LocalLLaMA/typed-decisions", "revision": TD_REV, "split": "all/test", "n_cases": len(df), "n_decisions": len(rows),
            "missing": sum(1 for x in rows if x.get("missing")), "overall": agg(lambda x: True),
            "by_type": {t: agg(lambda x, t=t: x["type"] == t) for t in ("choice", "noul", "score")},
            "latency_ms_p50_per_decision": float(np.median([x["ms"] for x in ok])) if ok else None, "seconds": time.time() - t0}


def v2so(d, a):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("pngwn/typed-decisions-v2-system-one", "data/test-00000-of-00001.parquet", repo_type="dataset", revision=V2SO_REV))
    if a.limit: df = df.head(a.limit)
    correct, n, t0 = 0, 0, time.time()
    for i, (_, r) in enumerate(df.iterrows()):
        opts = [str(o) for o in (json.loads(r["options"]) if isinstance(r["options"], str) else list(r["options"]))]
        res = d.decide(str(r["state"]), [{"question": str(r["question"]), "options": opts}])[0]
        pr = res["probs"]; j = max(range(len(opts)), key=lambda k: float(pr.get(opts[k], 0.0)))
        correct += int(j == int(r["answer_index"])); n += 1
        if (i + 1) % 500 == 0: print(f"  {i+1}/{len(df)} rows {time.time()-t0:.0f}s acc {correct/n:.4f}", flush=True)
    return {"dataset": "pngwn/typed-decisions-v2-system-one", "revision": V2SO_REV, "split": "test", "n": n,
            "overall": {"n": n, "accuracy": correct / n if n else None}, "seconds": time.time() - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--suite", choices=["td", "v2so"], default="td"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--revision", default=None, help="model repository revision (the reads here: 2b 9839cc9d, 0.8b a0a01d6f)")
    a = ap.parse_args()
    d, rev = load_decider(a.model, a.device, a.revision)
    out = (td if a.suite == "td" else v2so)(d, a)
    out.update(model=a.model, model_revision=rev, kind="general, zero-shot (open Decider, own interface, shipped temperature)")
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("model", "overall") if k in out} | ({"by_type": out["by_type"]} if "by_type" in out else {}), indent=1))
