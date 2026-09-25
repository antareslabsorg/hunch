#!/usr/bin/env python3
"""Score TypeSafe's Jev 1.13 on Hunch's own task families (the calibration split) through OpenRouter, beside the stored Hunch
reads of the same questions (declared in the project log (not published)).

    OPENROUTER_API_KEY=... python scripts/bench_jev_infamily.py --out docs/results/launch/jev/jev_calibration.json

The questions are the 3,523 of the release checkpoints' calibration reads. Each is one request carrying what Hunch reads: the
state verbatim, the instruction, a choice's candidates as criteria {label: description}, a boolean's false/true criteria, a
score's levels. Jev answers zero-shot; Hunch was trained on these 12 families (not on this split, which only fitted its
temperature). Accuracy is `hunch.metrics.accuracy` against each record's target, on the questions Jev answered, for all three
models on the same questions; Brier and NLL are secondary. Latency is not measured here (4 requests are in flight).
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from bench_jev import MODEL, call
from hunch.metrics import accuracy, brier, nll

READS = {"Hunch 0.6B": "docs/results/day6/w16-06-v3dgp10-1200-s55_calibration.json",
         "Hunch 1.7B": "docs/results/day7/v1/g2-17-v4t_calibration.json"}
OVERLAP = "docs/results/launch/contamination_ids.json"   # docs/21: calibration questions whose state occurs in training text


def body(r: dict) -> dict:
    q = r["question"]
    if q["type"] == "choice":
        labels = [c["label"] for c in q["candidates"]]
        if len(set(labels)) != len(labels):
            raise SystemExit(f"{r['id']}: duplicate candidate labels")
        jq = {"type": "choice", "instructions": q["instruction"],
              "criteria": {c["label"]: c.get("description") or c["label"] for c in q["candidates"]}}
    elif q["type"] == "boolean":
        jq = {"type": "noul", "instructions": q["instruction"], **({"criteria": q["criteria"]} if q.get("criteria") else {})}
    elif q["type"] == "score":
        jq = {"type": "score", "instructions": q["instruction"], "criteria": list(q["levels"])}
    else:
        raise SystemExit(f"{r['id']}: question type {q['type']!r}")
    return {"model": MODEL, "state": r["state"], "questions": {"q": jq}}


def vector(r: dict, ans: dict):
    """Jev's answer in the record's candidate order (the target's), the number of options it gave no probability, or None."""
    q = r["question"]
    if q["type"] == "boolean":
        return (np.array([1 - float(ans["noul"]), float(ans["noul"])]), 0) if "noul" in ans else (None, 0)
    keys = [c["label"] for c in q["candidates"]] if q["type"] == "choice" else [str(i) for i in range(len(q["levels"]))]
    pr = ans.get("probabilities") or {}
    p = np.array([float(pr.get(k, 0.0)) for k in keys])   # an option Jev leaves out gets 0, as bench_jev.py scores it
    return (p / p.sum() if p.sum() > 0 else None), sum(k not in pr for k in keys)


def scores(rows: list, key: str) -> dict:
    ok = [x for x in rows if x[key] is not None]
    if not ok:
        return None
    m = lambda f: float(np.mean([f(np.array([x[key]]), np.array([x["target"]]))[0] for x in ok]))
    return {"n": len(ok), "accuracy": m(accuracy), "brier": m(brier), "nll": m(nll)}


def paired(rows: list, a: str, b: str, draws: int = 10_000, seed: int = 0) -> dict:
    """Accuracy of a minus b on the same questions, with a bootstrap interval that resamples state groups."""
    groups = sorted({x["group"] for x in rows}); gi = {g: i for i, g in enumerate(groups)}
    d = np.zeros(len(groups)); n = np.zeros(len(groups))
    for x in rows:
        d[gi[x["group"]]] += accuracy(np.array([x[a]]), np.array([x["target"]]))[0] - accuracy(np.array([x[b]]), np.array([x["target"]]))[0]
        n[gi[x["group"]]] += 1
    rng, stats = np.random.default_rng(seed), []
    for _ in range(draws):
        idx = rng.integers(0, len(groups), len(groups)); stats.append(d[idx].sum() / n[idx].sum())
    return {"difference": float(d.sum() / n.sum()), "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))],
            "draws": draws, "seed": seed, "unit": "state group"}


def report(rows: list, other: str = "jev", label: str = "Jev") -> dict:
    """Every model on the same questions; `other` is the row key of the model beside the two Hunch reads."""
    fams = sorted({x["family"] for x in rows})
    names = [other] + list(READS)
    return {"overall": {k: scores(rows, k) for k in names},
            "families": {f: {k: scores([x for x in rows if x["family"] == f], k) for k in names} for f in fams},
            "paired": {f"{h} - {label}": paired(rows, h, other) for h in READS}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")
    reads = {h: json.loads((ROOT / p).read_text())["predictions"] for h, p in READS.items()}
    ids = [x["id"] for x in reads["Hunch 1.7B"]]
    if [x["id"] for x in reads["Hunch 0.6B"]] != ids:
        raise SystemExit("the two stored reads cover different questions")
    if a.limit:
        ids = ids[:a.limit]
    want, recs = set(ids), {}
    for line in open(ROOT / "data/sprint_v3d/calibration.jsonl"):
        r = json.loads(line)
        if r["id"] in want:
            recs[r["id"]] = r
    if len(recs) != len(ids):
        raise SystemExit(f"{len(ids) - len(recs)} questions of the reads are missing from the calibration split")
    hp = {h: {x["id"]: x for x in reads[h]} for h in READS}
    for i in ids:   # the stored target must be the record's target, or the two sides are not scored alike
        if not np.allclose(hp["Hunch 1.7B"][i]["target"], recs[i]["target"]["probs"]):
            raise SystemExit(f"{i}: stored target differs from the calibration record")
    t0, done = time.time(), [0]

    def one(i):
        r = recs[i]
        status, resp, ms, retries = call(key, body(r))
        row = {"id": i, "family": r["family"], "group": hp["Hunch 1.7B"][i]["group"], "qtype": r["question"]["type"],
               "status": status, "retries": retries, "target": [float(v) for v in r["target"]["probs"]], "jev": None, "missing_options": 0}
        if status == 200 and isinstance(resp, dict) and "q" in resp.get("answers", {}):
            p, miss = vector(r, resp["answers"]["q"])
            row.update({"jev": None if p is None else [round(float(v), 6) for v in p], "missing_options": miss,
                        "served": resp.get("model"), "cost": float((resp.get("usage") or {}).get("cost") or 0),
                        "input_tokens": (resp.get("usage") or {}).get("input_tokens")})
        else:
            row["error"] = str(resp)[:300]
        for h in READS:
            row[h] = hp[h][i]["probs"]
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
        return row

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, ids))
    answered = [x for x in rows if x["jev"] is not None]
    overlap = set(json.loads((ROOT / OVERLAP).read_text())["calibration"]["exact_ids"])
    clean = [x for x in answered if x["id"] not in overlap]
    out = {"model": MODEL, "served_as": sorted({x.get("served") for x in rows if x.get("served")}), "measured_at": time.strftime("%Y-%m-%d %H:%M %Z"),
           "questions": "the calibration-split questions of the two release reads", "reads": READS, "n": len(rows),
           "answered": len(answered), "failed_requests": sum(x["status"] != 200 for x in rows),
           "unusable_answers": sum(x["status"] == 200 and x["jev"] is None for x in rows),
           "answers_with_missing_options": sum(x["missing_options"] > 0 for x in answered), "retries": sum(x["retries"] for x in rows),
           "cost_usd": round(sum(x.get("cost", 0) for x in rows), 6), "seconds": round(time.time() - t0, 1),
           "all": report(answered), "without_exact_overlap": {"excluded": len(answered) - len(clean), **report(clean)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    Path(a.out).with_suffix(".items.jsonl").write_text("".join(json.dumps({k: v for k, v in x.items() if k not in READS}) + "\n" for x in rows))
    print(json.dumps({k: out[k] for k in ("served_as", "answered", "failed_requests", "unusable_answers", "cost_usd")}, indent=1))
    print(json.dumps({k: v for k, v in out["all"]["overall"].items()}, indent=1)); print(json.dumps(out["all"]["paired"], indent=1))
