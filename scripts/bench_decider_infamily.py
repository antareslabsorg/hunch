#!/usr/bin/env python3
"""The open Decider on Hunch's own task families (the calibration split), beside the stored Hunch reads (declared in the project log (not published)).

    python scripts/bench_decider_infamily.py --model Mapika/decider-2b --revision 9839cc9d908be16c5988c0d041034b5fdf82c7a2 \
        --out docs/results/launch/decider_calibration_decider-2b.json [--probe]

The questions, request bodies, answer mapping and scoring are those of `scripts/bench_jev_infamily.py` (the Jev measurement),
sent one question at a time to the Decider's own Jev-shaped interface (`Decider.system_one`, shipped temperature). `--probe`
sends one question per family, prints the answer shapes and writes nothing.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from bench_jev_infamily import OVERLAP, READS, body, report, vector
from bench_td_decider import load_decider

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--revision", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--calibration", default=str(ROOT / "data/sprint_v3d/calibration.jsonl"))
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    reads = {h: {x["id"]: x for x in json.loads((ROOT / p).read_text())["predictions"]} for h, p in READS.items()}
    ids = list(reads["Hunch 1.7B"])
    if list(reads["Hunch 0.6B"]) != ids:
        raise SystemExit("the two stored reads cover different questions")
    want, recs = set(ids), {}
    for line in open(a.calibration):
        r = json.loads(line)
        if r["id"] in want:
            recs[r["id"]] = r
    if len(recs) != len(ids):
        raise SystemExit(f"{len(ids) - len(recs)} questions of the reads are missing from {a.calibration}")
    for i in ids:   # the stored target must be the record's target, or the sides are not scored alike
        if not np.allclose(reads["Hunch 1.7B"][i]["target"], recs[i]["target"]["probs"]):
            raise SystemExit(f"{i}: stored target differs from the calibration record")
    if a.probe:
        seen = {}
        for i in ids:
            seen.setdefault(recs[i]["family"], i)
        ids = list(seen.values())
    d, sha = load_decider(a.model, a.device, a.revision)
    rows, t0 = [], time.time()
    for n, i in enumerate(ids):
        r, b = recs[i], body(recs[i])
        t = time.time()
        try:
            ans, err = d.system_one(b["state"], b["questions"])["answers"].get("q"), None
        except Exception as e:   # a question the Decider cannot take is reported, not dropped
            ans, err = None, repr(e)[:300]
        p, miss = vector(r, ans) if ans else (None, 0)
        row = {"id": i, "family": r["family"], "group": reads["Hunch 1.7B"][i]["group"], "qtype": r["question"]["type"],
               "target": [float(v) for v in r["target"]["probs"]], "decider": None if p is None else [round(float(v), 6) for v in p],
               "missing_options": miss, "ms": round((time.time() - t) * 1000, 1), **({"error": err} if err else {})}
        for h in READS:
            row[h] = reads[h][i]["probs"]
        rows.append(row)
        if a.probe:
            print(f"{r['family'][:36]:36s} keys={sorted(ans) if ans else err}  missing={miss}  decider_argmax={None if p is None else int(np.argmax(p))}"
                  f"  target_argmax={int(np.argmax(row['target']))}  hunch17_argmax={int(np.argmax(row['Hunch 1.7B']))}")
        elif (n + 1) % 250 == 0:
            print(f"  {n + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    if a.probe:
        sys.exit(0)
    answered = [x for x in rows if x["decider"] is not None]
    overlap = set(json.loads((ROOT / OVERLAP).read_text())["calibration"]["exact_ids"])
    clean = [x for x in answered if x["id"] not in overlap]
    import torch
    out = {"model": a.model, "revision": sha, "interface": "Decider.system_one, shipped temperature",
           "device": torch.cuda.get_device_name(0) if a.device == "cuda" else a.device, "measured_at": time.strftime("%Y-%m-%d %H:%M %Z"),
           "questions": "the calibration-split questions of the two release reads", "reads": READS, "n": len(rows), "answered": len(answered),
           "errors": sum("error" in x for x in rows), "unusable_answers": sum("error" not in x and x["decider"] is None for x in rows),
           "answers_with_missing_options": sum(x["missing_options"] > 0 for x in answered), "seconds": round(time.time() - t0, 1),
           "all": report(answered, "decider", "Decider"), "without_exact_overlap": {"excluded": len(answered) - len(clean), **report(clean, "decider", "Decider")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    Path(a.out).with_suffix(".items.jsonl").write_text("".join(json.dumps({k: v for k, v in x.items() if k not in READS}) + "\n" for x in rows))
    print(json.dumps({k: out[k] for k in ("answered", "errors", "unusable_answers", "seconds")}, indent=1))
    print(json.dumps(out["all"]["overall"], indent=1)); print(json.dumps(out["all"]["paired"], indent=1))
