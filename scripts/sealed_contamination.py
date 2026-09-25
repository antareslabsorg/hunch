#!/usr/bin/env python3
"""Exact state matches between the sealed test read's questions and the releases' training text (docs/29).

    python scripts/sealed_contamination.py --train data/sprint_v3d/train.jsonl data/sprint_v3dgp10/train.jsonl \
        data/teacher_v1/train.jsonl --out docs/results/launch/sealed_contamination_ids.json

The comparison is `contamination_scan`'s own (state_text, then normalize), the one behind docs/21's exact matches for dev,
calibration and held-out. It reads the in-family test split only after the seal was lifted (2026-09-25 00:00 CEST), and only
for the question ids the sealed read scored (docs/results/launch/sealed/). It writes ids only, never text.
"""
import argparse, importlib.util, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("contamination_scan", ROOT / "scripts/contamination_scan.py")
cs = importlib.util.module_from_spec(_s); _s.loader.exec_module(cs)
UNSEAL = datetime(2026, 9, 25, 0, 0, tzinfo=timezone(timedelta(hours=2)))
READ = ROOT / "docs/results/launch/sealed/w16-06-v3dgp10-1200-s55_test300.json"   # both reads scored the same ids

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, nargs="+", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test", type=Path, default=ROOT / "data/sprint_v3d/test.jsonl")
    a = ap.parse_args()
    if datetime.now(timezone.utc) < UNSEAL:
        raise SystemExit("refusing: the in-family test split is sealed until 2026-09-25 00:00 CEST")
    ids = {p["id"] for p in json.loads(READ.read_text())["predictions"]}
    train = set()
    for path in a.train:
        for line in path.open():
            if line.strip():
                train.add(cs.normalize(cs.state_text(cs.row_state(json.loads(line)))))
    test = {}
    for line in a.test.open():
        if line.strip():
            obj = json.loads(line); rid = cs.row_id(obj, "")
            if rid in ids:
                test[rid] = cs.normalize(cs.state_text(cs.row_state(obj)))
    if set(test) != ids:
        raise SystemExit(f"the test split does not hold every sealed question: {len(ids - set(test))} missing")
    exact = sorted(i for i, n in test.items() if n in train)
    a.out.write_text(json.dumps({"test_sealed_read": {"n": len(test), "train_files": [str(Path(*p.parts[-2:])) for p in a.train], "exact_ids": exact}}, indent=1) + "\n")
    print(f"{len(exact)} of {len(test)} sealed questions have a state that also occurs in the training text")
