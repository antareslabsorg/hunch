#!/usr/bin/env python3
"""Training data built from outside sources (the project log (not published), 2026-09-23 22:54).

    python scripts/build_v1_data.py teacher --src .cache/decider/teacher_data --out data/teacher_v1   # the 1.7B release
    python scripts/build_v1_data.py td-fit  --out data/td_fit      # research only: no released checkpoint uses it

td-fit   LocalLLaMA/typed-decisions `all/train` (rev c76749ec58bd, Apache-2.0) -> records through the evaluation's own
         `to_request`, so training and scoring see identical questions; targets are the suite's full gold distributions.
         15 % of cases (hash of the case id) go to dev.jsonl. The test split is never written anywhere.
teacher  The open Decider's teacher_data (github.com/Mapika/decider, Apache-2.0, commit in <src>/COMMIT) -> records in
         families `teacher.*`. Questions the teacher did not re-answer the same way (`teacher_ok` false) are dropped;
         targets carry the teacher's probability on the written answer (`teacher_p`) with the remainder spread evenly,
         one-hot where no probability exists. 5 % dev per family by state hash. States that match a typed-decisions test
         state or one of our held-out states (normalised exact match) are removed and counted; in addition, five
         40-character windows of every typed-decisions test state are searched for in the teacher text and the hits counted.

Every record goes through hunch.schema.Record.from_dict + validate before it is written; the run ends with counts.
"""
import argparse, hashlib, json, random, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from hunch.schema import Record  # noqa: E402

TD_REPO, TD_REV = "LocalLLaMA/typed-decisions", "c76749ec58bd"


def h100(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 100


def norm(x) -> str:
    s = x if isinstance(x, str) else json.dumps(x, sort_keys=True, ensure_ascii=False)
    return re.sub(r"\s+", " ", s.lower()).strip()


def emit(recs: list, out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fams, splits = Counter(), Counter()
    with (out_dir / "train.jsonl").open("w") as ftr, (out_dir / "dev.jsonl").open("w") as fdv:
        for d in recs:
            r = Record.from_dict(d); r.validate()                      # schema gate: nothing unvalidated is written
            (fdv if r.split == "dev" else ftr).write(json.dumps(d, ensure_ascii=False) + "\n")
            fams[(r.family, r.split)] += 1; splits[r.split] += 1
    report.update(splits=dict(splits), families={f"{f}|{s}": n for (f, s), n in sorted(fams.items())})
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "families"}, indent=1))
    for k, v in sorted(report["families"].items()): print(f"  {k:48s} {v}")


def target(k: int, ans: int, tp) -> dict:
    if tp is None:
        p = [0.0] * k; p[ans] = 1.0
    else:
        tp = min(max(float(tp), 1.0 / k), 1.0); rest = (1.0 - tp) / (k - 1) if k > 1 else 0.0
        p = [rest] * k; p[ans] = tp
    s = sum(p)
    # schema vocabulary: a model's written answer without adjudication is a hard pseudo-label; with the teacher's own
    # probability on it (the rest spread evenly) it is an approximate teacher distribution
    return {"semantic_kind": "label_distribution", "provenance": "hard_pseudo_label" if tp is None else "teacher_distribution",
            "probs": [x / s for x in p], "hard": ans}


def td_fit(a) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from bench_typed_decisions import to_request
    rows = pq.read_table(hf_hub_download(TD_REPO, "all/train-00000-of-00001.parquet", repo_type="dataset", revision=TD_REV)).to_pylist()
    assert all(r["split"] == "train" for r in rows), "refusing: a non-train row reached the fit builder"
    recs = []
    for row in rows:
        req, gold = to_request(row), json.loads(row["gold"])
        split = "dev" if h100(row["id"]) < 15 else "train"
        for q in req["questions"]:
            g = gold[q["id"]]["probabilities"]
            if q["type"] == "choice": p = [float(g.get(c["id"], 0.0)) for c in q["candidates"]]
            elif q["type"] == "boolean": p = [float(g.get("false", 0.0)), float(g.get("true", 0.0))]
            else: p = [float(g.get(str(i), 0.0)) for i in range(len(q["levels"]))]
            s = sum(p); assert s > 0, (row["id"], q["id"]); p = [x / s for x in p]
            recs.append({"id": f"td:{row['id']}:{q['id']}", "source": f"{TD_REPO}@{TD_REV}", "source_group": f"td:{row['id']}",
                         "family": f"fitted.td.{row['workflow']}.{q['type']}", "split": split, "license_class": "B",
                         "state": req["state"],
                         "question": {"id": q["id"], "type": q["type"], "instruction": q["instruction"],
                                      "candidates": q.get("candidates", []), "levels": q.get("levels", []), "criteria": {}},
                         "target": {"semantic_kind": "label_distribution", "provenance": "teacher_distribution",
                                    "probs": p, "hard": max(range(len(p)), key=p.__getitem__)},
                         "meta": {"workflow": row["workflow"]}})
    emit(recs, Path(a.out), {"source": f"{TD_REPO}@{TD_REV} all/train", "cases": len(rows), "questions": len(recs)})


def teacher(a) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    src = Path(a.src); commit = (src / "COMMIT").read_text().strip() if (src / "COMMIT").exists() else "unknown"
    # contamination guards: typed-decisions test states and our held-out states
    test = pq.read_table(hf_hub_download(TD_REPO, "all/test-00000-of-00001.parquet", repo_type="dataset", revision=TD_REV)).to_pylist()
    td_states = [norm(json.loads(r["state"])) for r in test]
    guard = set(td_states)
    for l in open(ROOT / "data/sprint_v3d/heldout.jsonl"):
        guard.add(norm(json.loads(l)["state"]))
    rng = random.Random(0)
    windows = [s[i:i + 40] for s in td_states if len(s) > 60 for i in rng.sample(range(len(s) - 40), 5)]
    items, dropped = [], Counter()

    def add(file, i, j, state, fam, q, ans_idx, tp, meta):
        items.append((file, i, j, state, fam, q, ans_idx, tp, meta))

    def load(name):
        return [json.loads(l) for l in open(src / f"{name}.jsonl")]

    for name, fam_of in [("custom_questions", lambda q: f"teacher.custom.{q['type']}"), ("routing_messages", lambda q: "teacher.routing"),
                         ("routing_terse", lambda q: "teacher.routing_terse"),
                         ("commands", lambda q: "teacher.commands." + ("risk" if q["type"] == "choice" else "scope"))]:
        for i, r in enumerate(load(name)):
            for j, q in enumerate(r["questions"]):
                if q.get("teacher_ok") is False: dropped["teacher_disagreed"] += 1; continue
                add(name, i, j, r["state"], fam_of(q), q, None, q.get("teacher_p"), {k: r.get(k) for k in ("domain", "kind", "recipe", "group")})
    for i, r in enumerate(load("situations")):
        q = {"type": "choice", "instructions": r["question"], "criteria": None, "options": r["options"], "answer": r["answer"]}
        add("situations", i, 0, r["situation"], "teacher.situations", q, int(r["answer"]), None, {"domain": r.get("domain")})

    corpus = "\n".join(norm(it[3]) for it in items)
    hit_windows = sum(1 for w in windows if w in corpus)
    recs = []
    for file, i, j, state, fam, q, ans_idx, tp, meta in items:
        st = state if isinstance(state, (str, dict, list)) else str(state)
        if isinstance(st, str) and not st.strip(): dropped["empty_state"] += 1; continue
        if norm(st) in guard: dropped["matches_td_test_or_heldout"] += 1; continue
        t, crit = q["type"], q.get("criteria")
        if file == "situations":
            cands = [{"id": str(k), "label": str(o), "description": ""} for k, o in enumerate(q["options"])]
            qd = {"type": "choice", "candidates": cands, "levels": [], "criteria": {}}; k, ans = len(cands), ans_idx
        elif t == "choice":
            if not isinstance(crit, dict) or q.get("answer") not in crit: dropped["bad_choice"] += 1; continue
            keys = list(crit)
            qd = {"type": "choice", "candidates": [{"id": kk, "label": kk, "description": "" if v is None else (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))} for kk, v in crit.items()],
                  "levels": [], "criteria": {}}
            k, ans = len(keys), keys.index(q["answer"])
        elif t == "noul":
            if not isinstance(q.get("answer"), bool): dropped["bad_noul"] += 1; continue
            bc = {kk: str(v) for kk, v in crit.items() if kk in ("true", "false") and v} if isinstance(crit, dict) else {}
            qd = {"type": "boolean", "candidates": [], "levels": [], "criteria": bc}; k, ans = 2, int(q["answer"])
        elif t == "score":
            if not isinstance(crit, list) or not isinstance(q.get("answer"), int) or not 0 <= q["answer"] < len(crit): dropped["bad_score"] += 1; continue
            qd = {"type": "score", "candidates": [], "levels": [str(x) for x in crit], "criteria": {}}; k, ans = len(crit), q["answer"]
        else:
            dropped[f"type_{t}"] += 1; continue
        if k < 2: dropped["k<2"] += 1; continue
        split = "dev" if h100(f"{file}|{norm(st)}") < 5 else "train"
        recs.append({"id": f"teacher:{file}:{i}:{j}", "source": f"Mapika/decider@{commit[:12]}:teacher_data/{file}.jsonl",
                     "source_group": f"teacher:{file}:{i}", "family": fam, "split": split, "license_class": "B", "state": st,
                     "question": {"id": f"q{j}", "instruction": q["instructions"], **qd}, "target": target(k, ans, tp),
                     "meta": {**{kk: v for kk, v in meta.items() if v is not None}, "teacher_p": tp}})
    emit(recs, Path(a.out), {"source": f"Mapika/decider@{commit} teacher_data (Apache-2.0)", "questions_in": len(items),
                             "dropped": dict(dropped), "td_test_windows_checked": len(windows), "td_test_windows_found_in_teacher": hit_windows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["td-fit", "teacher"]); ap.add_argument("--out", required=True); ap.add_argument("--src", default=".cache/decider_data")
    a = ap.parse_args()
    td_fit(a) if a.mode == "td-fit" else teacher(a)
