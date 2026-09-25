#!/usr/bin/env python3
"""Compute every derived number the launch documents quote, from the result files, into one file.

    python scripts/launch_derived.py            # writes docs/results/launch/derived.json

Derived means not stored as a single field in any one result file: the fitted temperature and dev smooth ECE at it (the
release's own fit, `compare_results.fit_temperatures`), pooled probe rates (n-weighted over the three held-out families),
case-level bootstrap intervals on typed-decisions (`td_bootstrap.analyse`, 10,000 draws, seed 0) and paired differences,
and Laya-list summaries against a reference checkpoint. An interval is computed from per-decision rows only when those rows
reproduce the accuracy of the summary file the documents cite; otherwise this script stops. `scripts/verify_claims.py` then checks the documents against this file like any other.
"""
import importlib.util, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs/results"
OUT = RES / "launch/derived.json"
_s = importlib.util.spec_from_file_location("compare_results", ROOT / "scripts/compare_results.py")
cr = importlib.util.module_from_spec(_s); _s.loader.exec_module(cr)
_b = importlib.util.spec_from_file_location("td_bootstrap", ROOT / "scripts/td_bootstrap.py")
tb = importlib.util.module_from_spec(_b); _b.loader.exec_module(tb)
_f = importlib.util.spec_from_file_location("fit_temperature", ROOT / "scripts/fit_temperature.py")
ft = importlib.util.module_from_spec(_f); _f.loader.exec_module(ft)

# the two release checkpoints -> where their files live; laya_ref is the checkpoint a Laya list is compared with. Every input
# ships in the launch repository, so this script run there reproduces the shipped derived.json exactly.
RUNS = {
    "w16-06-v3dgp10-1200-s55": {"dir": "day6", "robust": "day6/robust", "laya": "day7/laya", "td": "day6/td_zeroshot_w16-06-v3dgp10-1200-s55",
                                "td_items": "day7/v1/rescore/td_w16-06-v3dgp10-1200-s55"},
    "g2-17-v4t": {"dir": "day7/v1", "robust": "day7/v1/robust", "laya": "day7/v1/laya", "td": "day7/v1/td_g2-17-v4t",
                  "td_items": "day7/v1/rescore/td_g2-17-v4t", "laya_ref": "h200-17-fullk-v3d"},
}


def temperature(run, d):
    cal, dev = RES / d / f"{run}_calibration.json", RES / d / f"{run}_dev.json"
    return ft.fit(cal, dev) if cal.exists() and dev.exists() else {}   # the same fit a user runs on a retrained checkpoint


def pooled(path, key):
    if not path.exists(): return {}
    d = json.loads(path.read_text()); t, fams = d["table"], d["families"]
    out = {}
    for v in sorted({v for f in fams for v in t[f]}):
        rows = [t[f][v] for f in fams if v in t[f]]; n = sum(r["n"] for r in rows)
        out[v] = sum(r[key] * r["n"] for r in rows) / n
    return out


def laya(run, spec):
    ref = spec.get("laya_ref")
    a, b = RES / spec["laya"] / f"laya_list_{run}.json", RES / "day7/laya" / f"laya_list_{ref}.json" if ref else None
    if not ref or not a.exists() or not b.exists(): return {}
    A, B = json.loads(a.read_text())["suites"], json.loads(b.read_text())["suites"]
    d = [A[k]["raw"]["accuracy"] - B[k]["raw"]["accuracy"] for k in A if k in B]
    return {"reference": ref, "suites": len(d), "better": sum(x > 0.01 for x in d), "worse": sum(x < -0.01 for x in d),
            "flat": sum(abs(x) <= 0.01 for x in d), "median_delta": statistics.median(d), "mean_delta": statistics.mean(d)}


# the open Deciders, typed-decisions only; their per-decision rows come from a re-score that reproduced the first read
TD_ONLY = {"decider-2b": {"td": "day7/v1/td_decider-2b", "td_items": "day7/v1/rescore/td_decider-2b"},
           "decider-0.8b": {"td": "day7/v1/td_decider-0.8b", "td_items": "day7/v1/rescore/td_decider-0.8b"}}
PAIRED = [("g2-17-v4t", "decider-2b"), ("g2-17-v4t", "decider-0.8b")]
# the 0.6B candidate that did not replace the 0.6B release, on the Laya list (its card quotes the medians)
LAYA_ONLY = {"g1-06-v4t": {"laya": "day7/v1/laya", "laya_ref": "w16-06-v3dgp10-1200-s55"},
             "g1b-06-v4t-s1": {"laya": "day7/v1/laya", "laya_ref": "w16-06-v3dgp10-1200-s55"},
             "g2b-17-v4t-s1": {"laya": "day7/v1/laya", "laya_ref": "h200-17-fullk-v3d"}}   # the 1.7B release's replication seed


def td_items(spec):
    """Per-decision rows for a run, checked against the summary the documents cite; None when the rows do not exist."""
    if not spec.get("td"): return None
    items = RES / (spec.get("td_items", spec["td"]) + ".items.jsonl")
    if not items.exists(): return None
    rows, cited = tb.load(items), json.loads((RES / (spec["td"] + ".json")).read_text())["overall"]
    acc = sum(r["correct"] for r in rows.values()) / len(rows)
    if len(rows) != cited["n"] or abs(acc - cited["accuracy"]) > 1e-9:
        raise SystemExit(f"{items.relative_to(ROOT)}: {len(rows)} rows, accuracy {acc} does not reproduce {spec['td']}.json "
                         f"({cited['n']}, {cited['accuracy']}); no interval from it")
    return rows


def td_ci(spec):
    rows = td_items(spec)
    if rows is None: return {}
    r = tb.analyse(rows)
    return {"accuracy_ci": list(map(float, (r["ci"][0][0], r["ci"][1][0]))), "kl_ci": list(map(float, (r["ci"][0][1], r["ci"][1][1])))}


def paired(a, b, specs):
    A, B = td_items(specs[a]), td_items(specs[b])
    if A is None or B is None: return {}
    r = tb.analyse(A, B)
    return {"decisions": r["decisions"], **{m: {"diff": float(r["diff"][j]), "ci": [float(r["diff_ci"][0][j]), float(r["diff_ci"][1][j])]}
                                            for j, m in enumerate(("accuracy", "kl", "brier"))}}


def without_exact(ids, key, preds, expect=None):
    """Accuracy with and without the questions whose state text also occurs in training (their exact matches, `ids[key]`:
    docs/21's for dev, scripts/sealed_contamination.py's for the sealed test read). `expect` is the accuracy the documents
    already cite for all of `preds`; the rows must reproduce it, or this script stops."""
    if not ids.exists() or not preds.exists(): return {}
    exact = set(json.loads(ids.read_text())[key]["exact_ids"])
    rows = json.loads(preds.read_text())["predictions"]
    ok = lambda r: max(range(len(r["probs"])), key=lambda i: r["probs"][i]) == (r["target"] if isinstance(r["target"], int) else max(range(len(r["target"])), key=lambda i: r["target"][i]))
    keep = [r for r in rows if r["id"] not in exact]
    acc = sum(map(ok, rows)) / len(rows)
    if expect is not None and abs(acc - expect) > 1e-12:
        raise SystemExit(f"{preds.name}: rows give accuracy {acc}, the cited file {expect}")
    return {"n": len(rows), "n_exact_in_training": len(rows) - len(keep), "accuracy": acc,
            "accuracy_without_exact": sum(map(ok, keep)) / len(keep)}


# pooled held-out NLL over the families whose interface none of these checkpoints trained (Circa's interface is in the 1.7B's
# data, not its parent's): the question-weighted mean over GoEmotions and CaseHOLD, per checkpoint the 1.7B card compares
UNTRAINED_HELDOUT = {"g2-17-v4t": "day7/v1/g2-17-v4t_heldout2000.json", "g2b-17-v4t-s1": "day7/v1/g2b-17-v4t-s1_heldout2000.json",
                     "h200-17-fullk-v3d": "h200-full/results/h200-17-fullk-v3d_heldout2000.json"}


def heldout_without_circa(path):
    fam = json.loads((RES / path).read_text())["summary"]["families"]
    keep = {f: v for f, v in fam.items() if f != "heldout.pragmatics.circa"}
    n = sum(v["n"] for v in keep.values())
    return {"families": sorted(keep), "n": n, "nll": sum(v["nll"] * v["n"] for v in keep.values()) / n,
            "accuracy": sum(v["accuracy"] * v["n"] for v in keep.values()) / n}


def score_chance(items):
    """Chance accuracy on typed-decisions' ordinal score questions: the mean of 1/levels over them (the questions are the suite's)."""
    rows = [json.loads(l) for l in (RES / items).read_text().splitlines() if l.strip()]
    sc = [r for r in rows if r["type"] == "score"]
    levels = {}
    for r in sc:
        levels[str(len(r["p"]))] = levels.get(str(len(r["p"])), 0) + 1
    return {"decisions": len(sc), "levels": dict(sorted(levels.items())), "chance_accuracy": sum(1 / len(r["p"]) for r in sc) / len(sc)}


FLOOR = {"w16-06-v3dgp10-1200-s55": ("hunch/formats/out/equivalence-0.6b.json", "pytorch-fp32-cpu vs archive"),
         "g2-17-v4t": ("hunch/formats/out/equivalence-1.7b.json", "pytorch-fp32-cuda vs archive")}


def fast_engine(run):
    """hunch/fast.py against the reference on one RTX 5090 (launch/fast), by the gates declared on 2026-09-25: (A) in fp32, no
    held-out answer changed and largest TV below 1e-4; (B) in bf16, against the fp32 reference, the on-device builds' floor rule
    (FORMATS.md). A speed ratio is computed only for a precision whose gate passed, so a failed one cannot be quoted."""
    f = lambda name: json.loads((RES / f"launch/fast/{name}_{run}.json").read_text())
    path, key = FLOOR[run]
    floor = json.loads((ROOT / path).read_text())["noise_floor"][key]
    a, b = f("equivA")["heldout"], f("equivB")["heldout"]
    n, d, fn, fd = b["n"], b["answers_changed"], floor["n"], floor["n_argmax_diff"]
    pool = (d + fd) / (n + fn)
    z = (d / n - fd / fn) / (pool * (1 - pool) * (1 / n + 1 / fn)) ** 0.5   # as floor_gate in hunch/formats/equivalence_test.py
    gate = {"fp32": a["answers_changed"] == 0 and a["tv_max"] < 1e-4, "bf16": b["tv_max"] <= floor["tv_max"] and z <= 1.645}
    jev = json.loads((RES / "launch/jev/jev_typed_decisions.json").read_text())["latency_ms"]["p50"]
    ref = f("latency_ref")["ms_per_request"]["p50"]
    out = {"gate_fp32_exact": gate["fp32"], "gate_bf16_floor": gate["bf16"], "bf16_z": z, "bf16_x_floor_tv": b["tv_max"] / floor["tv_max"],
           "reference_bf16_p50_ms": ref}
    for amp, name in (("fp32", "latency_fast_fp32"), ("bf16", "latency_fast")):
        p50 = f(name)["ms_per_request"]["p50"]
        out[amp] = {"p50_ms": p50, "x_faster_than_reference_bf16": ref / p50 if gate[amp] else None,
                    "x_below_jev_round_trip": jev / p50 if gate[amp] else None}
    # the MLX f16 build on the laptop (declared the same day): exact against the MLX reference read, then timed like its table
    m, lat = f("mlx_equiv"), f("latency_mlx_fast")
    ok = m["vs_mlx_reference"]["answers_changed"] == 0 and m["vs_mlx_reference"]["tv_max"] < 1e-4
    ref_l = json.loads((RES / f"day7/v1/latency/latency_mlx_{run}.json").read_text())["ms_per_request"]["p50"]
    p50 = lat["ms_per_request"]["p50"]
    out["mlx_laptop"] = {"gate_exact": ok, "p50_ms": p50, "reference_p50_ms": ref_l, "x_faster_than_reference": ref_l / p50 if ok else None,
                         "jev_round_trip_over_hunch": jev / p50 if ok else None}
    return out


def main():
    out = {"method": __doc__.strip().splitlines()[0], "runs": {}}
    out["fast_engine"] = {run: fast_engine(run) for run in RUNS}
    sealed = json.loads((RES / "launch/sealed.json").read_text())
    out["typed_decisions_score_chance"] = score_chance("day7/v1/rescore/td_w16-06-v3dgp10-1200-s55.items.jsonl")
    out["heldout_without_circa"] = {run: heldout_without_circa(path) for run, path in UNTRAINED_HELDOUT.items()}
    for run, spec in RUNS.items():
        out["runs"][run] = {"temperature": temperature(run, spec["dir"]),
                            "injection_flip_to_target": pooled(RES / spec["robust"] / f"injection_{run}.json", "flip_to_target_rate"),
                            "jaggedness_flip_rate": pooled(RES / spec["robust"] / f"jaggedness_{run}.json", "flip_rate"),
                            "laya_vs_reference": laya(run, spec), "typed_decisions": td_ci(spec),
                            "dev_contamination": without_exact(RES / "launch/contamination_ids.json", "dev", RES / spec["dir"] / f"{run}_dev.json"),
                            "sealed_contamination": without_exact(RES / "launch/sealed_contamination_ids.json", "test_sealed_read",
                                                                  RES / f"launch/sealed/{run}_test300.json", sealed[run]["at_T1"]["accuracy"])}
    for run, spec in TD_ONLY.items():
        out["runs"][run] = {"typed_decisions": td_ci(spec)}
    out["typed_decisions_paired"] = {f"{a} - {b}": paired(a, b, {**RUNS, **TD_ONLY}) for a, b in PAIRED}
    out["laya_candidates"] = {run: laya(run, spec) for run, spec in LAYA_ONLY.items()}
    ids = RES / "launch/contamination_ids.json"
    if ids.exists():   # exact state matches with the releases' training text, per evaluation set (docs/21)
        out["contamination_exact_matches"] = {k: len(v["exact_ids"]) for k, v in json.loads(ids.read_text()).items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for run, r in out["runs"].items():
        print(run, {k: (round(v["T"], 4) if k == "temperature" and v else ("…" if v else "—")) for k, v in r.items()})


if __name__ == "__main__":
    main()
