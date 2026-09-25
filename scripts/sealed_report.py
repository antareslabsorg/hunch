#!/usr/bin/env python3
"""The one sealed in-family test read per shipped checkpoint, beside the frozen backbone on the same questions.

    python scripts/sealed_report.py [--read <run>=<file> ...] --out docs/results/launch/sealed.json

Each read is a `hunch.evaluate --split test --max-per-family 300` output at T = 1. For every checkpoint this reports
accuracy, NLL, Brier and smooth ECE, pooled and per family, at T = 1 and at the release temperature from `derived.json`.
Accuracy does not depend on the temperature. Beside it is the frozen-backbone read of the same size
(`docs/results/frozen-test-full/`). The frozen numbers are compared question for question only if the question ids are
identical; if they are not, the script stops. The checkpoints were fixed before the read, and nothing is selected on it.
"""
import argparse, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("compare_results", ROOT / "scripts/compare_results.py")
cr = importlib.util.module_from_spec(_s); _s.loader.exec_module(cr)

READS = {"w16-06-v3dgp10-1200-s55": "docs/results/launch/sealed/w16-06-v3dgp10-1200-s55_test300.json",
         "g2-17-v4t": "docs/results/launch/sealed/g2-17-v4t_test300.json"}
SIZE = {"w16-06-v3dgp10-1200-s55": "Qwen3-0.6B", "g2-17-v4t": "Qwen3-1.7B"}
FROZEN = "docs/results/frozen-test-full/frozen_test_{}_v3d.json"


def summarise(preds: list, T: float = 1.0) -> dict:
    P, Y, qt = cr._arr([x["probs"] for x in preds]), cr._arr([x["target"] for x in preds]), [x["qtype"] for x in preds]
    P = cr.temper(P, T) if T != 1.0 else P
    nll, brier, acc = cr.metrics(P, Y)
    return {"n": len(preds), "accuracy": float(acc.mean()), "nll": float(nll.mean()), "brier": float(brier.mean()),
            "smece": float(cr.smooth_ece(*cr.calibration_pairs(P, Y, qt)))}


def report(run: str, path: Path, derived: dict) -> dict:
    d = json.loads(path.read_text())
    if d["summary"].get("split") != "test":
        raise SystemExit(f"{path}: split is {d['summary'].get('split')!r}, not test")
    preds, T = d["predictions"], derived["runs"][run]["temperature"]["T"]
    frozen = json.loads((ROOT / FROZEN.format(SIZE[run])).read_text())["predictions"]
    if sorted(x["id"] for x in preds) != sorted(x["id"] for x in frozen):
        raise SystemExit(f"{run}: the sealed read and the frozen read do not cover the same questions")
    fams = sorted({x["family"] for x in preds})
    by = lambda rows, f: [x for x in rows if x["family"] == f]
    return {"file": str(path.relative_to(ROOT)), "T": T, "families": len(fams),
            "at_T1": summarise(preds), "at_T": summarise(preds, T), "frozen_at_T1": summarise(frozen),
            "per_family": {f: {"at_T1": summarise(by(preds, f)), "at_T": summarise(by(preds, f), T),
                               "frozen_at_T1": summarise(by(frozen, f))} for f in fams}}


NAMES = {"w16-06-v3dgp10-1200-s55": "Hunch 0.6B", "g2-17-v4t": "Hunch 1.7B"}
COLS = [("at_T1", "accuracy", "accuracy"), ("frozen_at_T1", "accuracy", "frozen accuracy"), ("at_T1", "nll", "NLL, T = 1"),
        ("at_T", "nll", "NLL, release T"), ("at_T1", "smece", "smooth ECE, T = 1"), ("at_T", "smece", "smooth ECE, release T")]


def markdown(out: dict, rel: str, doc: str) -> tuple[str, list]:
    """The per-family tables for `doc`, and one claims.json entry for every number in them."""
    L, claims = [], []
    for run, r in out.items():
        L += [f"## {NAMES[run]} (`{run}`)", "", "| family | n | " + " | ".join(c[2] for c in COLS) + " |", "|---|---:|" + "---:|" * len(COLS)]
        for label, base in [("**pooled**", [run])] + [(f, [run, "per_family", f]) for f in r["per_family"]]:
            node = r if len(base) == 1 else r["per_family"][base[-1]]
            cells = [format(node[part][metric], ".4f") for part, metric, _ in COLS]
            claims += [{"doc": doc, "file": rel, "path": base + [part, metric], "fmt": ".4f"} for part, metric, _ in COLS]
            L.append(f"| {label} | {node['at_T1']['n']} | " + " | ".join(cells) + " |")
        link = "(" + "results/launch/derived.json" + ")"   # relative to docs/, where the page lives; split so no link sits in this source
        L += ["", f"Release temperature T = {r['T']:.4f}, from [`derived.json`]{link}.", ""]
        claims.append({"doc": doc, "file": rel, "path": [run, "T"], "fmt": ".4f"})
    return "\n".join(L) + "\n", claims


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", action="append", default=[], help="<run>=<evaluate output>, overriding the default path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", help="also write the per-family tables (Markdown) here, and their claims beside it as <file>.claims.json")
    ap.add_argument("--doc", default="docs/29-sealed-test.md", help="the tables' path in the launch repository, for the claims")
    a = ap.parse_args()
    reads = dict(READS, **dict(r.split("=", 1) for r in a.read))
    derived = json.loads((ROOT / "docs/results/launch/derived.json").read_text())
    out = {run: report(run, ROOT / p, derived) for run, p in reads.items()}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    if a.tables:
        md, claims = markdown(out, str(Path(a.out).resolve().relative_to(ROOT)), a.doc)
        Path(a.tables).write_text(md); Path(a.tables + ".claims.json").write_text(json.dumps(claims, indent=1) + "\n")
    for run, r in out.items():
        t1, tf, fz = r["at_T1"], r["at_T"], r["frozen_at_T1"]
        print(f"{run}: {t1['n']} questions, {r['families']} families; accuracy {t1['accuracy']:.4f} (frozen {fz['accuracy']:.4f}); "
              f"NLL {t1['nll']:.4f} at T = 1, {tf['nll']:.4f} at T = {r['T']}; smooth ECE {t1['smece']:.4f} / {tf['smece']:.4f}")
