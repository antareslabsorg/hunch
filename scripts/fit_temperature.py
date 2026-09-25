#!/usr/bin/env python3
"""Fit the release temperature of a checkpoint and write the hunch_config.json that `hunch.infer.Hunch.load` reads.

    python scripts/fit_temperature.py --calibration results/calibration.json --dev results/dev.json [--write-config runs/<run>/best]

Both inputs are `hunch.evaluate` outputs (T = 1) of the same checkpoint. The temperature is fitted on the calibration split
alone (`compare_results.fit_temperatures`: one global T on a 121-point log grid, 0.25–4); dev, which is disjoint, reports the
smooth ECE before and after. Without `--write-config`, `Hunch.load` uses T = 1.
"""
import argparse, importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_s = importlib.util.spec_from_file_location("compare_results", ROOT / "scripts/compare_results.py")
cr = importlib.util.module_from_spec(_s); _s.loader.exec_module(cr)


def fit(calibration: Path, dev: Path) -> dict:
    c, _ = cr.load(str(calibration)); T, _ = cr.fit_temperatures(c)
    dv = json.loads(Path(dev).read_text())["predictions"]
    P, Y, qt = cr._arr([x["probs"] for x in dv]), cr._arr([x["target"] for x in dv]), [x["qtype"] for x in dv]
    return {"T": float(T), "dev_smece_T1": float(cr.smooth_ece(*cr.calibration_pairs(P, Y, qt))),
            "dev_smece_at_T": float(cr.smooth_ece(*cr.calibration_pairs(cr.temper(P, T), Y, qt)))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, required=True); ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--write-config", type=Path, help="checkpoint directory to write hunch_config.json into")
    a = ap.parse_args()
    r = fit(a.calibration, a.dev)
    print(f"T = {r['T']:.4f}; dev smooth ECE {r['dev_smece_T1']:.4f} at T = 1, {r['dev_smece_at_T']:.4f} at T")
    if a.write_config:
        from hunch.serialize import SERIALIZATION_VERSION
        cfg = a.write_config / "hunch_config.json"
        cfg.write_text(json.dumps({"temperature": round(r["T"], 4), "temperature_out_of_family": 1.0,
                                   "serialization": SERIALIZATION_VERSION}, indent=1) + "\n")
        print(f"wrote {cfg}")
