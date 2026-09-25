#!/usr/bin/env python
"""Gate report: trained model vs. frozen baseline with paired source-group bootstrap, plus temperature scaling.

    python scripts/compare_results.py --trained results/r0-17_heldout.json --baseline results/frozen_heldout_bf16.json \
        --calibration results/r0-17_calibration.json [--out results/gate_r0-17.json]

For every family present in both files: paired differences (trained − baseline) of NLL, Brier and accuracy with 95%
bootstrap intervals over source groups (groups resampled, both models evaluated on the same questions), smooth ECE of
the trained model before and after a single temperature fitted on the calibration file (probabilities are rescaled as
p^(1/T), which equals dividing logits by T), and the sprint gates from design notes, unpublished / an internal review:
  G1 accuracy(trained) ≥ accuracy(baseline) + 0.05
  G2 NLL and Brier deltas: 95% CI entirely below zero
  G3 smECE ≤ 0.08 before calibration and ≤ 0.05 after
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hunch.metrics import calibration_pairs, smooth_ece  # noqa: E402


def load(path):
    d = json.load(open(path))
    return {p["id"]: p for p in d["predictions"]}, d["summary"]


def _arr(rows):
    k = max(len(r) for r in rows)
    out = np.zeros((len(rows), k))
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def metrics(p, t):
    nll = -(t * np.log(np.clip(p, 1e-12, 1))).sum(-1)
    brier = ((p - t) ** 2).sum(-1)
    acc = (p.argmax(-1) == t.argmax(-1)).astype(float)
    return nll, brier, acc


def temper(p: np.ndarray, T: float) -> np.ndarray:
    q = np.clip(p, 1e-12, 1) ** (1.0 / T)
    q = q * (p > 0)  # keep exact zeros (padding / impossible) at zero
    return q / q.sum(-1, keepdims=True)


def fit_temperatures(cal: dict) -> tuple[float, dict[str, float]]:
    """Returns (global_T, per_type_T) fitted on the calibration split (design notes, unpublished)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 121))
    pc = _arr([c["probs"] for c in cal.values()])
    tc = _arr([c["target"] for c in cal.values()])
    losses = [-(tc * np.log(np.clip(temper(pc, T), 1e-12, 1))).sum(-1).mean() for T in grid]
    global_T = float(grid[int(np.argmin(losses))])

    by_type = defaultdict(list)
    for c in cal.values():
        by_type[c["qtype"]].append(c)

    typed_T = {}
    for qt, rows in by_type.items():
        p = _arr([r["probs"] for r in rows])
        t = _arr([r["target"] for r in rows])
        l = [-(t * np.log(np.clip(temper(p, T), 1e-12, 1))).sum(-1).mean() for T in grid]
        typed_T[qt] = float(grid[int(np.argmin(l))])
    return global_T, typed_T


def paired_bootstrap(groups, d_nll, d_brier, d_acc, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    out = {"nll": [], "brier": [], "accuracy": []}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        out["nll"].append(d_nll[sel].mean()); out["brier"].append(d_brier[sel].mean()); out["accuracy"].append(d_acc[sel].mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}, len(uniq)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--calibration", help="trained model's predictions on the calibration split (for temperature)")
    ap.add_argument("--out")
    args = ap.parse_args()

    tr, tr_sum = load(args.trained)
    bl, _ = load(args.baseline)
    common = sorted(set(tr) & set(bl))
    if not common:
        raise SystemExit("no overlapping question ids between trained and baseline predictions")
    T, typed_T = 1.0, {}
    if args.calibration:
        cal, _ = load(args.calibration)
        T, typed_T = fit_temperatures(cal)
    # With no --calibration there is no fitted temperature, so `smece_T` is the RAW value. Recording
    # it under a label that says "fitted on the calibration split" -- which this did unconditionally
    # until 2026-09-21 -- makes an untempered number indistinguishable from a tempered one in the
    # archive, and G3 then certifies a run on a quantity the file misnames. 58 gate rows already on
    # disk carry that mislabel; they are listed in the project log (not published) rather than rewritten.
    calibrator = ("global_T_fitted_on_calibration_split" if args.calibration
                  else "NONE_no_calibration_file_T_is_1_smece_T_is_raw")

    by_fam = defaultdict(list)
    for qid in common:
        by_fam[tr[qid]["family"]].append(qid)
    report = {"trained": args.trained, "baseline": args.baseline, "temperature": T, "typed_temperature": typed_T, "n_common": len(common), "families": {}}
    for fam, qids in sorted(by_fam.items()):
        rows_t = [tr[q] for q in qids]; rows_b = [bl[q] for q in qids]
        p_t = _arr([r["probs"] for r in rows_t]); p_b = _arr([r["probs"] for r in rows_b]); tgt = _arr([r["target"] for r in rows_t])
        n_t, b_t, a_t = metrics(p_t, tgt); n_b, b_b, a_b = metrics(p_b, tgt)
        groups = np.array([r["group"] for r in rows_t])
        ci, n_groups = paired_bootstrap(groups, n_t - n_b, b_t - b_b, a_t - a_b)
        qt = [r["qtype"] for r in rows_t]
        conf, corr = calibration_pairs(p_t, tgt, qt)
        conf_T, corr_T = calibration_pairs(temper(p_t, T), tgt, qt)
        
        # Also evaluate per-type scaling
        p_typed = np.zeros_like(p_t)
        for i, qtype in enumerate(qt):
            p_typed[i : i + 1] = temper(p_t[i : i + 1], typed_T.get(qtype, T))
        conf_typed, corr_typed = calibration_pairs(p_typed, tgt, qt)

        sm_before = smooth_ece(conf, corr)
        sm_glob = smooth_ece(conf_T, corr_T)
        sm_typed = smooth_ece(conf_typed, corr_typed)
        # Docs 04 § 7: keep T = 1 if the objective does not improve
        # FIXED (external audit, 2026-09-21). This was `min(sm_before, sm_glob, sm_typed)` -- the
        # smallest of raw / global-T / per-type-T, computed ON THE HELD-OUT DATA BEING EVALUATED.
        # That picks the calibrator using the outcome it is then reported against, and the card
        # described the result as "a temperature fitted on the calibration split", which it was not.
        # The calibrator is now fixed before the held-out read: global temperature, fitted on the
        # disjoint calibration split. The alternatives are recorded for information, never selected on.
        sm_after = sm_glob

        conf_b, corr_b = calibration_pairs(p_b, tgt, qt)
        fam_rep = {
            "n": len(qids), "n_groups": n_groups,
            "trained": {"nll": float(n_t.mean()), "brier": float(b_t.mean()), "accuracy": float(a_t.mean()), "smece": float(sm_before), "smece_T": float(sm_after), "smece_T_pertype_unused": float(sm_typed), "smece_raw_unused": float(sm_before), "calibrator": calibrator},
            "baseline": {"nll": float(n_b.mean()), "brier": float(b_b.mean()), "accuracy": float(a_b.mean()), "smece": float(smooth_ece(conf_b, corr_b))},
            "delta": {"nll": float((n_t - n_b).mean()), "brier": float((b_t - b_b).mean()), "accuracy": float((a_t - a_b).mean())},
            "delta_ci95": ci,
            "G3_smece_why": (None if args.calibration else
                             "no --calibration given: no temperature was fitted, so the "
                             "post-temperature half of G3 could not be evaluated"),
            "gates": {
                "G1_acc_plus_5pts": bool(a_t.mean() >= a_b.mean() + 0.05),
                "G2_nll_brier_ci_below_zero": bool(ci["nll"][1] < 0 and ci["brier"][1] < 0),
                # G3 is defined as "<= 0.08 raw AND <= 0.05 after a fitted temperature". Without a
                # calibration file the second half is unevaluated, so the gate is None (unknown), not
                # False and certainly not True on the raw value standing in for the tempered one.
                "G3_smece": (bool(sm_before <= 0.08 and sm_after <= 0.05) if args.calibration else None),

            },
        }
        report["families"][fam] = fam_rep
    fams = report["families"]
    report["all_gates_pass_every_family"] = all(all(f["gates"].values()) for f in fams.values())
    # A None gate is UNEVALUATED. Summing it as 0 would report "failed" for something never tested,
    # and bool-summing would crash; both hide the distinction that matters. Count and report it.
    report["gate_pass_counts"] = {
        g: sum(1 for f in fams.values() if f["gates"][g] is True)
        for g in next(iter(fams.values()))["gates"]}
    report["gate_unevaluated_counts"] = {
        g: sum(1 for f in fams.values() if f["gates"][g] is None)
        for g in next(iter(fams.values()))["gates"]}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"temperature T={T:.3f} (global), typed={typed_T} (fitted on calibration split)" if args.calibration else "no calibration file: T=1")
    print(f"{'family':40s} {'n':>4s} {'acc_tr':>7s} {'acc_bl':>7s} {'dNLL':>7s} {'CI':>16s} {'dBrier':>7s} {'CI':>16s} {'smECE':>6s} {'smECE_T':>7s} gates")
    for fam, f in fams.items():
        c = f["delta_ci95"]
        print(f"{fam:40s} {f['n']:4d} {f['trained']['accuracy']:7.3f} {f['baseline']['accuracy']:7.3f} {f['delta']['nll']:7.3f} [{c['nll'][0]:6.2f},{c['nll'][1]:6.2f}] "
              f"{f['delta']['brier']:7.3f} [{c['brier'][0]:6.2f},{c['brier'][1]:6.2f}] {f['trained']['smece']:6.3f} {f['trained']['smece_T']:7.3f} "
              + " ".join(k.split('_')[0] + ('✓' if v else '✗') for k, v in f["gates"].items()))
    print("gate pass counts:", report["gate_pass_counts"], "| all families pass:", report["all_gates_pass_every_family"])


if __name__ == "__main__":
    main()
