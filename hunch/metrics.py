"""Evaluation metrics (design notes, unpublished), computed from per-question predictions.

Input: lists of (probs, target_probs, hard, group, family, qtype). Everything is numpy; no torch needed.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def _pad(rows: list[list[float]]) -> np.ndarray:
    k = max(len(r) for r in rows)
    out = np.zeros((len(rows), k))
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def nll(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -(t * np.log(np.clip(p, 1e-12, 1.0))).sum(-1)


def brier(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    return ((p - t) ** 2).sum(-1)


def accuracy(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (p.argmax(-1) == t.argmax(-1)).astype(float)


def ece_binned(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def smooth_ece(conf: np.ndarray, correct: np.ndarray, sigma: float = 0.05) -> float:
    """Kernel-smoothed calibration error with a reflected Gaussian kernel (a fixed-bandwidth variant of smECE,
    Błasiok & Nakkiran 2023). Reports E_x |E[correct | conf ~ x] - x| under the smoothed conditional expectation."""
    if len(conf) == 0:
        return float("nan")
    grid = np.linspace(0, 1, 201)
    d = grid[:, None] - conf[None, :]
    w = np.exp(-0.5 * (d / sigma) ** 2) + np.exp(-0.5 * ((grid[:, None] + conf[None, :]) / sigma) ** 2) + np.exp(-0.5 * ((2 - grid[:, None] - conf[None, :]) / sigma) ** 2)
    density = w.sum(1)
    cond = (w * correct[None, :]).sum(1) / np.maximum(density, 1e-12)
    mean_conf = (w * conf[None, :]).sum(1) / np.maximum(density, 1e-12)
    weights = density / density.sum()
    return float((weights * np.abs(cond - mean_conf)).sum())


def calibration_pairs(p: np.ndarray, t: np.ndarray, qtype: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean: (p_true, t_true). Others: top-label (max prob, target mass on argmax)."""
    conf, corr = [], []
    for i, qt in enumerate(qtype):
        if qt == "boolean":
            conf.append(p[i, 1]); corr.append(t[i, 1])
        else:
            j = int(p[i].argmax()); conf.append(p[i, j]); corr.append(t[i, j])
    return np.array(conf), np.array(corr)


def summarize(preds: list[dict], base_rates: dict[str, np.ndarray] | None = None, n_boot: int = 500, seed: int = 0) -> dict:
    """preds: dicts with keys probs, target, group, family, qtype (and optional 'known_q' bool).
    Returns overall and per-family metrics with group-bootstrap 95% CIs for NLL/Brier/accuracy."""
    p = _pad([d["probs"] for d in preds]); t = _pad([d["target"] for d in preds])
    fam = np.array([d["family"] for d in preds]); grp = np.array([d["group"] for d in preds]); qt = [d["qtype"] for d in preds]
    out = {"n": len(preds), "families": {}}
    rng = np.random.default_rng(seed)

    def block(mask: np.ndarray, fam_name: str | None) -> dict:
        pm, tm = p[mask], t[mask]
        nl, br, acc = nll(pm, tm), brier(pm, tm), accuracy(pm, tm)
        conf, corr = calibration_pairs(pm, tm, [q for q, m in zip(qt, mask) if m])
        res = {"n": int(mask.sum()), "nll": float(nl.mean()), "brier": float(br.mean()), "accuracy": float(acc.mean()),
               "ece15": ece_binned(conf, corr), "smece": smooth_ece(conf, corr),
               "mean_entropy": float((-(pm * np.log(np.clip(pm, 1e-12, 1))).sum(-1)).mean())}
        if base_rates:
            # base-rate predictor per (family, K): the mean training target vector for questions of that shape
            base_b, model_b = [], []
            for row, (pm_i, tm_i) in zip(np.where(mask)[0], zip(pm, tm)):
                k = len(preds[row]["probs"])
                base = base_rates.get((preds[row]["family"], k))
                if base is None:
                    continue
                base_b.append(((base - tm_i[:k]) ** 2).sum())
                model_b.append(((pm_i[:k] - tm_i[:k]) ** 2).sum())
            if base_b:
                res["brier_skill"] = float(1 - np.mean(model_b) / max(np.mean(base_b), 1e-12))
                res["brier_skill_coverage"] = len(base_b) / int(mask.sum())
        known = np.array([bool(d.get("known_q", False)) for d in preds])[mask]
        if known.any():
            res["known_q_l2"] = float(((pm[known] - tm[known]) ** 2).sum(-1).mean())
        # group bootstrap
        groups = grp[mask]; uniq = np.unique(groups)
        if len(uniq) >= 5 and n_boot:
            gi = {g: np.where(groups == g)[0] for g in uniq}
            stats = {"nll": [], "brier": [], "accuracy": []}
            for _ in range(n_boot):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([gi[g] for g in pick])
                stats["nll"].append(nl[idx].mean()); stats["brier"].append(br[idx].mean()); stats["accuracy"].append(acc[idx].mean())
            res["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in stats.items()}
            res["n_groups"] = int(len(uniq))
        return res

    out["overall"] = block(np.ones(len(preds), dtype=bool), None)
    for f in sorted(set(fam)):
        out["families"][f] = block(fam == f, f)
    return out


def base_rates_from_records(records) -> dict[tuple[str, int], np.ndarray]:
    """Mean target vector per (family, K) over the training split: the 'always predict the base rate' predictor."""
    sums: dict[tuple[str, int], np.ndarray] = {}
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for r in records:
        key = (r.family, len(r.target.probs))
        v = np.asarray(r.target.probs, dtype=float)
        sums[key] = sums[key] + v if key in sums else v.copy()
        counts[key] += 1
    return {k: sums[k] / counts[k] for k in sums}


if __name__ == "__main__":  # note: sanity
    rng = np.random.default_rng(0)
    preds = []
    for i in range(400):
        q = rng.dirichlet([1, 1, 1]); y = rng.choice(3, p=q)
        preds.append({"probs": q.tolist(), "target": [float(j == y) for j in range(3)], "group": f"g{i // 4}", "family": "f", "qtype": "choice"})
    s = summarize(preds, base_rates={("f", 3): np.array([1 / 3] * 3)}, n_boot=100)
    assert 0 < s["overall"]["nll"] < 2 and 0 <= s["overall"]["smece"] < 0.15, s["overall"]
    assert "brier_skill" in s["families"]["f"] and s["families"]["f"]["brier_skill"] > 0, s["families"]["f"]
    perfect = smooth_ece(np.array([0.2] * 500 + [0.8] * 500), np.array([0] * 400 + [1] * 100 + [0] * 100 + [1] * 400))
    assert perfect < 0.03, perfect
    print("metrics ok", {k: round(v, 3) for k, v in s["overall"].items() if isinstance(v, float)})
