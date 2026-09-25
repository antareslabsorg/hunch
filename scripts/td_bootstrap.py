#!/usr/bin/env python3
"""Case-level bootstrap intervals for typed-decisions reads, and paired differences between two models.

    python scripts/td_bootstrap.py A.items.jsonl [B.items.jsonl] [--n 10000] [--seed 0]
    python scripts/td_bootstrap.py --selftest

Input: the `*.items.jsonl` rows written beside each summary by scripts/bench_typed_decisions.py / bench_td_decider.py.
The unit resampled is the case (each carries its five decisions), so within-case correlation stays in the interval.
Prints accuracy / KL / Brier for A with 95 % percentile intervals; with B, the paired difference A - B over the decisions
both answered, its interval, and the share of resamples in which A is not better (a one-sided bootstrap p).
"""
import argparse, json
import numpy as np

M = ("correct", "kl", "brier")


def load(path):
    rows = (json.loads(l) for l in open(path))
    return {(r["case"], r["q"]): r for r in rows if "correct" in r}


def analyse(A, B=None, n=10000, seed=0):
    keys = sorted(A.keys() & B.keys()) if B else sorted(A)
    cases = sorted({k[0] for k in keys}); idx = {c: i for i, c in enumerate(cases)}
    S, D, cnt = np.zeros((len(cases), 3)), np.zeros((len(cases), 3)), np.zeros(len(cases))
    for k in keys:
        i = idx[k[0]]; cnt[i] += 1
        S[i] += [A[k][m] for m in M]
        if B: D[i] += [A[k][m] - B[k][m] for m in M]
    draws = np.random.default_rng(seed).integers(0, len(cases), size=(n, len(cases)))
    denom = cnt[draws].sum(1)[:, None]
    out = {"decisions": int(cnt.sum()), "cases": len(cases), "point": S.sum(0) / cnt.sum(),
           "ci": np.percentile(S[draws].sum(1) / denom, [2.5, 97.5], axis=0)}
    if B:
        bd = D[draws].sum(1) / denom
        out.update(diff=D.sum(0) / cnt.sum(), diff_ci=np.percentile(bd, [2.5, 97.5], axis=0),
                   p_not_better=[float((bd[:, 0] <= 0).mean()), float((bd[:, 1] >= 0).mean()), float((bd[:, 2] >= 0).mean())])
    return out


def selftest():
    rng = np.random.default_rng(1)
    A = {(f"c{i}", f"q{j}"): {"correct": float(rng.random() < .7), "kl": float(rng.random()), "brier": float(rng.random())}
         for i in range(50) for j in range(5)}
    r = analyse(A, A, n=500)
    assert np.allclose(r["point"], [np.mean([v[m] for v in A.values()]) for m in M])     # point = plain mean
    assert np.allclose(r["diff"], 0) and np.allclose(r["diff_ci"], 0)                     # a model against itself
    assert all(lo <= p <= hi for lo, p, hi in zip(r["ci"][0], r["point"], r["ci"][1]))
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?"); ap.add_argument("b", nargs="?")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    x = ap.parse_args()
    if x.selftest or not x.a:
        selftest(); raise SystemExit
    r = analyse(load(x.a), load(x.b) if x.b else None, x.n, x.seed)
    print(f"A = {x.a}: {r['decisions']} decisions in {r['cases']} cases, {x.n} case resamples")
    for j, m in enumerate(("accuracy", "KL", "Brier")):
        print(f"  {m:8s} {r['point'][j]:.4f}  95% [{r['ci'][0][j]:.4f}, {r['ci'][1][j]:.4f}]")
    if x.b:
        print(f"A - B, B = {x.b} (paired on the same decisions):")
        for j, m in enumerate(("accuracy", "KL", "Brier")):
            print(f"  {m:8s} {r['diff'][j]:+.4f}  95% [{r['diff_ci'][0][j]:+.4f}, {r['diff_ci'][1][j]:+.4f}]  "
                  f"share of resamples where A is not better: {r['p_not_better'][j]:.4f}")
