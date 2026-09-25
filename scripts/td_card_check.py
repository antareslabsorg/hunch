#!/usr/bin/env python3
"""The typed-decisions card's published numbers, read from the card itself, and a harness check against its "Prior" row.

    python scripts/td_card_check.py            # writes docs/results/launch/td_card.json

The leaderboard and reference tables are parsed from the card (README.md) at the pinned revision, so the third-party rows the
benchmark page quotes come from a file anyone can regenerate, not from a transcription.

The prior predicts, for every test question, the training split's summed gold probability per label of every question with
that name (pooled over workflows); it never looks at the input. If our KL, Brier and label alignment are right, this reproduces
the card's published Prior row (0.470 / 0.347 / 0.189) up to rounding. (Keeping workflows apart gives a stronger prior,
0.4785 / 0.327 / 0.181, which is not the card's row.)
"""
import hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_typed_decisions import REVISION, gold_vector  # noqa: E402


def order(g: dict) -> list:
    return ["false", "true"] if g["type"] == "noul" else sorted(g["probabilities"], key=int if g["type"] == "score" else str)


def card_tables(text: str) -> dict:
    """Rows of the card's markdown tables that carry an accuracy column, keyed by the model or reference name."""
    out, head = {}, None
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else None
        if not cells: head = None; continue
        if head is None: head = [re.sub(r"[^a-z]+", "_", c.lower()).strip("_") for c in cells]; continue
        if set(line) <= set("|-: "): continue
        row = dict(zip(head, cells))
        name = re.sub(r"\*|\[|\]\([^)]*\)|`[^`]*`", "", row.get("model") or row.get("reference") or "").replace("]", "")
        name = re.sub(r"\(\s*\)", "", name).strip()
        num = lambda k: float(row[k].strip("*")) if re.fullmatch(r"\**[\d.]+\**", row.get(k, "")) else None
        acc = num("accuracy") if "accuracy" in row else num("accuracy_")
        if name and acc is not None:
            out[name] = {"kind": row.get("kind") or row.get("what_it_is"), "accuracy": acc,
                         "kl": num("kl_from_gold") if "kl_from_gold" in row else None, "brier": num("brier") if "brier" in row else None}
    return out


def main() -> None:
    readme = Path(hf_hub_download("LocalLLaMA/typed-decisions", "README.md", repo_type="dataset", revision=REVISION)).read_text()
    load = lambda s: pd.read_parquet(hf_hub_download("LocalLLaMA/typed-decisions", f"all/{s}-00000-of-00001.parquet",
                                                     repo_type="dataset", revision=REVISION))
    sums = defaultdict(lambda: defaultdict(float))   # question name -> label -> summed gold probability
    for _, r in load("train").iterrows():
        for name, g in json.loads(r["gold"]).items():
            for k, v in g["probabilities"].items(): sums[name][k] += float(v)
    rows = []
    for _, r in load("test").iterrows():
        for name, g in json.loads(r["gold"]).items():
            o, q = order(g), gold_vector(g, order(g))
            c = np.array([sums[name].get(k, 0.0) for k in o]); p = c / c.sum()
            rows.append((float(o[int(np.argmax(p))] == str(g["label"])),
                         float((q * (np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1)))).sum()),
                         float(((p - q) ** 2).sum())))
    a = np.array(rows)
    out = {"dataset": "LocalLLaMA/typed-decisions", "revision": REVISION, "card_sha256": hashlib.sha256(readme.encode()).hexdigest(),
           "card": card_tables(readme), "n_decisions": len(rows),
           "prior_from_train": {"accuracy": a[:, 0].mean(), "kl": a[:, 1].mean(), "brier": a[:, 2].mean()}}
    dst = ROOT / "docs/results/launch/td_card.json"
    dst.write_text(json.dumps(out, indent=1) + "\n")
    print(dst.relative_to(ROOT), out["prior_from_train"])


if __name__ == "__main__":
    main()
