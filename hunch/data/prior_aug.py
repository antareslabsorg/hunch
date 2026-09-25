"""`prior` arm: teach the scorer that an uninformative state deserves the family's base rate.

Out-of-family over-confidence is the sprint's failing gate: on families the model never saw, its probabilities
are far sharper than its accuracy. Nothing in the current recipe ever shows the model an input that carries no
evidence for the answer. This augmentation does exactly that: for a deterministic fraction of training questions
the state is replaced by (a) another question's state from the same family or (b) a truncated/shuffled version of
its own state, and the target becomes the family's mean target vector restricted to the kept candidate subset.

If the model learns "evidence missing -> predict the prior", out-of-family calibration should improve without
costing in-family accuracy; if it learns nothing, this arm is indistinguishable from the control.

    python -m hunch.data.prior_aug --base data/sprint_v3dg --out data/sprint_v3dgp10 --frac 0.10   # the release mixture
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np

from hunch.schema import Record, Target, read_jsonl, write_jsonl
from .manifest import manifest_digest

SALT = "prior-aug-2026-09-20"
COPY = ("dev", "calibration", "heldout")


def _h(*parts) -> bytes:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()


def _uniform(rec_id: str, frac: float) -> bool:
    """Deterministic per-record coin flip (the id must be in the hash, or every record gets the same verdict)."""
    return int.from_bytes(_h(SALT, "pick", frac, rec_id)[:4], "big") / 2**32 < frac


def corrupt_state(state, rng: random.Random, donor):
    """Two deterministic corruptions: a donor state from the same family, or a truncated/shuffled own state."""
    if donor is not None and rng.random() < 0.5:
        return donor
    text = json.dumps(state, ensure_ascii=False) if not isinstance(state, str) else state
    words = text.split()
    if len(words) > 8:
        cut = max(4, len(words) // 5)
        words = words[:cut]
    rng.shuffle(words)
    return " ".join(words)


def build(base: Path, out: Path, frac: float, seed: int = 11) -> None:
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    manifest = json.loads((base / "manifest.json").read_text())
    for split in ("train",) + COPY:
        expected = manifest_digest(manifest["files"][split])
        if hashlib.sha256((base / f"{split}.jsonl").read_bytes()).hexdigest() != expected:
            raise ValueError(f"{split}: base differs from its manifest")

    rows = list(read_jsonl(base / "train.jsonl"))
    by_key: dict[tuple[str, int], list[Record]] = defaultdict(list)
    for r in rows:
        by_key[(r.family, r.question.k)].append(r)   # families can carry several K (e.g. bitext 11 and 27->16)
    priors: dict[tuple[str, int], np.ndarray] = {}
    for key, group in by_key.items():
        if len(group) >= 2:
            priors[key] = np.mean(np.array([r.target.probs for r in group]), axis=0)

    rng = random.Random(seed)
    out_rows, n_prior = [], 0
    for i, r in enumerate(rows):
        if not _uniform(r.id, frac):
            out_rows.append(r)
            continue
        key = (r.family, r.question.k)
        if key not in priors:
            out_rows.append(r)   # too few questions of this shape to define a prior: leave it untouched
            continue
        donors = by_key[key]
        donor = donors[rng.randrange(len(donors))]
        state = corrupt_state(r.state, rng, donor.state if donor is not r else None)
        prior = priors[key]
        probs = [float(p) for p in prior]
        probs[-1] = 1.0 - sum(probs[:-1])
        target = Target(r.target.semantic_kind, r.target.provenance, probs, None)
        meta = {**r.meta, "prior_aug": {"salt": SALT, "frac": frac, "seed": seed, "source": "(family, K) base rate",
                                        "corruption": "donor-or-truncated", "donor": donor.id}}
        new = replace(r, state=state, target=target, meta=meta)
        new.validate()
        out_rows.append(new)
        n_prior += 1
    if n_prior == 0:
        raise ValueError("no rows were augmented; check --frac")

    out.mkdir(parents=True)
    write_jsonl(out_rows, out / "train.jsonl")
    for split in COPY:
        shutil.copyfile(base / f"{split}.jsonl", out / f"{split}.jsonl")
        if hashlib.sha256((base / f"{split}.jsonl").read_bytes()).hexdigest() != hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest():
            raise RuntimeError(f"{split}: copy checksum mismatch")
    receipt = {"base": str(base), "base_manifest_sha256": hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest(),
               "frac": frac, "seed": seed, "train_count": len(out_rows), "prior_rows": n_prior,
               "families": dict(Counter(r.family for r in out_rows)), "copied_verbatim": list(COPY),
               "sealed_splits_read": [], "files": {s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in ("train",) + COPY}}
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "families"}, indent=1), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()
    build(args.base, args.out, args.frac, args.seed)


if __name__ == "__main__":
    main()
