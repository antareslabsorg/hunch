#!/usr/bin/env python3
"""Fairness-sensitive read of the toxicity families (design notes, unpublished): does the scorer treat comments that mention an
identity differently from comments that do not?

    python scripts/fairness_eval.py --preds results/<run>_devall.json results/<run>_calibrationall.json [...] --out results/fairness.json

For every predictions file (dev split; the sealed test split is never read) and for each Civil Comments family —
`moderation.civil_comments` (Boolean attribute questions) and `moderation.score.civil_toxicity_level` (5 levels) —
rows are split by whether the comment mentions a term from the identity lexicon below (a proxy for identity
content, not an annotation: it misses references without the listed words and catches quoted or negated
mentions). Per group: n, accuracy, cross-entropy against the target distribution (the card's NLL), smooth ECE
when the project's implementation is importable (else a 15-bin ECE, named as such), and the false-toxic rate —
the share of comments whose gold says not toxic that the scorer calls toxic (Boolean: gold P(true) < 0.5 and
predicted P(true) >= 0.5; Score: gold level <= 1 and predicted level >= 3). Gaps (identity minus none) carry
bootstrap 95 % intervals (1,000 resamples, seed 0). Groups under 200 rows are reported as too small.
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the card's smooth ECE lives in hunch/; without this a direct run silently fell back to ece15
SPLITS = [ROOT / "data/sprint_v3d/dev.jsonl", ROOT / "data/sprint_v3d/calibration.jsonl"]  # never test.jsonl
FAMILIES = ["moderation.civil_comments", "moderation.score.civil_toxicity_level"]
LEXICON = {  # the Jigsaw unintended-bias identity groups, as surface terms; matched as whole words, case-insensitive
    "gender": ["woman", "women", "man", "men", "female", "male", "girl", "girls", "boy", "boys", "wife", "husband", "mother", "father",
               "transgender", "trans", "nonbinary", "non-binary"],
    "sexual orientation": ["gay", "gays", "lesbian", "lesbians", "bisexual", "homosexual", "homosexuals", "queer", "lgbt", "lgbtq", "heterosexual", "straight"],
    "religion": ["christian", "christians", "catholic", "catholics", "protestant", "jew", "jews", "jewish", "muslim", "muslims", "islam", "islamic",
                 "hindu", "hindus", "buddhist", "buddhists", "atheist", "atheists", "sikh", "mormon", "evangelical"],
    "race / ethnicity": ["black", "blacks", "white", "whites", "asian", "asians", "latino", "latina", "latinos", "hispanic", "hispanics", "african",
                         "africans", "arab", "arabs", "mexican", "mexicans", "chinese", "indian", "indians", "native american", "indigenous"],
    "disability": ["disabled", "disability", "disabilities", "wheelchair", "blind", "deaf", "autistic", "autism", "mentally ill", "mental illness",
                   "schizophrenic", "bipolar", "retarded", "handicapped"],
}
PATTERNS = {cat: re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I) for cat, terms in LEXICON.items()}

try:  # the card's smooth ECE (hunch/metrics.py), if importable
    from hunch.metrics import smooth_ece as _smece  # type: ignore
    CAL_NAME = "smece"
    def calib(conf, correct): return float(_smece(np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)))
except Exception:
    CAL_NAME = "ece15"
    def calib(conf, correct):
        conf, correct = np.asarray(conf), np.asarray(correct, dtype=float); bins = np.linspace(0, 1, 16); e = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any(): e += m.mean() * abs(conf[m].mean() - correct[m].mean())
        return float(e)

def load_dev():
    text, fam = {}, {}
    for path in SPLITS:
        assert path.name != "test.jsonl"
        for l in open(path):
            r = json.loads(l)
            if r["family"] in FAMILIES:
                s = r["state"]; text[r["id"]] = s.get("comment", "") if isinstance(s, dict) else str(s); fam[r["id"]] = r["family"]
    return text, fam

def identity_cats(t):
    return [c for c, p in PATTERNS.items() if p.search(t)]

def row_stats(r, family):
    p = np.array(r["probs"], float); t = r["target"]; tv = np.array(t, float) if isinstance(t, list) else np.eye(len(p))[int(t)]
    gold = int(tv.argmax()); pred = int(p.argmax())
    xent = -float((tv * np.log(np.clip(p, 1e-12, 1))).sum())
    if family == "moderation.civil_comments":
        gold_toxic = tv[1] >= 0.5 if len(tv) > 1 else gold == 1; pred_toxic = p[1] >= 0.5
    else:
        gold_toxic = gold >= 3; pred_toxic = pred >= 3
    gold_not_toxic = (tv[1] < 0.5 if len(tv) > 1 else gold == 0) if family == "moderation.civil_comments" else gold <= 1
    return dict(correct=int(pred == gold), xent=xent, conf=float(p.max()), false_toxic=(int(pred_toxic) if gold_not_toxic else None),
                missed_toxic=(int(not pred_toxic) if gold_toxic else None))

def summarize(rows):
    c = np.array([x["correct"] for x in rows]); x = np.array([x["xent"] for x in rows])
    ft = [x["false_toxic"] for x in rows if x["false_toxic"] is not None]; mt = [x["missed_toxic"] for x in rows if x["missed_toxic"] is not None]
    return {"n": len(rows), "accuracy": float(c.mean()), "xent": float(x.mean()), CAL_NAME: calib([r["conf"] for r in rows], c),
            "false_toxic_rate": (float(np.mean(ft)) if ft else None), "n_gold_not_toxic": len(ft),
            "missed_toxic_rate": (float(np.mean(mt)) if mt else None), "n_gold_toxic": len(mt)}

def gap_ci(a, b, key, rng, B=1000):
    def stat(rows):
        vals = [r[key] for r in rows if r[key] is not None]; return float(np.mean(vals)) if vals else np.nan
    point = stat(a) - stat(b); boots = []
    for _ in range(B):
        sa = [a[i] for i in rng.integers(0, len(a), len(a))]; sb = [b[i] for i in rng.integers(0, len(b), len(b))]
        boots.append(stat(sa) - stat(sb))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]); return {"gap": point, "ci95": [float(lo), float(hi)]}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    text, fam = load_dev(); out = {"lexicon": LEXICON, "calibration_metric": CAL_NAME, "seed": a.seed, "min_group": 200, "runs": {}}
    by_tag = defaultdict(list)  # pool dev + calibration files of the same run
    for pf in a.preds:
        tag = Path(pf).stem
        for suf in ("_devall", "_calibrationall", "_dev2000", "_dev", "_calibration"):
            if tag.endswith(suf): tag = tag[: -len(suf)]; break
        tag = tag.replace("frozen_devall_", "frozen_").replace("frozen_calibrationall_", "frozen_")
        by_tag[tag] += json.load(open(pf))["predictions"]
    for tag, rows in by_tag.items():
        out["runs"][tag] = {}
        for family in FAMILIES:
            groups = {"identity": [], "none": []}; cats = defaultdict(int); seen_comments = set()
            for r in rows:
                if r.get("family") != family or r["id"] not in text: continue
                cs = identity_cats(text[r["id"]]); g = "identity" if cs else "none"
                groups[g].append(row_stats(r, family))
                if r["id"].rsplit(":", 1)[0] not in seen_comments:
                    seen_comments.add(r["id"].rsplit(":", 1)[0]); [cats.__setitem__(c, cats[c] + 1) for c in cs]
            res = {"groups": {g: summarize(v) for g, v in groups.items() if v}, "identity_category_counts_by_comment": dict(cats)}
            small = [g for g, v in groups.items() if len(v) < 200]
            res["too_small"] = small
            if not small:
                rng = np.random.default_rng(a.seed)
                res["gaps_identity_minus_none"] = {k: gap_ci(groups["identity"], groups["none"], k, rng) for k in ("correct", "xent", "false_toxic", "missed_toxic")}
            out["runs"][tag][family] = res
            gi, gn = res["groups"].get("identity", {}), res["groups"].get("none", {})
            print(f"{tag} | {family}: identity n={gi.get('n')} acc={gi.get('accuracy'):.3f} false-toxic={gi.get('false_toxic_rate')} missed-toxic={gi.get('missed_toxic_rate')} | "
                  f"none n={gn.get('n')} acc={gn.get('accuracy'):.3f} false-toxic={gn.get('false_toxic_rate')} missed-toxic={gn.get('missed_toxic_rate')} | too small: {small or 'no'}")
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n"); print("wrote", a.out)

if __name__ == "__main__":
    main()
