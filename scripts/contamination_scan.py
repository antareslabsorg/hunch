#!/usr/bin/env python3
"""Contamination scan: train vs every evaluation set the project reports on.

Reproduces docs/21-contamination-report.md from a clean checkout (full scan):

    .venv/bin/python scripts/contamination_scan.py
    .venv/bin/python scripts/contamination_scan.py --write docs/21-contamination-report.md

Smoke (not for the committed report):

    .venv/bin/python scripts/contamination_scan.py --max-train 2000 --max-eval 200 --write /tmp/contam-smoke.md

Never opens data/sprint_v3d/test.jsonl. Never uploads to the Hub. No GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATHS = [ROOT / "data" / "sprint_v3d" / "train.jsonl"]   # --train replaces it
# Explicit paths only — never glob sprint_v3d (sealed test.jsonl lives there).
LOCAL_EVAL = {
    "heldout": ROOT / "data" / "sprint_v3d" / "heldout.jsonl",
    "dev": ROOT / "data" / "sprint_v3d" / "dev.jsonl",
    "calibration": ROOT / "data" / "sprint_v3d" / "calibration.jsonl",
}
TD_REPO = "LocalLLaMA/typed-decisions"
TD_REVISION = "c76749ec58bd"
TD_FILE = "all/test-00000-of-00001.parquet"
V2_REPO = "pngwn/typed-decisions-v2-system-one"
V2_FILE = "data/test-00000-of-00001.parquet"

# Method parameters (fixed for determinism; recorded in the report).
NGRAM_N = 13
JACCARD_THRESHOLD = 0.8
MINHASH_PERM = 128
MINHASH_BANDS = 32  # rows/band = 4; high detection probability for s ≥ 0.8
MINHASH_SEED = 42
NEAR_DUP_THRESHOLD = 0.8
TOP_K = 10
PREVIEW_CHARS = 500
# Rare-gram inverted index: grams with document frequency ≤ this are indexed for
# candidate generation (catches mid-range Jaccard that LSH at 0.8 may miss; also
# supplies top-10 pairs when near-dup rates are zero).
RARE_DF_MAX = 200
RARE_CAND_CAP = 512
MERSENNE = np.uint64((1 << 61) - 1)
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# State extraction / normalization
# ---------------------------------------------------------------------------

def state_text(state) -> str:
    """Comparison text of a state: a plain string as is; a structured state (or a JSON string that decodes to one) as its
    string values in sorted-key order, one per line."""
    if isinstance(state, str):
        s = state.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                state = json.loads(s)
            except json.JSONDecodeError:
                return state
        else:
            return state
    if isinstance(state, (dict, list)):
        # Values only, in sorted-key order: the same text under different field names ("text" in one dataset,
        # "comment" in another) must collide. Serializing with the keys hid such pairs in the 2026-09-23 report.
        return "\n".join(leaf_values(state))
    return str(state)


def leaf_values(obj) -> list[str]:
    if isinstance(obj, dict):
        return [v for k in sorted(obj) for v in leaf_values(obj[k])]
    if isinstance(obj, list):
        return [v for x in obj for v in leaf_values(x)]
    return [] if obj is None else [str(obj)]


def normalize(text: str) -> str:
    """NFKC, lowercase, collapse whitespace — used for exact match, n-grams, MinHash."""
    t = unicodedata.normalize("NFKC", text).lower().strip()
    return _WS.sub(" ", t)


def row_state(obj) -> object:
    if "state" not in obj:
        raise KeyError(f"row missing 'state': keys={sorted(obj)[:20]}")
    return obj["state"]


def row_id(obj, fallback: str) -> str:
    for k in ("id", "Id", "ID"):
        if k in obj and obj[k] is not None:
            return str(obj[k])
    return fallback


# ---------------------------------------------------------------------------
# 13-gram Jaccard
# ---------------------------------------------------------------------------

def char_ngrams(s: str, n: int = NGRAM_N) -> set[str]:
    if len(s) < n:
        return set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set[str], b: set[str], a_text: str, b_text: str) -> float:
    """Jaccard on character n-gram sets. Short strings (<n): 1.0 iff equal else 0.0."""
    if len(a_text) < NGRAM_N or len(b_text) < NGRAM_N:
        return 1.0 if a_text == b_text else 0.0
    if not a and not b:
        return 1.0 if a_text == b_text else 0.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def gram_hash(g: str) -> int:
    return int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "little")


# ---------------------------------------------------------------------------
# MinHash + LSH (vendored; datasketch is not a project dependency)
# ---------------------------------------------------------------------------

class MinHashLSH:
    """num_perm permutations, bands bands of (num_perm // bands) rows each."""

    def __init__(
        self,
        num_perm: int = MINHASH_PERM,
        bands: int = MINHASH_BANDS,
        seed: int = MINHASH_SEED,
    ):
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
        self.b = rng.randint(0, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
        self.signatures: list[np.ndarray] = []
        self._buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(bands)]

    def _sig_from_hashes(self, token_hashes: list[int]) -> np.ndarray:
        if not token_hashes:
            return np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
        th = np.asarray(token_hashes, dtype=np.uint64)
        vals = (self.a * th[:, None] + self.b) % MERSENNE
        return vals.min(axis=0).astype(np.uint64)

    def signature_for_grams(self, grams: set[str], text: str) -> np.ndarray:
        if not grams:
            h = int.from_bytes(
                hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little"
            )
            return self._sig_from_hashes([h])
        return self._sig_from_hashes([gram_hash(g) for g in grams])

    def add(self, index: int, sig: np.ndarray) -> None:
        assert index == len(self.signatures)
        self.signatures.append(sig)
        for b in range(self.bands):
            start = b * self.rows
            band = sig[start : start + self.rows].tobytes()
            key = int.from_bytes(hashlib.blake2b(band, digest_size=8).digest(), "little")
            self._buckets[b][key].append(index)

    def query(self, sig: np.ndarray) -> set[int]:
        out: set[int] = set()
        for b in range(self.bands):
            start = b * self.rows
            band = sig[start : start + self.rows].tobytes()
            key = int.from_bytes(hashlib.blake2b(band, digest_size=8).digest(), "little")
            out.update(self._buckets[b].get(key, ()))
        return out


# ---------------------------------------------------------------------------
# Loaders (explicit paths / pinned Hub revisions)
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path, max_rows: int | None = None):
    sealed = (ROOT / "data" / "sprint_v3d" / "test.jsonl").resolve()
    if path.resolve() == sealed:
        raise RuntimeError("refusing to read sealed data/sprint_v3d/test.jsonl")
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if max_rows is not None and n >= max_rows:
                return


def load_local_eval(
    name: str, path: Path, max_rows: int | None
) -> tuple[list[str], list[str]]:
    ids, norms = [], []
    for i, obj in enumerate(iter_jsonl(path, max_rows)):
        ids.append(row_id(obj, f"{name}:{i}"))
        norms.append(normalize(state_text(row_state(obj))))
    return ids, norms


def load_hub_eval(
    label: str,
    repo: str,
    filename: str,
    revision: str | None,
    max_rows: int | None,
    cache_dir: Path,
) -> tuple[list[str], list[str], str | None, str | None]:
    """Returns ids, norms, resolved_revision, error_message."""
    try:
        from huggingface_hub import hf_hub_download, dataset_info
    except ImportError as e:
        return [], [], None, f"huggingface_hub not importable: {e}"

    try:
        if revision is None:
            info = dataset_info(repo, files_metadata=False)
            revision = info.sha
        path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir),
        )
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        cols = set(table.column_names)
        if "state" not in cols:
            return [], [], revision, f"{label}: parquet missing 'state'; have {sorted(cols)}"
        states = table.column("state").to_pylist()
        if "id" in cols:
            ids_raw = table.column("id").to_pylist()
        else:
            ids_raw = [f"{label}:{i}" for i in range(len(states))]
        if max_rows is not None:
            states = states[:max_rows]
            ids_raw = ids_raw[:max_rows]
        ids = [str(x) for x in ids_raw]
        norms = [normalize(state_text(st)) for st in states]
        return ids, norms, revision, None
    except Exception as e:  # noqa: BLE001 — record Hub errors in the report
        return [], [], revision, f"{label}: download/load failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Train index
# ---------------------------------------------------------------------------

class TrainIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.norms: list[str] = []
        self.exact: dict[str, list[int]] = defaultdict(list)
        self.lsh = MinHashLSH()
        # gram_hash(uint64) -> list of train indices; only for grams with 1 < df ≤ RARE_DF_MAX
        # (df=1 cannot match another train row; eval still hits via postings when shared)
        self.rare_postings: dict[int, list[int]] = {}

    def build(self, max_train: int | None) -> None:
        # Pass 1: norms + exact + MinHash; DF counted by gram hash to save RAM
        df: dict[int, int] = defaultdict(int)
        # temporary: per-doc list of rare-candidate hashes (rebuilt after DF known)
        print(f"Indexing train from {', '.join(map(str, TRAIN_PATHS))} ...", flush=True)
        rows = (obj for path in TRAIN_PATHS for obj in iter_jsonl(path, max_train))
        for i, obj in enumerate(rows):
            tid = row_id(obj, f"train:{i}")
            norm = normalize(state_text(row_state(obj)))
            grams = char_ngrams(norm)
            self.ids.append(tid)
            self.norms.append(norm)
            self.exact[norm].append(i)
            sig = self.lsh.signature_for_grams(grams, norm)
            self.lsh.add(i, sig)
            for g in grams:
                df[gram_hash(g)] += 1
            if (i + 1) % 25000 == 0:
                print(f"  train {i + 1}", flush=True)

        n = len(self.norms)
        self.id_set = set(self.ids)
        print(
            f"Train loaded n={n}. Building rare-gram postings "
            f"(1 ≤ df ≤ {RARE_DF_MAX})...",
            flush=True,
        )
        rare_ok = {h for h, c in df.items() if 1 <= c <= RARE_DF_MAX}
        del df
        postings: dict[int, list[int]] = defaultdict(list)
        for i, norm in enumerate(self.norms):
            for g in char_ngrams(norm):
                h = gram_hash(g)
                if h in rare_ok:
                    postings[h].append(i)
            if (i + 1) % 25000 == 0:
                print(f"  postings {i + 1}", flush=True)
        self.rare_postings = dict(postings)
        print(
            f"Train index ready: n={n}, rare_gram_keys={len(self.rare_postings)}",
            flush=True,
        )

    def candidates(self, text: str, grams: set[str], sig: np.ndarray) -> set[int]:
        cands = set(self.lsh.query(sig))
        cands.update(self.exact.get(text, ()))
        scored: dict[int, int] = defaultdict(int)
        for g in grams:
            posting = self.rare_postings.get(gram_hash(g))
            if not posting:
                continue
            for ti in posting:
                scored[ti] += 1
        if scored:
            top = heapq.nlargest(
                RARE_CAND_CAP, scored.items(), key=lambda kv: (kv[1], -kv[0])
            )
            cands.update(ti for ti, _ in top)
        return cands


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan_pair(name: str, eval_ids: list[str], eval_norms: list[str], train: TrainIndex) -> dict:
    n_eval = len(eval_norms)
    exact_hits = 0
    jaccard_hits = 0
    neardup_hits = 0
    exact_ids: list[str] = []
    exact_pairs: list[tuple] = []   # (eval id, first matching train id, state length) for the breakdown
    jaccard_ids: list[str] = []
    neardup_ids: list[str] = []
    heap: list[tuple] = []  # (jaccard, mh_sim, ei, tj) min-heap

    for ei, etext in enumerate(eval_norms):
        egrams = char_ngrams(etext)
        esig = train.lsh.signature_for_grams(egrams, etext)
        exact_idxs = train.exact.get(etext, [])
        if exact_idxs:
            exact_hits += 1
            exact_ids.append(eval_ids[ei])
            exact_pairs.append((eval_ids[ei], train.ids[exact_idxs[0]], len(etext)))

        cands = train.candidates(etext, egrams, esig)
        best_j = -1.0
        best_mh = -1.0
        best_tj = -1

        for tj in cands:
            ttext = train.norms[tj]
            j = jaccard(egrams, char_ngrams(ttext), etext, ttext)
            mh = float(np.mean(esig == train.lsh.signatures[tj]))
            if j > best_j or (j == best_j and mh > best_mh):
                best_j, best_mh, best_tj = j, mh, tj

        if best_j >= JACCARD_THRESHOLD:
            jaccard_hits += 1
            jaccard_ids.append(eval_ids[ei])
        if best_mh >= NEAR_DUP_THRESHOLD or best_j >= NEAR_DUP_THRESHOLD:
            neardup_hits += 1
            neardup_ids.append(eval_ids[ei])

        if best_tj >= 0:
            item = (best_j, best_mh, ei, best_tj)
            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

        if (ei + 1) % 10000 == 0:
            print(f"  [{name}] {ei + 1}/{n_eval}", flush=True)

    # If we have fewer than TOP_K pairs (tiny eval / no candidates), leave as-is.
    top = sorted(heap, reverse=True)
    pairs = []
    for j, mh, ei, tj in top:
        pairs.append(
            {
                "jaccard": j,
                "minhash_sim": mh,
                "eval_id": eval_ids[ei],
                "train_id": train.ids[tj],
                "eval_state": eval_norms[ei][:PREVIEW_CHARS],
                "train_state": train.norms[tj][:PREVIEW_CHARS],
                "exact": eval_norms[ei] == train.norms[tj],
            }
        )

    return {
        "name": name,
        "n_eval": n_eval,
        "n_train": len(train.norms),
        "exact_matches": exact_hits,
        "exact_match_rate": (exact_hits / n_eval) if n_eval else 0.0,
        "jaccard_hits": jaccard_hits,
        "jaccard_overlap_rate": (jaccard_hits / n_eval) if n_eval else 0.0,
        "neardup_hits": neardup_hits,
        "neardup_rate": (neardup_hits / n_eval) if n_eval else 0.0,
        "exact_ids": exact_ids,
        "exact_pairs": exact_pairs,
        "id_leaks": len(set(eval_ids) & train.id_set),
        "jaccard_ids": jaccard_ids,
        "neardup_ids": neardup_ids,
        "top_pairs": pairs,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt_rate(hits: int, n: int, rate: float) -> str:
    return f"{hits}/{n} = {rate:.6f}"


def source(row_id: str) -> str:
    return row_id.split(":", 1)[0]


def exact_breakdown(results: list[dict]) -> list[str]:
    """Where the exact matches come from, computed from the scan (not written by hand)."""
    rows = [r for r in results if not r.get("error") and r["exact_matches"]]
    if not rows:
        return ["No evaluation state equals a training state in any scanned set.", ""]
    out = []
    for r in rows:
        pairs = r["exact_pairs"]
        by_src: dict[str, int] = defaultdict(int)
        for e, _, _ in pairs:
            by_src[source(e)] += 1
        cross = sum(source(e) != source(t) for e, t, _ in pairs)
        lens = sorted(n for _, _, n in pairs)
        out += [f"**{r['name']}**: {len(pairs)} exact matches; {r['id_leaks']} evaluation ids occur in train. "
                f"{sum(n < 80 for n in lens)} of the matched states are under 80 characters (median {lens[len(lens) // 2]}); "
                f"{cross} match a training row from a different source dataset.", "",
                "| source of the evaluation row | exact matches |", "|---|---:|"]
        out += [f"| `{k}` | {v} |" for k, v in sorted(by_src.items(), key=lambda kv: -kv[1])] + [""]
    return out


def render_report(
    results: list[dict],
    n_train: int,
    runtime_s: float,
    full_scan: bool,
    v2_revision: str | None,
    hub_errors: list[str],
    public: bool = False,
) -> str:
    contaminated_sets = [
        r["name"]
        for r in results
        if r.get("error") is None
        and (r["exact_matches"] > 0 or r["jaccard_hits"] > 0 or r["neardup_hits"] > 0)
    ]
    scanned = [r["name"] for r in results if r.get("error") is None]
    failed = [r["name"] for r in results if r.get("error")]

    if contaminated_sets:
        clean = [s for s in scanned if s not in contaminated_sets]
        first = (
            f"**Yes — evaluation items appear in training** for: "
            f"{', '.join(contaminated_sets)}. "
            f"Sets with zero exact-match, zero 13-gram Jaccard≥{JACCARD_THRESHOLD}, "
            f"and zero MinHash near-duplicate≥{NEAR_DUP_THRESHOLD} among those scanned: "
            f"{', '.join(clean) if clean else '(none)'}."
        )
    elif scanned:
        first = (
            f"**No — no evaluation item appears in training** among the sets scanned "
            f"({', '.join(scanned)}): exact-match rate, 13-gram Jaccard overlap "
            f"(threshold {JACCARD_THRESHOLD}), and MinHash near-duplicate rate "
            f"(threshold {NEAR_DUP_THRESHOLD}) are all zero."
            + (f" Could not scan: {', '.join(failed)}." if failed else "")
        )
    else:
        first = (
            "Contamination status: **could not be determined** — no evaluation set "
            f"scanned successfully. Failed: {', '.join(failed) if failed else '(none)'}."
        )

    if full_scan:
        meta = (
            f"Date: **{time.strftime('%Y-%m-%d')}**. Runtime: **{runtime_s:.1f}s**. "
            "Scan: **full (not a sample)** — `--max-train` / `--max-eval` were unset."
        )
    else:
        meta = (
            f"Date: **{time.strftime('%Y-%m-%d')}**. Runtime: **{runtime_s:.1f}s**. "
            "Scan: **SAMPLED — do not treat as release evidence**."
        )

    lines = [
        "# Contamination report",
        "",
        first,
        "",
        meta,
        "",
        "## Method",
        "",
        "- **State field**: each row's `state` (the request schema, `hunch/schema.py`). A structured state is compared "
        "by its string values only, in sorted-key order, so the same text under different field names matches; "
        "JSON strings are parsed first when they decode as objects/arrays; plain strings are kept.",
        "- **Normalization** (exact match, n-grams, MinHash): Unicode NFKC, lowercase, "
        "collapse runs of whitespace to a single space, strip.",
        "- **Exact match**: equality of normalized state text. Rate = "
        "(# eval rows whose normalized state equals ≥1 train row) / n_eval.",
        f"- **13-gram Jaccard**: character {NGRAM_N}-grams over normalized text. "
        f"If either side has length < {NGRAM_N}, Jaccard is 1.0 iff the strings are equal else 0.0. "
        f"Overlap hit if max Jaccard vs train ≥ **{JACCARD_THRESHOLD}**.",
        f"- **MinHash near-duplicates**: vendored MinHash (datasketch not required), "
        f"**{MINHASH_PERM}** permutations, seed **{MINHASH_SEED}**, LSH **{MINHASH_BANDS}** bands "
        f"× {MINHASH_PERM // MINHASH_BANDS} rows. Near-dup hit if estimated MinHash similarity "
        f"≥ **{NEAR_DUP_THRESHOLD}** or verified Jaccard ≥ **{NEAR_DUP_THRESHOLD}**.",
        f"- **Candidates** for Jaccard / near-dup / top-{TOP_K}: MinHash LSH hits ∪ exact "
        f"matches ∪ train rows sharing a rare character {NGRAM_N}-gram "
        f"(train document frequency in 1..{RARE_DF_MAX}; at most {RARE_CAND_CAP} rare-gram "
        f"candidates per eval, ranked by shared rare-gram count). Rates are verified "
        f"Jaccard / MinHash on those candidates — not estimated-only.",
        f"- **Top-{TOP_K} pairs**: highest verified Jaccard (tie-break MinHash sim); "
        + ("listed by id and similarity only; this version quotes no dataset text." if public else
           f"each side truncated to {PREVIEW_CHARS} characters in this report; rates use full normalized text."),
        f"- **Train**: " + ", ".join(f"`{p.parent.name}/{p.name}`" for p in TRAIN_PATHS)
        + f" (n={n_train}). Never reads a `test.jsonl`.",
        f"- **Hub pins**: `{TD_REPO}` revision `{TD_REVISION}`; `{V2_REPO}` revision "
        f"`{v2_revision or '(unavailable)'}`.",
        "- **Hardware**: CPU only; no GPU.",
        "",
        "## Summary",
        "",
        "| Eval set | n_eval | exact-match rate | "
        f"13-gram Jaccard≥{JACCARD_THRESHOLD} | MinHash near-dup≥{NEAR_DUP_THRESHOLD} |",
        "|---|---:|---:|---:|---:|",
    ]

    for r in results:
        if r.get("error"):
            lines.append(f"| {r['name']} | — | ERROR | ERROR | ERROR |")
        else:
            lines.append(
                f"| {r['name']} | {r['n_eval']} | "
                f"{fmt_rate(r['exact_matches'], r['n_eval'], r['exact_match_rate'])} | "
                f"{fmt_rate(r['jaccard_hits'], r['n_eval'], r['jaccard_overlap_rate'])} | "
                f"{fmt_rate(r['neardup_hits'], r['n_eval'], r['neardup_rate'])} |"
            )

    lines += [
        "",
        f"n_train = **{n_train}**.",
        "",
        "## What the exact matches are",
        "",
    ]
    lines += exact_breakdown(results)

    if hub_errors:
        lines += ["## Hub load errors", ""]
        for e in hub_errors:
            lines += [f"- {e}", ""]

    for r in results:
        lines += [f"## {r['name']}", ""]
        if r.get("error"):
            lines += [f"Not scanned: `{r['error']}`", ""]
            continue
        lines += [
            f"- n_eval = {r['n_eval']}, n_train = {r['n_train']}",
            f"- exact-match: {fmt_rate(r['exact_matches'], r['n_eval'], r['exact_match_rate'])}",
            f"- 13-gram Jaccard≥{JACCARD_THRESHOLD}: "
            f"{fmt_rate(r['jaccard_hits'], r['n_eval'], r['jaccard_overlap_rate'])}",
            f"- MinHash near-dup≥{NEAR_DUP_THRESHOLD}: "
            f"{fmt_rate(r['neardup_hits'], r['n_eval'], r['neardup_rate'])}",
            "",
            f"### Top {TOP_K} most-similar pairs" + ("" if public else " (verbatim previews)"),
            "",
        ]
        if not r["top_pairs"]:
            lines += [
                "_(no candidate pairs recalled — LSH / exact / rare-gram index returned empty)_",
                "",
            ]
            continue
        if public:
            lines += ["| rank | Jaccard | MinHash | exact | evaluation id | training id |", "|---:|---:|---:|---|---|---|"]
            lines += [f"| {k} | {p['jaccard']:.4f} | {p['minhash_sim']:.4f} | {p['exact']} | `{p['eval_id']}` | `{p['train_id']}` |"
                      for k, p in enumerate(r["top_pairs"], 1)] + [""]
            continue
        for rank, p in enumerate(r["top_pairs"], 1):
            lines += [
                f"**{rank}. Jaccard={p['jaccard']:.6f}, MinHash≈{p['minhash_sim']:.6f}, "
                f"exact={p['exact']}**",
                f"- eval `{p['eval_id']}`:",
                "```",
                p["eval_state"],
                "```",
                f"- train `{p['train_id']}`:",
                "```",
                p["train_state"],
                "```",
                "",
            ]

    lines += [
        "## Reproduce",
        "",
        "```bash",
        "python scripts/contamination_scan.py --train " + " ".join(f"data/{p.parent.name}/{p.name}" for p in TRAIN_PATHS)
        + " --write docs/21-contamination-report.md --public docs/21-contamination-report.public.md"
        " --dump-ids docs/results/launch/contamination_ids.json",
        "```",
        "",
        "Requires `huggingface_hub`, `numpy` and `pyarrow`. MinHash is vendored in the script.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--write",
        type=Path,
        default=ROOT / "results" / "contamination-full.md",
        help="the full report, which quotes dataset text; keep it out of docs/, whose docs/21 is the ids-only --public version",
    )
    ap.add_argument("--max-train", type=int, default=None, help="Smoke: cap train rows")
    ap.add_argument("--max-eval", type=int, default=None, help="Smoke: cap each eval set")
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".cache" / "contamination_hf",
        help="huggingface_hub cache for the two Hub datasets",
    )
    ap.add_argument("--print-only", action="store_true", help="Print report; do not write")
    ap.add_argument("--dump-ids", type=Path, default=None, help="also write the matched eval ids per set (exact / jaccard / near-dup) as JSON")
    ap.add_argument("--train", type=Path, nargs="+", default=None, help="training files to scan against (default: data/sprint_v3d/train.jsonl)")
    ap.add_argument("--data", type=Path, default=None, help="directory holding heldout/dev/calibration.jsonl (default: data/sprint_v3d)")
    ap.add_argument("--public", type=Path, default=None, help="also write a version that lists pairs by id only, quoting no dataset text")
    args = ap.parse_args()
    if args.train:
        TRAIN_PATHS[:] = args.train
    if args.data:
        for k in LOCAL_EVAL: LOCAL_EVAL[k] = args.data / f"{k}.jsonl"

    full_scan = args.max_train is None and args.max_eval is None
    t_start = time.time()

    train = TrainIndex()
    train.build(args.max_train)
    n_train = len(train.norms)

    results: list[dict] = []
    hub_errors: list[str] = []
    v2_revision: str | None = None

    for name, path in LOCAL_EVAL.items():
        print(f"Scanning {name} ({path}) ...", flush=True)
        ids, norms = load_local_eval(name, path, args.max_eval)
        results.append(scan_pair(name, ids, norms, train))

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {TD_REPO}@{TD_REVISION} ...", flush=True)
    ids, norms, rev, err = load_hub_eval(
        "typed-decisions", TD_REPO, TD_FILE, TD_REVISION, args.max_eval, args.cache_dir
    )
    if err:
        hub_errors.append(err)
        results.append({"name": "typed-decisions", "error": err})
    else:
        print(f"Scanning typed-decisions (n={len(ids)}, rev={rev}) ...", flush=True)
        results.append(scan_pair("typed-decisions", ids, norms, train))

    print(f"Fetching {V2_REPO} (pin to fetched revision) ...", flush=True)
    ids, norms, rev, err = load_hub_eval(
        "typed-decisions-v2-system-one",
        V2_REPO,
        V2_FILE,
        None,
        args.max_eval,
        args.cache_dir,
    )
    v2_revision = rev
    if err:
        hub_errors.append(err)
        results.append({"name": "typed-decisions-v2-system-one", "error": err})
    else:
        print(
            f"Scanning typed-decisions-v2-system-one (n={len(ids)}, rev={rev}) ...",
            flush=True,
        )
        results.append(scan_pair("typed-decisions-v2-system-one", ids, norms, train))

    runtime_s = time.time() - t_start
    report = render_report(results, n_train, runtime_s, full_scan, v2_revision, hub_errors)
    if args.public:
        args.public.parent.mkdir(parents=True, exist_ok=True)
        args.public.write_text(render_report(results, n_train, runtime_s, full_scan, v2_revision, hub_errors, public=True), encoding="utf-8")
        print(f"Wrote {args.public} (ids only)", flush=True)
    if args.dump_ids:
        args.dump_ids.parent.mkdir(parents=True, exist_ok=True)
        args.dump_ids.write_text(json.dumps({r["name"]: {k: r[k] for k in ("exact_ids", "jaccard_ids", "neardup_ids")} for r in results if not r.get("error")}))
        print(f"Wrote matched ids to {args.dump_ids}", flush=True)

    if args.print_only:
        print(report)
    else:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(report, encoding="utf-8")
        print(f"Wrote {args.write} ({runtime_s:.1f}s)", flush=True)

    print("---SUMMARY---", flush=True)
    for r in results:
        if r.get("error"):
            print(f"{r['name']}: ERROR {r['error']}", flush=True)
        else:
            print(
                f"{r['name']}: n={r['n_eval']} exact={r['exact_match_rate']:.6f} "
                f"j13={r['jaccard_overlap_rate']:.6f} near={r['neardup_rate']:.6f}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
