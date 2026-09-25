"""Build the sprint dataset: JSONL per split plus a manifest with counts and hashes.

    python -m hunch.data.build --out data/sprint_v3d --max-per-source 20000 --allow-license-classes A,B --release-only ...   # see REPRODUCE

License classes are enforced here: a source outside the allowed set is skipped and recorded in the manifest.
Records are validated on write; nothing is truncated or repaired silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..schema import SPLITS, Record
from . import families, generators


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--sources", nargs="*", default=None, help="subset of HF ids; default = all sprint sources")
    p.add_argument("--max-per-source", type=int, default=20000, help="cap on examples per source per HF split (adapters apply it)")
    p.add_argument("--allow-license-classes", default="A,B", help="comma-separated; D-class sources are research-only")
    p.add_argument("--generators", type=int, default=5000, help="records per exact-truth generator (0 disables)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--authored", default=None, help="directory of authored source specs (docs/research/sources); enables authored descriptions, criteria and training-row instruction paraphrases")
    p.add_argument("--no-paraphrase", action="store_true", help="with --authored: keep the adapters' default instructions on every split (authored candidate descriptions and criteria only; no paraphrase augmentation)")
    p.add_argument("--extra-sources", nargs="*", default=[], help="opt-in sources by HF id added on top of the selected mixture (e.g. stanfordnlp/snli); release builds still enforce the admitted-source check on them")
    p.add_argument("--implicature", action="store_true", help="also emit the synthetic conversational-implicature generator family (pragmatics.implicature); off by default so existing data versions stay reproducible")
    p.add_argument("--release-only", action="store_true", help="admitted sources only (license class A/B and no closed-model labels); HelpSteer2 becomes a training family; held-out = Circa, CaseHOLD, GoEmotions")
    args = p.parse_args()

    if args.authored:
        from . import authored

        authored.DIR = Path(args.authored)
        authored.PARAPHRASE = not args.no_paraphrase
        if not authored.DIR.exists():
            raise SystemExit(f"--authored directory not found: {authored.DIR}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allowed = set(args.allow_license_classes.split(","))
    extra = [families.ALL_SOURCES[s] for s in args.extra_sources]
    if args.release_only:
        sources = families.RELEASE_TRAIN_SOURCES + extra + families.RELEASE_HELDOUT_SOURCES
        allowed &= {"A", "B"}
        bad = [s.name for s in sources if not s.admitted or s.license_class not in allowed]
        if bad:
            raise SystemExit(f"release build contains non-admitted sources: {bad}")
    else:
        sources = (list(families.SPRINT_TRAIN_SOURCES + families.SPRINT_HELDOUT_SOURCES) if not args.sources else [families.ALL_SOURCES[s] for s in args.sources]) + extra

    writers = {s: (out / f"{s}.jsonl").open("w", encoding="utf-8") for s in SPLITS}
    counts: dict[str, Counter] = defaultdict(Counter)
    ktypes: dict[str, Counter] = defaultdict(Counter)
    skipped, errors = [], {}
    started = time.time()

    def emit(rec: Record) -> None:
        rec.validate()
        writers[rec.split].write(rec.to_json() + "\n")
        counts[rec.family][rec.split] += 1
        ktypes[rec.family][f"{rec.question.type}:K={rec.question.k}"] += 1

    for src in sources:
        if src.license_class not in allowed:
            skipped.append({"source": src.name, "license_class": src.license_class})
            print(f"skip {src.name} (license class {src.license_class} not in {sorted(allowed)})", flush=True)
            continue
        t0 = time.time()
        n = 0
        try:
            for rec in src.build(max_per_split=args.max_per_source):
                emit(rec)
                n += 1
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            errors[src.name] = f"{type(exc).__name__}: {exc}"
            print(f"ERROR {src.name}: {errors[src.name]}", flush=True)
        print(f"{src.name}: {n} records in {time.time() - t0:.0f}s", flush=True)

    if args.generators:
        split_fn = lambda g: families.assign_split(g, has_official_eval=False, seed=args.seed)  # noqa: E731
        gen_list = generators.refund_records(args.generators, args.seed, split_fn) + generators.urn_records(args.generators, args.seed, split_fn)
        if args.implicature:  # opt-in so that v3/v3d/v4/v4d rebuild byte-identically to the archived manifests
            gen_list += generators.implicature_records(args.generators, args.seed, split_fn)
        for rec in gen_list:
            emit(rec)

    for w in writers.values():
        w.close()
    manifest = {
        "schema": "hunch-data-v1", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - started),
        "args": vars(args), "files": {s: {"path": f"{s}.jsonl", "sha256": sha256_file(out / f"{s}.jsonl"), "bytes": (out / f"{s}.jsonl").stat().st_size} for s in SPLITS},
        "counts_by_family_split": {f: dict(c) for f, c in counts.items()},
        "types_by_family": {f: dict(c) for f, c in ktypes.items()},
        "totals_by_split": {s: sum(c[s] for c in counts.values()) for s in SPLITS},
        "skipped_sources": skipped, "errors": errors, "release_only": args.release_only,
        "licenses": {f"{s.name} [{s.family}]": {"class": s.license_class, "admitted": s.admitted, "note": s.note} for s in sources},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest["totals_by_split"]), flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} source(s) failed: {sorted(errors)}")


if __name__ == "__main__":
    main()
