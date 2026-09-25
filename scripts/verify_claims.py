#!/usr/bin/env python3
"""Check that every number the launch documents quote is the number in the file it cites.

    python scripts/verify_claims.py [--claims claims.json]
    python scripts/verify_claims.py --selftest

`claims.json` is a list of {"doc", "file", "path", "fmt"} (and optionally "scale", e.g. 100 for a percentage). For each claim
the value at `path` (dotted, or a list of keys) inside the JSON `file` is formatted with `fmt` and must appear verbatim in `doc`. A missing
file, a missing key or a number the document does not contain is a failure. The check is deliberately literal: it proves
the quoted digits come from the cited file, not that the prose around them is right.

A second check covers the other direction: every decimal number in a checked document (outside code spans and links)
must be produced by one of its claims, or be declared in claims.json as {"doc", "constant", "why"} (a rule threshold,
a verbatim quotation). A number nobody cites or declares is a failure, so "every number is checked" stays true.
"""
import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dig(obj, path):
    """`path` is a dotted string, or a list of keys when a key itself contains dots (family names do)."""
    for part in (path if isinstance(path, list) else path.split(".")):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def check(claims: list, root: Path) -> list:
    problems, docs = [], {}
    for c in claims:
        if "constant" in c:   # declared, not measured: it must still appear in its document
            text = docs.setdefault(c["doc"], (root / c["doc"]).read_text() if (root / c["doc"]).is_file() else None)
            if text is None or not re.search(rf"(?<![\d.]){re.escape(c['constant'])}(?!\d)", text):
                problems.append(f"{c['doc']}: declared constant {c['constant']} does not appear")
            continue
        f = root / c["file"]
        if not f.is_file():
            problems.append(f"{c['doc']}: cited file missing: {c['file']}"); continue
        try:
            v = dig(json.loads(f.read_text()), c["path"])
        except (KeyError, IndexError, ValueError, TypeError) as e:
            problems.append(f"{c['doc']}: {c['file']} has no {c['path']} ({type(e).__name__})"); continue
        s = format(float(v) * c.get("scale", 1), c["fmt"])
        text = docs.setdefault(c["doc"], (root / c["doc"]).read_text() if (root / c["doc"]).is_file() else None)
        if text is None:
            problems.append(f"{c['doc']}: document missing"); continue
        if not re.search(rf"(?<![\d.]){re.escape(s)}(?!\d)", text):   # whole number: 0.818 must not match inside 0.8185
            problems.append(f"{c['doc']}: does not contain {s} ({c['file']} {c['path']})")
    return problems


DECIMAL = re.compile(r"(?<![\w.])(?<!\w-)(\d+\.\d+(?:e[-+]\d+)?)(?![\d.]*\w)")   # 0.8185, -5.30, 7.68e-02; not 0.6B, v0.1.0, 2.11.0, Apache-2.0


def coverage(claims: list, root: Path) -> list:
    """Every decimal number in a document that has claims is produced by one of its claims or declared a constant."""
    made: dict[str, set] = {}
    for c in claims:
        if "constant" in c:
            made.setdefault(c["doc"], set()).add(c["constant"]); continue
        try:
            v = dig(json.loads((root / c["file"]).read_text()), c["path"])
            made.setdefault(c["doc"], set()).add(format(float(v) * c.get("scale", 1), c["fmt"]).lstrip("+-"))
        except (OSError, KeyError, IndexError, ValueError, TypeError):
            continue   # check() reports it
    problems = []
    for doc, ok in sorted(made.items()):
        if not (root / doc).is_file():
            continue   # check() reports it
        text = re.sub(r"`[^`]*`|https?://\S+|\]\([^)]*\)", " ", (root / doc).read_text())   # code spans and links are not claims
        problems += [f"{doc}: {n} is neither a claim nor a declared constant" for n in sorted(set(DECIMAL.findall(text)) - ok)]
    return problems


def selftest() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "r.json").write_text(json.dumps({"overall": {"accuracy": 0.81849}, "rows": [0.5]}))
        (d / "doc.md").write_text("accuracy 0.8185, 81.8 %, row 0.50")
        ok = [{"doc": "doc.md", "file": "r.json", "path": "overall.accuracy", "fmt": ".4f"},
              {"doc": "doc.md", "file": "r.json", "path": "overall.accuracy", "fmt": ".1f", "scale": 100},
              {"doc": "doc.md", "file": "r.json", "path": "rows.0", "fmt": ".2f"}]
        assert check(ok, d) == [], check(ok, d)
        bad = [{"doc": "doc.md", "file": "r.json", "path": "overall.accuracy", "fmt": ".3f"},      # 0.818 not in doc
               {"doc": "doc.md", "file": "missing.json", "path": "x", "fmt": ".1f"},
               {"doc": "doc.md", "file": "r.json", "path": "overall.nll", "fmt": ".1f"}]
        assert len(check(bad, d)) == 3
        (d / "cov.md").write_text("reads 0.8185 at T = 1.0, bar 0.03, z -0.50, see `--tv 7.68e-02`, Qwen3-0.6B, v0.1.0, Apache-2.0")
        cov = [{"doc": "cov.md", "file": "r.json", "path": "overall.accuracy", "fmt": ".4f"},
               {"doc": "cov.md", "constant": "1.0", "why": "fixed temperature"}]
        assert coverage(cov, d) == ["cov.md: 0.03 is neither a claim nor a declared constant",
                                    "cov.md: 0.50 is neither a claim nor a declared constant"], coverage(cov, d)   # a negative is checked too
        assert check(cov + [{"doc": "cov.md", "constant": "0.04", "why": "stale"}], d) == ["cov.md: declared constant 0.04 does not appear"]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", default=str(ROOT / "claims.json")); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    claims = json.loads(Path(a.claims).read_text())
    problems = check(claims, ROOT) + coverage(claims, ROOT)
    print(f"{len(claims)} claims checked, coverage checked, {len(problems)} problem(s)")
    for p in problems: print("  " + p)
    sys.exit(1 if problems else 0)
