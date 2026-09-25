#!/usr/bin/env python3
"""PII scan of the sprint_v3d training, dev and calibration files.

Streams explicit JSONL paths, never globs directories, never opens test.jsonl.
Stdout is the Markdown report (also writable with --write). Stdlib + already-
installed packages only. No GPU.

Name lists (preferred order):
  1. Cache under /tmp/pii_scan_names/ (Census 2010 top surnames + SSA-derived
     given names via Hadley baby-names CSV).
  2. Fetch those sources if the cache is missing and the network allows.
  3. Fall back to a clearly labelled built-in partial list.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterator
# ---------------------------------------------------------------------------
# Paths (explicit only)
# ---------------------------------------------------------------------------

DEFAULT_FILES = [
    "data/sprint_v3d/train.jsonl",
    "data/sprint_v3d/heldout.jsonl",
    "data/sprint_v3d/dev.jsonl",
    "data/sprint_v3d/calibration.jsonl",
]

FORBIDDEN = "data/sprint_v3d/test.jsonl"

CACHE_DIR = "/tmp/pii_scan_names"
CENSUS_URL = "https://www2.census.gov/topics/genealogy/2010surnames/names.zip"
BABY_NAMES_URL = (
    "https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv"
)
UA = {"User-Agent": "hunch-pii-scan/1.0 (training-data audit)"}

REPORT_DATE = __import__("datetime").date.today().isoformat()   # the day the scan runs

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"([A-Za-z0-9][A-Za-z0-9._%+-]*)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"(?![A-Za-z0-9._%+-])"
)
# NANP with separators or +intl. Deliberately NO bare `NNN NNNN` —
# that form collides with legal reporter pins (`350 1998`, `led2d 621 2005`).
PHONE_RE = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"\+\d{1,3}([\s.-]?\(?\d{1,4}\)?[\s.-]?\d{2,4}){2,3}"
    r"|"
    r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"
    r")"
    r"(?!\w)"
)

# IBAN: 2 letters + 2 digits + 11–30 alnum (spaces optional).
IBAN_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"([A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30})"
    r"(?![A-Za-z0-9])"
)

# Card-shaped: 13–19 digits WITH at least one separator (space or dash).
# Separator-free digit runs are ignored (order IDs, hashes, ISBNs-without-dashes).
CARD_RE = re.compile(
    r"(?<!\d)"
    r"((?:\d{4}[ -]){2,4}\d{1,7}|\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{1,7})"
    r"(?!\d)"
)

# Street: house number + street name + suffix. Suffix must be abbreviated
# (St./Ave./…) OR a Capitalized full form, to avoid "15 minute drive",
# "took place", "down the road", "on the way".
STREET_RE = re.compile(
    r"(?<!\w)"
    r"("
    r"\d{1,5}"
    r"(?:\s+[NSEWnsew]{1,2}\.?)?"
    r"\s+[A-Za-z][A-Za-z0-9.'\-]*"
    r"(?:\s+[A-Za-z][A-Za-z0-9.'\-]*){0,2}"
    r"\s+(?:"
    r"Street|Avenue|Road|Boulevard|Lane|Drive|Court|Place|Way|Terrace|"
    r"Circle|Highway|Parkway|"
    r"St\.|Ave\.|Rd\.|Blvd\.|Ln\.|Dr\.|Ct\.|Pl\.|Ter\.|Cir\.|Hwy\.|Pkwy\."
    r")"
    r")"
    r"(?!\w)"
)

STREET_FP_TOKENS = frozenset(
    {
        "minute",
        "minutes",
        "hour",
        "hours",
        "second",
        "seconds",
        "people",
        "person",
        "each",
        "the",
        "my",
        "our",
        "your",
        "his",
        "her",
        "their",
        "from",
        "down",
        "on",
        "in",
        "at",
        "to",
        "of",
        "and",
        "or",
        "a",
        "an",
        "took",
        "take",
        "takes",
        "taken",
        "taking",
        "first",
        "second",
        "third",
        "only",
        "about",
        "around",
        "over",
        "under",
        "long",
        "short",
        "nice",
        "quick",
        "block",
        "blocks",
        "mile",
        "miles",
        "km",
        "kilometer",
        "kilometres",
        "kilometers",
        "supreme",
        "unanimous",
        "appeals",
        "appellate",
        "circuit",
        "district",
        "federal",
        "high",
        "trial",
        "known",
        "also",
        "called",
        "named",
        "us",
        "u.s",
        "united",
        "states",
        "county",
        "municipal",
        "superior",
        "magistrate",
        "juvenile",
        "traffic",
        "small",
        "claims",
    }
)# Title + capitalized name
TITLED_NAME_RE = re.compile(
    r"(?<!\w)((?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

# First Last (both Capitalized); validated against name lists later
FIRST_LAST_RE = re.compile(r"(?<!\w)([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")

TOKEN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")

# Synthetic / placeholder domains and phone fragments
PLACEHOLDER_EMAIL_DOMAINS = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "example.edu",
        "test.com",
        "test.org",
        "localhost",
        "invalid",
        "example.invalid",
        "mailinator.com",
        "fake.com",
        "email.com",  # often used as template slot in synthetic data
        "domain.com",
        "yourcompany.com",
        "company.com",
        "sample.com",
    }
)
PLACEHOLDER_EMAIL_LOCALS = frozenset(
    {
        "user",
        "username",
        "name",
        "email",
        "test",
        "tester",
        "example",
        "sample",
        "foo",
        "bar",
        "baz",
        "noreply",
        "no-reply",
        "donotreply",
        "placeholder",
        "john.doe",
        "jane.doe",
        "john",
        "jane",
        "foo.bar",
    }
)

# NANP fictional exchange 555-01xx; also common fake patterns
PLACEHOLDER_PHONE_FRAGMENTS = (
    "555-01",
    "555.01",
    "555 01",
    "(555) 01",
    "555-555",
    "555.555",
    "000-000",
    "000.000",
    "123-456-7890",
    "123.456.7890",
    "111-111-1111",
    "999-999-9999",
    "+1-555-01",
    "+155501",
)

PLACEHOLDER_ADDRESS_STREETS = frozenset(
    {
        "main",
        "first",
        "second",
        "third",
        "oak",
        "elm",
        "maple",
        "any",
        "example",
        "sample",
        "test",
        "fake",
        "placeholder",
        "lorem",
        "ipsum",
    }
)

# Template / dialogue slot names and classic crypto-persona placeholders
PLACEHOLDER_NAMES = frozenset(
    {
        "alice",
        "bob",
        "carol",
        "dave",
        "eve",
        "frank",
        "grace",
        "heidi",
        "ivan",
        "judy",
        "mallory",
        "olivia",
        "peggy",
        "sybil",
        "trudy",
        "victor",
        "walter",
        "foo",
        "bar",
        "baz",
        "qux",
        "john",
        "jane",
        "doe",
        "smith",  # only when paired as John/Jane Smith — handled in classify
        "lorem",
        "ipsum",
        "placeholder",
        "someone",
        "anybody",
        "person",
        "customer",
        "user",
        "client",
    }
)

# Ambiguous capitalized English words that are also common given/surnames —
# do not emit standalone given-name hits for these (First Last / titled still ok).
AMBIGUOUS_GIVEN = frozenset(
    {
        "will",
        "may",
        "april",
        "june",
        "august",
        "faith",
        "hope",
        "grace",
        "rose",
        "lily",
        "ivy",
        "pearl",
        "day",
        "king",
        "queen",
        "page",
        "case",
        "love",
        "chase",
        "hunter",
        "parker",
        "mason",
        "carter",
        "brooks",
        "forest",
        "river",
        "sky",
        "summer",
        "winter",
        "autumn",
        "spring",
        "robin",
        "sparrow",
        "brown",
        "black",
        "white",
        "green",
        "gray",
        "grey",
        "young",
        "long",
        "short",
        "little",
        "big",
        "north",
        "south",
        "east",
        "west",
        "christian",
        "gay",
        "pat",
        "max",
        "sam",
        "alex",
        "chris",
        "jordan",
        "taylor",
        "morgan",
        "casey",
        "jamie",
        "shannon",
        "kelly",
        "terry",
        "kerry",
        "jean",
        "gene",
        "bill",
        "bob",
        "joe",
        "don",
        "ray",
        "guy",
        "mark",
        "art",
        "van",
        "lee",
        "sue",
        "ann",
        "amy",
        "joy",
        "kay",
        "bea",
        "abe",
        "ned",
        "ted",
        "tim",
        "tom",
        "dan",
        "ben",
        "ken",
        "ron",
        "hal",
        "cal",
        "mac",
        "vic",
        "roy",
        "ian",
        "ali",
        "aria",
        "nova",
        "luna",
        "body",
        "home",
        "best",
        "real",
        "true",
        "fair",
        "free",
        "open",
        "early",
        "later",
        "major",
        "minor",
        "general",
        "colonel",
        "captain",
        "bishop",
        "dean",
        "clerk",
        "baker",
        "cook",
        "miller",
        "fisher",
        "hunter",
        "carpenter",
        "porter",
        "butler",
        "ward",
        "guard",
        "price",
        "worth",
        "rich",
        "poor",
        "smart",
        "bright",
        "strong",
        "hardy",
        "stern",
        "sharp",
        "swift",
        "wild",
        "savage",
        "noble",
        "royal",
        "prince",
        "duke",
        "earl",
        "baron",
        "knight",
        "saint",
        "angel",
        "devil",
        "heaven",
        "hell",
        "earth",
        "world",
        "state",
        "city",
        "town",
        "village",
        "county",
        "country",
        "america",
        "england",
        "france",
        "china",
        "india",
        "japan",
        "korea",
        "russia",
        "germany",
        "spain",
        "italy",
        "ireland",
        "scotland",
        "wales",
        "canada",
        "mexico",
        "brazil",
        "australia",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "july",
        "september",
        "october",
        "november",
        "december",
    }
)

# Sentence-start words that look like names but are not person references.
SENTENCE_STARTERS = frozenset(
    {
        "the",
        "this",
        "that",
        "these",
        "those",
        "there",
        "then",
        "than",
        "when",
        "what",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "and",
        "but",
        "for",
        "nor",
        "or",
        "yet",
        "so",
        "if",
        "as",
        "at",
        "by",
        "in",
        "of",
        "on",
        "to",
        "up",
        "with",
        "from",
        "into",
        "over",
        "after",
        "before",
        "during",
        "while",
        "although",
        "because",
        "since",
        "until",
        "unless",
        "though",
        "whether",
        "either",
        "neither",
        "both",
        "each",
        "every",
        "all",
        "any",
        "some",
        "no",
        "not",
        "only",
        "own",
        "same",
        "other",
        "such",
        "very",
        "just",
        "also",
        "even",
        "still",
        "already",
        "always",
        "never",
        "often",
        "sometimes",
        "here",
        "now",
        "once",
        "please",
        "thanks",
        "thank",
        "hi",
        "hello",
        "dear",
        "regarding",
        "re",
        "subject",
        "section",
        "chapter",
        "article",
        "appendix",
        "figure",
        "table",
        "note",
        "warning",
        "error",
        "info",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "well",
        "oh",
        "ah",
        "um",
        "uh",
    }
)

BUILTIN_GIVEN = [
    "James",
    "John",
    "Robert",
    "Michael",
    "William",
    "David",
    "Richard",
    "Joseph",
    "Thomas",
    "Charles",
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "Barbara",
    "Susan",
    "Jessica",
    "Sarah",
    "Karen",
    "Nancy",
    "Lisa",
    "Betty",
    "Margaret",
    "Sandra",
    "Ashley",
    "Dorothy",
    "Kimberly",
    "Emily",
    "Donna",
]
BUILTIN_SURNAME = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Rodriguez",
    "Martinez",
    "Hernandez",
    "Lopez",
    "Gonzalez",
    "Wilson",
    "Anderson",
    "Thomas",
    "Taylor",
    "Moore",
    "Jackson",
    "Martin",
]


# ---------------------------------------------------------------------------
# Name list loading
# ---------------------------------------------------------------------------


@dataclass
class NameLists:
    given: frozenset[str]
    surname: frozenset[str]
    source_note: str
    partial: bool


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _ensure_name_cache(top_surnames: int = 2000, top_given: int = 2000) -> tuple[str, str, str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    sur_path = os.path.join(CACHE_DIR, "census_2010_surnames_top.txt")
    giv_path = os.path.join(CACHE_DIR, "ssa_top_given.txt")
    meta_path = os.path.join(CACHE_DIR, "names_meta.json")

    if not os.path.isfile(sur_path):
        zdata = _http_get(CENSUS_URL)
        with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
            with zf.open("Names_2010Census.csv") as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="latin-1"))
                surnames: list[str] = []
                for row in reader:
                    name = (row.get("name") or row.get("NAME") or "").strip()
                    if not name or name.upper() == "ALL OTHER NAMES":
                        continue
                    surnames.append(name.title())
                    if len(surnames) >= top_surnames:
                        break
        with open(sur_path, "w", encoding="utf-8") as f:
            f.write("\n".join(surnames) + "\n")

    if not os.path.isfile(giv_path):
        raw = _http_get(BABY_NAMES_URL).decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))
        best: dict[str, float] = {}
        years: set[str] = set()
        for row in reader:
            years.add(row["year"])
            name = row["name"].strip()
            pct = float(row["percent"])
            if name not in best or pct > best[name]:
                best[name] = pct
        given = [n for n, _ in sorted(best.items(), key=lambda x: -x[1])[:top_given]]
        with open(giv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(given) + "\n")
        meta = {
            "census_surnames": {
                "source": CENSUS_URL,
                "file": "Names_2010Census.csv",
                "vintage": "2010 US Census",
                "selection": f"top {top_surnames} by frequency rank",
            },
            "given_names": {
                "source": BABY_NAMES_URL,
                "upstream": "US SSA baby names via Hadley Wickham data-baby-names",
                "years_in_file": f"{min(years)}-{max(years)}" if years else "unknown",
                "selection": f"top {top_given} by peak percent across years",
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    if not os.path.isfile(meta_path):
        meta = {
            "census_surnames": {
                "source": CENSUS_URL,
                "vintage": "2010 US Census",
                "selection": f"top {top_surnames} by frequency rank",
                "note": "meta reconstructed; lists loaded from cache files",
            },
            "given_names": {
                "source": BABY_NAMES_URL,
                "upstream": "US SSA baby names via Hadley Wickham data-baby-names",
                "selection": f"top {top_given} by peak percent across years",
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return sur_path, giv_path, meta_path


def load_name_lists() -> NameLists:
    try:
        sur_path, giv_path, meta_path = _ensure_name_cache()
        with open(sur_path, encoding="utf-8") as f:
            surnames = {ln.strip() for ln in f if ln.strip()}
        with open(giv_path, encoding="utf-8") as f:
            given = {ln.strip() for ln in f if ln.strip()}
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        note = (
            f"US Census 2010 top surnames ({len(surnames)}; {meta['census_surnames']['source']}); "
            f"SSA-derived given names via Hadley baby-names "
            f"({len(given)}; {meta['given_names']['source']}, "
            f"{meta['given_names'].get('years_in_file', meta['given_names'].get('years', 'years per cache meta'))}). "
            f"Cached under {CACHE_DIR}."
        )
        return NameLists(
            given=frozenset(given),
            surname=frozenset(surnames),
            source_note=note,
            partial=False,
        )
    except Exception as exc:  # noqa: BLE001 — documented fallback
        note = (
            f"PARTIAL name channel: download/cache failed ({type(exc).__name__}: {exc}). "
            f"Using built-in list of {len(BUILTIN_GIVEN)} given + {len(BUILTIN_SURNAME)} surnames."
        )
        return NameLists(
            given=frozenset(BUILTIN_GIVEN),
            surname=frozenset(BUILTIN_SURNAME),
            source_note=note,
            partial=True,
        )


# ---------------------------------------------------------------------------
# Redaction & classification
# ---------------------------------------------------------------------------


def redact_email(local: str, domain: str) -> str:
    if len(local) <= 2:
        masked_local = local[:1] + "*"
    else:
        masked_local = local[:1] + ("*" * (len(local) - 3)) + local[-2:]
    parts = domain.split(".")
    if len(parts) >= 2:
        masked_dom = parts[0][:1] + "***." + parts[-1]
    else:
        masked_dom = domain[:1] + "***"
    return f"{masked_local}@{masked_dom}"


def redact_digits(s: str, keep_last: int = 2) -> str:
    out = []
    digits = [c for c in s if c.isdigit()]
    n = len(digits)
    di = 0
    for c in s:
        if c.isdigit():
            if di >= n - keep_last:
                out.append(c)
            else:
                out.append("X")
            di += 1
        else:
            out.append(c)
    return "".join(out)


def redact_street(s: str) -> str:
    masked = re.sub(r"\d", "X", s, count=0)
    return re.sub(r"\s+", " ", masked).strip()


def redact_name(s: str) -> str:
    parts = s.split()
    red = []
    for p in parts:
        if len(p) <= 1:
            red.append("*")
        else:
            red.append(p[0] + ("*" * (len(p) - 1)))
    return " ".join(red)


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_ok(num: str) -> bool:
    digits = [int(c) for c in num if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def classify_email(local: str, domain: str) -> str:
    dom = domain.lower().rstrip(".")
    loc = local.lower()
    if dom in PLACEHOLDER_EMAIL_DOMAINS or dom.endswith(".example") or dom.endswith(".test"):
        return "synthetic-placeholder"
    if loc in PLACEHOLDER_EMAIL_LOCALS:
        return "synthetic-placeholder"
    if re.fullmatch(r"(user|test|sample|foo|bar)\d*", loc):
        return "synthetic-placeholder"
    if "example" in dom or "placeholder" in dom or "fake" in dom:
        return "synthetic-placeholder"
    return "looks-real"


def classify_phone(raw: str) -> str:
    compact = re.sub(r"\s+", " ", raw.strip())
    lower = compact.lower()
    for frag in PLACEHOLDER_PHONE_FRAGMENTS:
        if frag in lower or frag.replace("-", "") in _digits_only(compact):
            return "synthetic-placeholder"
    digits = _digits_only(compact)
    if digits.startswith("55501") or "55501" in digits:
        return "synthetic-placeholder"
    if len(set(digits)) <= 2:
        return "synthetic-placeholder"
    if digits in {"1234567890", "0123456789", "9876543210"}:
        return "synthetic-placeholder"
    return "looks-real"


def classify_street(raw: str) -> str:
    lower = raw.lower()
    if any(
        tok in lower
        for tok in (
            "example",
            "placeholder",
            "lorem",
            "ipsum",
            "fake",
            "test street",
            "any street",
        )
    ):
        return "synthetic-placeholder"
    tokens = re.findall(r"[A-Za-z0-9']+", lower)
    # drop leading house number
    if tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    if any(t in STREET_FP_TOKENS for t in tokens[:-1]):
        return "synthetic-placeholder"
    m = re.match(r"\d+\s+([a-z0-9.'\-]+)", lower)
    if m and m.group(1) in PLACEHOLDER_ADDRESS_STREETS:
        if re.search(r"\b(?:apt|suite|unit|#)\s*\d+[a-z]?\b", lower):
            return "looks-real"
        return "synthetic-placeholder"
    if re.match(r"123\s+main\b", lower):
        return "synthetic-placeholder"
    return "looks-real"


def street_is_plausible(raw: str) -> bool:
    """Reject time/distance/court collocations that survive the regex."""
    tokens = re.findall(r"[A-Za-z0-9']+", raw)
    if len(tokens) < 3:
        return False
    body = [t.lower() for t in tokens[1:-1]]
    if any(t in STREET_FP_TOKENS for t in body):
        return False
    suffix = tokens[-1].lower().rstrip(".")
    if suffix in {"drive", "way", "road", "place", "court", "lane"} and tokens[-1].islower():
        return False
    # "N Supreme Court" / "N US Supreme Court"
    if suffix == "court" and any(
        t in {"supreme", "appeals", "circuit", "district", "high", "superior", "federal"}
        for t in body
    ):
        return False
    # Bare "N … Hwy." without a highway-like name
    if suffix in {"hwy", "highway", "pkwy", "parkway"} and (
        not body or all(t in STREET_FP_TOKENS or len(t) <= 2 for t in body)
    ):
        return False
    return True

def classify_iban_or_card(kind: str, raw: str) -> str:
    digits = _digits_only(raw)
    if kind == "card":
        # ISBN-13 (978/979) is not a payment card.
        if digits.startswith(("978", "979")):
            return "synthetic-placeholder"
        if len(set(digits)) <= 2:
            return "synthetic-placeholder"
        if digits.startswith("4" * 4) and len(set(digits)) <= 3:
            return "synthetic-placeholder"
        if digits in {
            "4111111111111111",
            "4000000000000002",
            "5555555555554444",
            "5105105105105100",
            "378282246310005",
            "371449635398431",
            "6011111111111117",
        }:
            return "synthetic-placeholder"
        if not luhn_ok(digits):
            return "synthetic-placeholder"
        return "looks-real"
    compact = re.sub(r"\s+", "", raw.upper())
    if compact.startswith("XX") or "EXAMPLE" in compact:
        return "synthetic-placeholder"
    if len(set(compact[4:])) <= 2:
        return "synthetic-placeholder"
    try:
        rearranged = compact[4:] + compact[:4]
        numeric = "".join(str(ord(c) - 55) if c.isalpha() else c for c in rearranged)
        if int(numeric) % 97 == 1:
            return "looks-real"
        return "synthetic-placeholder"
    except Exception:  # noqa: BLE001
        return "looks-real"

def classify_name(raw: str, channel: str) -> str:
    parts = [p for p in re.split(r"\s+", raw.strip()) if p]
    lower_parts = [p.lower().rstrip(".") for p in parts]
    # Strip title
    if lower_parts and lower_parts[0] in {"mr", "mrs", "ms", "miss", "dr", "prof"}:
        lower_parts = lower_parts[1:]
    if not lower_parts:
        return "synthetic-placeholder"
    if all(p in PLACEHOLDER_NAMES for p in lower_parts):
        return "synthetic-placeholder"
    if lower_parts in (["john", "doe"], ["jane", "doe"], ["john", "smith"], ["jane", "smith"]):
        return "synthetic-placeholder"
    if channel == "given_token" and lower_parts[0] in PLACEHOLDER_NAMES:
        return "synthetic-placeholder"
    return "looks-real"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

CHANNELS = (
    "email",
    "phone",
    "iban",
    "card",
    "street",
    "name_full",
    "name_titled",
    "name_given",
)


@dataclass
class Hit:
    channel: str
    label: str  # synthetic-placeholder | looks-real
    redacted: str
    file: str
    family: str
    row_id: str
    field_path: str


@dataclass
class ScanState:
    names: NameLists
    per_file: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    per_family: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    per_file_family: dict[str, dict[str, Counter]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(Counter))
    )
    rows_per_file: Counter = field(default_factory=Counter)
    examples: list[Hit] = field(default_factory=list)
    example_budget: int = 20
    # Per-channel pools so late-file contact hits are not starved by early names.
    _pools: dict[str, list[Hit]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _pool_cap: int = 40
    _seen_ex: set[tuple[str, str, str]] = field(default_factory=set)

    def add(self, hit: Hit) -> None:
        key = f"{hit.channel}:{hit.label}"
        self.per_file[hit.file][key] += 1
        self.per_family[hit.family][key] += 1
        self.per_file_family[hit.file][hit.family][key] += 1
        self.per_file[hit.file][hit.channel] += 1
        self.per_family[hit.family][hit.channel] += 1
        self.per_file_family[hit.file][hit.family][hit.channel] += 1
        if hit.label == "looks-real":
            self.per_file[hit.file]["looks_real_any"] += 1
            self.per_family[hit.family]["looks_real_any"] += 1
        else:
            self.per_file[hit.file]["synthetic_any"] += 1
            self.per_family[hit.family]["synthetic_any"] += 1
        sig = (hit.channel, hit.label, hit.redacted)
        if sig in self._seen_ex:
            return
        pool = self._pools[hit.channel]
        if len(pool) >= self._pool_cap:
            # Prefer looks-real over placeholder inside a full pool.
            if hit.label != "looks-real":
                return
            for i, old in enumerate(pool):
                if old.label == "synthetic-placeholder":
                    self._seen_ex.discard((old.channel, old.label, old.redacted))
                    self._seen_ex.add(sig)
                    pool[i] = hit
                    return
            return
        self._seen_ex.add(sig)
        pool.append(hit)

    def select_examples(self) -> list[Hit]:
        """Pick up to example_budget examples with channel/label diversity."""
        order = (
            "email",
            "phone",
            "iban",
            "card",
            "street",
            "name_full",
            "name_titled",
            "name_given",
        )
        pool: list[Hit] = []
        for ch in order:
            items = list(self._pools.get(ch, []))
            items.sort(
                key=lambda h: (
                    0 if h.label == "looks-real" else 1,
                    h.file,
                    h.family,
                    h.redacted,
                )
            )
            pool.extend(items)
        picked: list[Hit] = []
        seen_channel_label: Counter = Counter()
        # First pass: prefer contact channels, ≤4 per (channel, label)
        for h in pool:
            key = (h.channel, h.label)
            limit = 4 if h.channel in {"email", "phone", "iban", "card", "street"} else 3
            if seen_channel_label[key] >= limit:
                continue
            picked.append(h)
            seen_channel_label[key] += 1
            if len(picked) >= self.example_budget:
                return picked
        picked_sigs = {(p.channel, p.label, p.redacted) for p in picked}
        for h in pool:
            sig = (h.channel, h.label, h.redacted)
            if sig in picked_sigs:
                continue
            picked.append(h)
            picked_sigs.add(sig)
            if len(picked) >= self.example_budget:
                break
        return picked
def iter_strings(obj: object, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(obj, str):
        if obj:
            yield path or "$", obj
    elif isinstance(obj, dict):
        for k in sorted(obj.keys()):
            p = f"{path}.{k}" if path else str(k)
            yield from iter_strings(obj[k], p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_strings(v, f"{path}[{i}]")


def scan_text(
    text: str,
    *,
    file: str,
    family: str,
    row_id: str,
    field_path: str,
    state: ScanState,
) -> None:
    names = state.names

    for m in EMAIL_RE.finditer(text):
        local, domain = m.group(1), m.group(2)
        label = classify_email(local, domain)
        state.add(
            Hit(
                "email",
                label,
                redact_email(local, domain),
                file,
                family,
                row_id,
                field_path,
            )
        )

    for m in PHONE_RE.finditer(text):
        raw = m.group(0)
        digits = _digits_only(raw)
        if len(digits) < 10 or len(digits) > 15:
            continue
        # Skip citation-like contexts (reporter volume + year).
        window = text[max(0, m.start() - 24) : m.end() + 8].lower()
        if re.search(
            r"\b(led2?d?|s\.?\s*ct\.?|u\.?s\.?c?\.?|f\.?\s*\d?d|n\.?e\.?2?d?|"
            r"p\.?\s*2?d?|a\.?\s*2?d?|so?\.?\s*2?d?|cal\.?\s*(?:app|rptr)?|"
            r"stat\.?|c\.?f\.?r\.?|u\.?s\.?c\.?)\b",
            window,
        ):
            continue
        label = classify_phone(raw)
        state.add(
            Hit(
                "phone",
                label,
                redact_digits(raw),
                file,
                family,
                row_id,
                field_path,
            )
        )

    for m in IBAN_RE.finditer(text):
        raw = m.group(1)
        compact = re.sub(r"\s+", "", raw)
        if len(compact) < 15:
            continue
        label = classify_iban_or_card("iban", raw)
        state.add(
            Hit(
                "iban",
                label,
                redact_digits(raw, keep_last=2),
                file,
                family,
                row_id,
                field_path,
            )
        )

    for m in CARD_RE.finditer(text):
        raw = m.group(1)
        digits = _digits_only(raw)
        if len(digits) < 13 or len(digits) > 19:
            continue
        if " " not in raw and "-" not in raw:
            continue
        # ISBN-13 is not a card.
        if digits.startswith(("978", "979")):
            continue
        label = classify_iban_or_card("card", raw)
        state.add(
            Hit(
                "card",
                label,
                redact_digits(raw),
                file,
                family,
                row_id,
                field_path,
            )
        )

    for m in STREET_RE.finditer(text):
        raw = m.group(1)
        if not street_is_plausible(raw):
            continue
        label = classify_street(raw)
        state.add(
            Hit(
                "street",
                label,
                redact_street(raw),
                file,
                family,
                row_id,
                field_path,
            )
        )
    for m in TITLED_NAME_RE.finditer(text):
        raw = m.group(1)
        # Validate last token against name lists
        toks = raw.split()
        person = toks[-1]
        if person not in names.given and person not in names.surname:
            continue
        label = classify_name(raw, "titled")
        state.add(
            Hit(
                "name_titled",
                label,
                redact_name(raw),
                file,
                family,
                row_id,
                field_path,
            )
        )

    for m in FIRST_LAST_RE.finditer(text):
        first, last = m.group(1), m.group(2)
        if first.lower() in SENTENCE_STARTERS:
            continue
        if first not in names.given:
            continue
        if last not in names.surname:
            continue
        # Avoid "New York", "United States"-like: last must not be a country/city word already excluded
        raw = f"{first} {last}"
        label = classify_name(raw, "full")
        state.add(
            Hit(
                "name_full",
                label,
                redact_name(raw),
                file,
                family,
                row_id,
                field_path,
            )
        )

    # Standalone given-name tokens (stricter): not ambiguous, not sentence-initial after .!?
    for m in TOKEN_RE.finditer(text):
        tok = m.group(1)
        if tok not in names.given:
            continue
        if tok.lower() in AMBIGUOUS_GIVEN or tok.lower() in SENTENCE_STARTERS:
            continue
        # Skip if this token is part of a First Last we already count
        # (simple check: next token capitalized surname)
        after = text[m.end() : m.end() + 32]
        m2 = re.match(r"\s+([A-Z][a-z]+)\b", after)
        if m2 and m2.group(1) in names.surname:
            continue
        # Skip title-prefixed (already counted)
        before = text[max(0, m.start() - 12) : m.start()]
        if re.search(r"(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+$", before):
            continue
        # Skip ALL-CAPS contexts somehow already title-cased only
        label = classify_name(tok, "given_token")
        state.add(
            Hit(
                "name_given",
                label,
                redact_name(tok),
                file,
                family,
                row_id,
                field_path,
            )
        )


def scan_file(path: str, state: ScanState, limit: int | None = None) -> None:
    if os.path.normpath(path).endswith(os.path.normpath(FORBIDDEN)) or path.endswith(
        "test.jsonl"
    ):
        raise SystemExit(f"refusing to open sealed file: {path}")
    base = os.path.basename(path)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            state.rows_per_file[base] += 1
            family = row.get("family") or row.get("source") or "(unknown)"
            row_id = str(row.get("id", f"line-{i}"))
            for field_path, text in iter_strings(row):
                scan_text(
                    text,
                    file=base,
                    family=str(family),
                    row_id=row_id,
                    field_path=field_path,
                    state=state,
                )
            if (i + 1) % 25000 == 0:
                print(f"[scan] {base}: {i + 1} rows", file=sys.stderr)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _contact_breakdown(state: ScanState) -> list[str]:
    lines = []
    headers = ["channel", "looks-real", "synthetic-placeholder", "total"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for ch in ("email", "phone", "iban", "card", "street"):
        real = _sum_label(state, (ch,), "looks-real")
        syn = _sum_label(state, (ch,), "synthetic-placeholder")
        lines.append(
            "| "
            + " | ".join([f"`{ch}`", str(real), str(syn), str(real + syn)])
            + " |"
        )
    return lines


def _count_table(state: ScanState, files: list[str]) -> list[str]:
    lines = []
    headers = [
        "file",
        "rows",
        "email",
        "phone",
        "iban",
        "card",
        "street",
        "name_full",
        "name_titled",
        "name_given",
        "looks_real",
        "synthetic",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for path in files:
        base = os.path.basename(path)
        c = state.per_file[base]
        row = [
            f"`{base}`",
            str(state.rows_per_file[base]),
            str(c.get("email", 0)),
            str(c.get("phone", 0)),
            str(c.get("iban", 0)),
            str(c.get("card", 0)),
            str(c.get("street", 0)),
            str(c.get("name_full", 0)),
            str(c.get("name_titled", 0)),
            str(c.get("name_given", 0)),
            str(c.get("looks_real_any", 0)),
            str(c.get("synthetic_any", 0)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    # totals
    tot = Counter()
    for base in state.per_file:
        tot.update(state.per_file[base])
    row = [
        "**total**",
        str(sum(state.rows_per_file.values())),
        str(tot.get("email", 0)),
        str(tot.get("phone", 0)),
        str(tot.get("iban", 0)),
        str(tot.get("card", 0)),
        str(tot.get("street", 0)),
        str(tot.get("name_full", 0)),
        str(tot.get("name_titled", 0)),
        str(tot.get("name_given", 0)),
        str(tot.get("looks_real_any", 0)),
        str(tot.get("synthetic_any", 0)),
    ]
    lines.append("| " + " | ".join(row) + " |")
    return lines


def _family_table(state: ScanState) -> list[str]:
    lines = []
    headers = [
        "family",
        "email",
        "phone",
        "iban",
        "card",
        "street",
        "name_full",
        "name_titled",
        "name_given",
        "looks_real",
        "synthetic",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    families = sorted(state.per_family.keys())
    for fam in families:
        c = state.per_family[fam]
        if not any(c.get(ch, 0) for ch in CHANNELS):
            continue
        row = [
            f"`{fam}`",
            str(c.get("email", 0)),
            str(c.get("phone", 0)),
            str(c.get("iban", 0)),
            str(c.get("card", 0)),
            str(c.get("street", 0)),
            str(c.get("name_full", 0)),
            str(c.get("name_titled", 0)),
            str(c.get("name_given", 0)),
            str(c.get("looks_real_any", 0)),
            str(c.get("synthetic_any", 0)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _sum_label(state: ScanState, channels: tuple[str, ...], label: str) -> int:
    total = 0
    for c in state.per_file.values():
        for ch in channels:
            total += c.get(f"{ch}:{label}", 0)
    return total


def _looks_real_verdict(state: ScanState) -> tuple[str, str]:
    """Return (headline_tag, prose).

    Contact channels (email/phone/iban/card/street) are the release-blocking question.
    Name channels are reported separately: public-figure and dialogue names in published
    datasets will match the lists and label `looks-real` under the literal rules.
    """
    contact_real = _sum_label(
        state, ("email", "phone", "iban", "card", "street"), "looks-real"
    )
    contact_syn = _sum_label(
        state, ("email", "phone", "iban", "card", "street"), "synthetic-placeholder"
    )
    name_full_real = _sum_label(state, ("name_full", "name_titled"), "looks-real")
    name_given_real = _sum_label(state, ("name_given",), "looks-real")

    if contact_real > 0:
        tag = "contact-looks-real"
        prose = (
            f"**Yes — live-looking contact PII patterns are present:** "
            f"{contact_real} hit(s) labelled `looks-real` on email / phone / IBAN / card / "
            f"street. Also {name_full_real} full/titled-name and {name_given_real} "
            f"given-name token hit(s) labelled `looks-real` (these include public-figure "
            f"and dialogue names in published source text). "
            f"Contact placeholders: {contact_syn}. See § Examples."
        )
        return tag, prose

    if contact_syn > 0 and name_full_real == 0 and name_given_real == 0:
        tag = "contact-placeholder-only"
        prose = (
            f"**No live-looking contact PII.** {contact_syn} email/phone/IBAN/card/street "
            f"hit(s) were all classified `synthetic-placeholder` under § Method."
        )
        return tag, prose

    if contact_syn == 0 and name_full_real == 0 and name_given_real == 0:
        tag = "clean"
        prose = (
            "**No.** No email, phone, IBAN, card, street, or name-list hits were found "
            "in the four scanned files."
        )
        return tag, prose

    # Names only (possibly with contact placeholders)
    extra = (
        f" Contact-shaped hits classified placeholder: {contact_syn}."
        if contact_syn
        else ""
    )
    tag = "names-only"
    prose = (
        f"**No live-looking contact PII** (email / phone / IBAN / card / street "
        f"`looks-real` = 0).{extra} Name-list hits labelled `looks-real`: "
        f"{name_full_real} full/titled, {name_given_real} given-token. Under the "
        f"literal rules these are not treated as synthetic placeholders (they are not "
        f"Alice/Bob/John Doe slots), but they include celebrity, fictional, and ordinary "
        f"first names that appear in published dialogue and comments — not harvested "
        f"private contact records. A reader who wants zero name-list hits should "
        f"disagree with the name channel policy, not with the contact verdict."
    )
    return tag, prose


def render_report(state: ScanState, files: list[str], repo_root: str) -> str:
    tag, verdict = _looks_real_verdict(state)
    if tag == "contact-looks-real":
        present, absent = [], []
        for ch in ("email", "phone", "iban", "card", "street"):
            (present if _sum_label(state, (ch,), "looks-real") else absent).append(ch)
        headline_contact = (
            "Live-looking contact patterns are present on "
            + ", ".join(present)
            + ". Zero looks-real hits on "
            + ", ".join(absent)
            + "."
        )
    else:
        headline_contact = None
    headline = {
        "contact-looks-real": headline_contact,
        "contact-placeholder-only": (
            "Contact-shaped strings are placeholder-only; no live-looking contact PII."
        ),
        "clean": "No detector hits on the four scanned files.",
        "names-only": (
            "No live-looking contact PII; name-list hits only (public/dialogue names under literal rules)."
        ),
    }.get(tag, tag)
    lines: list[str] = []
    lines.append(f"# PII scan of sprint_v3d bundles — {REPORT_DATE}")
    lines.append("")
    lines.append(f"**Headline:** {headline}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "Direct answer to “is anything that looks like real PII present?”:"
    )
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Scanned explicit paths only (never a directory glob; never "
        "`data/sprint_v3d/test.jsonl`). Each JSONL row is parsed; every string value "
        "is walked deterministically (`dict` keys sorted). Source family is the row "
        "field `family` (present on all inspected rows; `source` / `source_group` also "
        "exist but families are the training/evaluation taxonomy used here)."
    )
    lines.append("")
    lines.append("Detectors:")
    lines.append("")
    lines.append("- **email** — `local@domain.tld` with word boundaries.")
    lines.append(
        "- **phone** — NANP `(NNN) NNN-NNNN` / `NNN-NNN-NNNN` or `+` international; "
        "≥10 digits. Bare `NNN NNNN` is excluded (collides with legal reporter pins). "
        "Windows matching citation markers (`led2d`, `s.ct.`, …) are skipped."
    )
    lines.append(
        "- **IBAN** — `[A-Z]{2}[0-9]{2}` + 11–30 alphanumerics; classified with ISO 13616 mod-97 when parseable."
    )
    lines.append(
        "- **card** — 13–19 digits **with** space/dash separators and Luhn pass; "
        "ISBN-13 prefixes `978`/`979` excluded; separator-free digit runs ignored."
    )
    lines.append(
        "- **street** — house number + ≤3 name tokens + suffix. Full-word suffixes must be "
        "Capitalized (`Street`, `Avenue`, …); abbreviations (`St.`, `Ave.`, …) allowed. "
        "Rejects time/distance collocations (`minute`/`hour`/`people`/… before the suffix)."
    )
    lines.append(
        "- **names** — (a) `name_full`: Capitalized First+Last where First ∈ given list "
        "and Last ∈ surname list; (b) `name_titled`: Mr/Ms/Dr/… + name on a list; "
        "(c) `name_given`: standalone Capitalized given-name token after removing "
        "ambiguous English/calendar/color words and skipping tokens that are the "
        "first half of a First Last pair."
    )
    lines.append("")
    lines.append(f"**Name lists:** {state.names.source_note}")
    if state.names.partial:
        lines.append("")
        lines.append(
            "> Name channel is **partial** (built-in fallback). Re-run with network "
            f"access so lists can populate `{CACHE_DIR}`."
        )
    lines.append("")
    lines.append("### Placeholder vs looks-real rules")
    lines.append("")
    lines.append(
        "These rules are explicit so a reader can disagree. A hit is "
        "`synthetic-placeholder` when any of the following hold; otherwise "
        "`looks-real`:"
    )
    lines.append("")
    lines.append(
        "- **email:** domain in {example.com/org/net/edu, test.com/org, localhost, "
        "invalid, mailinator.com, fake.com, email.com, domain.com, company.com, "
        "yourcompany.com, sample.com} or ends with `.example`/`.test`; or local-part "
        "in {user, username, name, email, test, example, sample, foo, bar, baz, "
        "noreply, john.doe, jane.doe, …}; or local matches `(user|test|sample|foo|bar)\\d*`."
    )
    lines.append(
        "- **phone:** contains NANP fictional `555-01xx` / `555-555` / all-zero / "
        "`123-456-7890` / `111-111-1111` patterns; or ≤2 distinct digits."
    )
    lines.append(
        "- **card:** known test PANs; ISBN-13 `978`/`979`; ≤2 distinct digits; or fails Luhn."
    )
    lines.append(
        "- **IBAN:** country `XX`, contains EXAMPLE, low digit diversity, or fails mod-97."
    )
    lines.append(
        "- **street:** contains example/placeholder/lorem/fake; or body tokens in the "
        "time/distance stop list; or `N` + {main,first,oak,…} without apt/suite; or `123 Main …`."
    )
    lines.append(
        "- **names:** Alice/Bob/… crypto-persona set; John/Jane Doe; John/Jane Smith; "
        "or all tokens in the placeholder-name set."
    )
    lines.append("")
    lines.append(
        "Redaction in examples: emails keep first character and last two of the "
        "local-part and mask the rest; phones/IBAN/cards replace leading digits with "
        "`X` (keep last two); streets digit-mask; names keep first letter per token."
    )
    lines.append("")
    lines.append("## Counts per file")
    lines.append("")
    lines.extend(_count_table(state, files))
    lines.append("")
    lines.append(
        "Column notes: channel columns are total hits (placeholder + looks-real). "
        "`looks_real` / `synthetic` sum labelled hits across channels (a row may "
        "contribute multiple hits)."
    )
    lines.append("")
    lines.append("### Contact-channel label split (all files)")
    lines.append("")
    lines.extend(_contact_breakdown(state))
    lines.append("")
    lines.append("## Counts per source family")
    lines.append("")
    fam_lines = _family_table(state)
    if len(fam_lines) <= 2:
        lines.append("_No detector hits in any family._")
    else:
        lines.extend(fam_lines)
    lines.append("")
    lines.append("## Examples (20, redacted)")
    lines.append("")
    examples = state.select_examples()
    if not examples:
        lines.append("_No detector hits; no examples._")
    else:
        lines.append(
            f"Showing {len(examples)} unique redacted example(s) "
            f"(deduped by channel+label+redacted form; selected for channel coverage; "
            f"budget 20). Labels are `synthetic-placeholder` or `looks-real`."
        )
        lines.append("")
        for i, ex in enumerate(examples, 1):
            lines.append(
                f"{i}. **{ex.label}** · `{ex.channel}` · file `{ex.file}` · "
                f"family `{ex.family}` · field `{ex.field_path}` · "
                f"id `{ex.row_id}` → `{ex.redacted}`"
            )
        lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("From the repo root (no GPU; streams train.jsonl):")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python3 scripts/pii_scan.py --write docs/25-pii-scan.md"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "Optional: `--limit N` smoke-tests the first N rows per file; omit for the full scan. "
        f"Default files: {', '.join(f'`{f}`' for f in DEFAULT_FILES)}."
    )
    lines.append("")
    lines.append(
        f"Report date: {REPORT_DATE}. Script: `scripts/pii_scan.py`. "
        f"Name-list cache: `{CACHE_DIR}`."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit JSONL paths (default: the four sprint_v3d splits)",
    )
    p.add_argument("--limit", type=int, default=None, help="Max rows per file (smoke test)")
    p.add_argument(
        "--write",
        default=None,
        help="Write report markdown to this path (also always printed to stdout)",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="Repo root for resolving relative paths (default: cwd)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = os.path.abspath(args.repo_root or os.getcwd())
    files = args.files or DEFAULT_FILES
    files = [f if os.path.isabs(f) else os.path.join(root, f) for f in files]
    for f in files:
        if os.path.basename(f) == "test.jsonl" or f.endswith(FORBIDDEN):
            print(f"refusing sealed path: {f}", file=sys.stderr)
            return 2
        if not os.path.isfile(f):
            print(f"missing file: {f}", file=sys.stderr)
            return 2

    names = load_name_lists()
    state = ScanState(names=names, example_budget=20)
    for path in files:
        scan_file(path, state, limit=args.limit)

    report = render_report(
        state,
        files=[os.path.relpath(f, root) if f.startswith(root) else f for f in files],
        repo_root=root,
    )
    sys.stdout.write(report)
    if args.write:
        out = args.write if os.path.isabs(args.write) else os.path.join(root, args.write)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[wrote {out}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
