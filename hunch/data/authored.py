"""Reader for the authored source specifications in docs/research/sources/*.md.

Each spec has '## <task>: exact instruction and ten paraphrases' sections with a numbered list (0 = exact instruction,
1..10 = paraphrases) and fenced JSON blocks listing candidates as {"id", "label", "description"}. This module parses
them into a small object; adapters use the descriptions as candidate text / Boolean criteria and the instruction
bank for training-time paraphrase augmentation (evaluation rows always use the exact instruction, index 0).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_NUM = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")


@dataclass
class Spec:
    path: Path
    instruction_banks: dict[str, list[str]] = field(default_factory=dict)  # heading -> [exact, paraphrases...]
    json_blocks: list[tuple[str, object]] = field(default_factory=list)  # (heading, parsed json)

    def instructions(self, section: str | None = None) -> list[str] | None:
        for heading, bank in self.instruction_banks.items():
            if section is None or section.lower() in heading.lower():
                if len(bank) >= 1:
                    return bank
        return None

    def candidates(self, section: str | None = None, index: int = 0) -> list[dict] | None:
        hits = [blk for heading, blk in self.json_blocks
                if isinstance(blk, list) and blk and isinstance(blk[0], dict) and "id" in blk[0]
                and (section is None or section.lower() in heading.lower())]
        return hits[index] if len(hits) > index else None

    def descriptions(self, section: str | None = None, index: int = 0) -> dict[str, str]:
        cands = self.candidates(section, index) or []
        return {str(c["id"]): str(c.get("description", "")).strip() for c in cands if c.get("description")}

    def criteria(self, section: str | None = None, index: int = 0) -> dict[str, str] | None:
        d = self.descriptions(section, index)
        return {"false": d["false"], "true": d["true"]} if {"false", "true"} <= set(d) else None


def parse_spec(path: str | Path) -> Spec:
    path = Path(path)
    spec = Spec(path)
    heading, in_json, buf, bank = "", False, [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if in_json:
            if line.strip().startswith("```"):
                in_json = False
                try:
                    spec.json_blocks.append((heading, json.loads("\n".join(buf))))
                except json.JSONDecodeError:
                    spec.json_blocks.append((heading, None))
                buf = []
            else:
                buf.append(line)
            continue
        if line.strip().startswith("```json"):
            in_json = True
            continue
        if line.startswith("## ") or line.startswith("### "):
            heading = line.lstrip("#").strip()
            bank = [] if "instruction" in heading.lower() else None
            if bank is not None:
                spec.instruction_banks[heading] = bank
            continue
        m = _NUM.match(line)
        if m and bank is not None:
            idx = int(m.group(1))
            if idx == len(bank):
                bank.append(m.group(2))
    return spec


_CACHE: dict[str, Spec | None] = {}
DIR: Path | None = None  # set by the build CLI (--authored); None disables authored inputs
PARAPHRASE = True  # set False by the build CLI (--no-paraphrase): descriptions/criteria only, adapters' default instructions on every split


def spec(name: str) -> Spec | None:
    """Spec for '<name>.md' under DIR, or None when authored inputs are disabled or the file is missing."""
    if DIR is None:
        return None
    if name not in _CACHE:
        p = Path(DIR) / f"{name}.md"
        _CACHE[name] = parse_spec(p) if p.exists() else None
    return _CACHE[name]


def pick_instruction(bank: list[str] | None, default: str, key: str, split: str) -> str:
    """Exact instruction for eval splits; a deterministic paraphrase (or the exact one) for training rows.
    With PARAPHRASE off the authored bank is ignored entirely: every split keeps the adapter's default instruction, so a
    build differs from a plain one only by the authored candidate descriptions and criteria."""
    if not bank or not PARAPHRASE:
        return default
    if split != "train" or len(bank) == 1:
        return bank[0]
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    return bank[h % len(bank)]  # index 0 (exact) included in the training mix


if __name__ == "__main__":  # note: parse the real pack if present
    import sys

    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "docs" / "research" / "sources"
    DIR = d
    for name in ("banking77", "clinc150", "massive-en", "bitext", "vitaminc", "boolq", "paws", "civil-comments", "aegis-2.0", "sst-5"):
        s = spec(name)
        if s is None:
            print(name, "missing")
            continue
        banks = {h: len(b) for h, b in s.instruction_banks.items()}
        blocks = [(h[:40], len(b) if isinstance(b, list) else type(b).__name__) for h, b in s.json_blocks]
        print(f"{name:15s} banks={banks} blocks={blocks[:4]}{'...' if len(blocks) > 4 else ''}")
    b = spec("banking77")
    assert b and len(b.instructions()) == 11 and len(b.descriptions()) == 77, (len(b.instructions()), len(b.descriptions()))
    assert spec("boolq").criteria() is not None
    print("authored ok")
