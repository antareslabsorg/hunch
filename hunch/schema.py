"""Data records (`hunch-data-v1`) and their validation.

One record = one question about one state, with one resolved target. Validation is strict and loud:
a malformed record raises instead of being silently dropped or repaired.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal

QType = Literal["choice", "boolean", "score"]
SPLITS = ("train", "dev", "calibration", "test", "heldout")
LICENSE_CLASSES = ("A", "B", "C", "D")
SEMANTIC_KINDS = ("event_probability", "label_distribution", "action_preference")
PROVENANCES = (
    "deterministic_truth",  # rule engine, executed code, verified fact
    "programmatic_conditional_distribution",  # exact q from a known random mechanism
    "observed_outcome",  # one realized outcome of an uncertain event
    "adjudicated_human_label",  # a single human/gold label under a published annotation spec
    "human_annotation_distribution",  # counts or fractions over several annotators
    "teacher_distribution",  # an open-weight teacher's distribution (imitation target)
    "hard_pseudo_label",  # a model's hard label without adjudication
)
BOOLEAN_IDS = ("false", "true")
BOOLEAN_DEFAULT_TEXT = {"false": "The proposition is false.", "true": "The proposition is true."}


@dataclass
class Candidate:
    id: str
    label: str
    description: str = ""

    def text(self) -> str:
        return f"{self.label}: {self.description}" if self.description else self.label


@dataclass
class Question:
    id: str
    type: QType
    instruction: str
    candidates: list[Candidate] = field(default_factory=list)  # choice only
    levels: list[str] = field(default_factory=list)  # score only, ordered
    criteria: dict[str, str] = field(default_factory=dict)  # boolean only: {"true": ..., "false": ...}

    def candidate_ids(self) -> list[str]:
        if self.type == "choice":
            return [c.id for c in self.candidates]
        if self.type == "boolean":
            return list(BOOLEAN_IDS)
        return [str(i) for i in range(len(self.levels))]

    def candidate_texts(self) -> list[str]:
        """The text of every candidate path, in candidate_ids() order. Never includes ids or indices."""
        if self.type == "choice":
            return [c.text() for c in self.candidates]
        if self.type == "boolean":
            return [self.criteria.get(k, BOOLEAN_DEFAULT_TEXT[k]) for k in BOOLEAN_IDS]
        return list(self.levels)

    @property
    def k(self) -> int:
        n = len(self.candidate_ids())
        if n == 0:  # an empty or unknown question: raise the reason instead of letting a caller divide by zero
            self.validate()
            raise ValueError(f"{self.id}: no candidates")
        return n

    def validate(self) -> None:
        if self.type not in ("choice", "boolean", "score"):
            raise ValueError(f"{self.id}: unknown question type {self.type!r}")
        if not self.instruction.strip():
            raise ValueError(f"{self.id}: empty instruction")
        if self.type == "choice":
            if not 2 <= len(self.candidates) <= 255:
                raise ValueError(f"{self.id}: choice needs 2..255 candidates, got {len(self.candidates)}")
            ids = [c.id for c in self.candidates]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{self.id}: duplicate candidate ids")
            if any(not c.label.strip() for c in self.candidates):
                raise ValueError(f"{self.id}: candidate with empty label")
        elif self.type == "score":
            if not 2 <= len(self.levels) <= 10:
                raise ValueError(f"{self.id}: score needs 2..10 levels, got {len(self.levels)}")
            if any(not lv.strip() for lv in self.levels):
                raise ValueError(f"{self.id}: empty level description")
        else:
            bad = set(self.criteria) - set(BOOLEAN_IDS)
            if bad:
                raise ValueError(f"{self.id}: boolean criteria keys must be true/false, got {bad}")


@dataclass
class Target:
    semantic_kind: str
    provenance: str
    probs: list[float]  # aligned with Question.candidate_ids(); sums to 1
    hard: int | None = None  # index of the hard label when one exists

    def validate(self, k: int, qid: str) -> None:
        if self.semantic_kind not in SEMANTIC_KINDS:
            raise ValueError(f"{qid}: unknown semantic_kind {self.semantic_kind!r}")
        if self.provenance not in PROVENANCES:
            raise ValueError(f"{qid}: unknown provenance {self.provenance!r}")
        if len(self.probs) != k:
            raise ValueError(f"{qid}: target has {len(self.probs)} probs for {k} candidates")
        if any((not math.isfinite(p)) or p < 0 or p > 1 for p in self.probs):
            raise ValueError(f"{qid}: probabilities must be finite and in [0,1]")
        s = math.fsum(self.probs)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"{qid}: probabilities sum to {s}, expected 1; no silent renormalization")
        if self.hard is not None and not 0 <= self.hard < k:
            raise ValueError(f"{qid}: hard label index {self.hard} out of range")


@dataclass
class Record:
    id: str
    source: str
    source_group: str
    family: str
    split: str
    license_class: str
    state: str | dict | list
    question: Question
    target: Target
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"{self.id}: unknown split {self.split!r}")
        if self.license_class not in LICENSE_CLASSES:
            raise ValueError(f"{self.id}: unknown license class {self.license_class!r}")
        if not self.source_group:
            raise ValueError(f"{self.id}: empty source_group (splits are assigned by group)")
        if isinstance(self.state, str):
            if not self.state.strip():
                raise ValueError(f"{self.id}: empty state")
        elif not isinstance(self.state, (dict, list)) or not self.state:
            raise ValueError(f"{self.id}: state must be nonempty str/dict/list")
        self.question.validate()
        self.target.validate(self.question.k, f"{self.id}/{self.question.id}")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)

    @staticmethod
    def from_dict(d: dict) -> "Record":
        q = d["question"]
        question = Question(
            id=q["id"],
            type=q["type"],
            instruction=q["instruction"],
            candidates=[Candidate(**c) for c in q.get("candidates", [])],
            levels=list(q.get("levels", [])),
            criteria=dict(q.get("criteria", {})),
        )
        t = d["target"]
        target = Target(semantic_kind=t["semantic_kind"], provenance=t["provenance"], probs=list(t["probs"]), hard=t.get("hard"))
        rec = Record(
            id=d["id"], source=d["source"], source_group=d["source_group"], family=d["family"], split=d["split"],
            license_class=d["license_class"], state=d["state"], question=question, target=target, meta=dict(d.get("meta", {})),
        )
        rec.validate()
        return rec


def one_hot(k: int, index: int) -> list[float]:
    return [1.0 if i == index else 0.0 for i in range(k)]


def write_jsonl(records: Iterator[Record] | list[Record], path: str | Path) -> int:
    n = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            r.validate()
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield Record.from_dict(json.loads(line))
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


if __name__ == "__main__":  # note: one runnable check
    q = Question(id="route", type="choice", instruction="Which handling path applies?",
                 candidates=[Candidate("a", "Approve", "Eligible."), Candidate("r", "Reject", "Not eligible.")])
    rec = Record(id="x", source="demo", source_group="g1", family="demo.route", split="train", license_class="A",
                 state={"ticket": "refund please"}, question=q,
                 target=Target("label_distribution", "deterministic_truth", one_hot(2, 0), hard=0))
    rec.validate()
    assert Record.from_dict(json.loads(rec.to_json())) == rec
    assert Question(id="b", type="boolean", instruction="Is it angry?").candidate_texts() == list(BOOLEAN_DEFAULT_TEXT.values())
    try:
        Target("label_distribution", "deterministic_truth", [0.6, 0.5]).validate(2, "bad")
        raise AssertionError("unit-sum violation accepted")
    except ValueError:
        pass
    print("schema ok")
