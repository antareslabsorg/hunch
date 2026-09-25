"""Serialization of (state, question, candidate) into token paths.

Sprint variant `hunch-serialization-v1-plain`: plain-text section markers, no added special tokens.
Segments are tokenized separately and concatenated as ids so that every candidate path of a question shares
byte-identical state and question prefixes (tokenizing the joined string could merge across boundaries).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import Record

SERIALIZATION_VERSION = "hunch-serialization-v1-plain"

STATE_HEADER = "### State\n"
QUESTION_HEADER = "\n### Question\n"
CANDIDATE_HEADER = "\n### Candidate\n"
READOUT = "\n### Decision:"


def state_text(state: str | dict | list) -> str:
    """Canonical text for a state. Strings verbatim; JSON with preserved key order, 2-space indent, UTF-8."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)


def question_text(record: Record) -> str:
    q = record.question
    parts = [f"type: {q.type}\n", q.instruction.strip(), "\n"]
    if q.type == "boolean" and q.criteria:
        for key in ("false", "true"):
            if key in q.criteria:
                parts.append(f"{key}: {q.criteria[key].strip()}\n")
    return "".join(parts)


@dataclass
class TokenPaths:
    state_ids: list[int]
    question_ids: list[int]
    candidate_ids: list[list[int]]  # per candidate, includes the readout suffix
    version: str = SERIALIZATION_VERSION

    def flat(self) -> list[list[int]]:
        prefix = self.state_ids + self.question_ids
        return [prefix + c for c in self.candidate_ids]

    def readout_positions(self) -> list[int]:
        return [len(p) - 1 for p in self.flat()]


def _enc(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def build_paths(record: Record, tokenizer, max_path_tokens: int | None = None) -> TokenPaths:
    state_ids = _enc(tokenizer, STATE_HEADER + state_text(record.state))
    question_ids = _enc(tokenizer, QUESTION_HEADER + question_text(record))
    readout_ids = _enc(tokenizer, READOUT)
    cands = [_enc(tokenizer, CANDIDATE_HEADER + t.strip()) + readout_ids for t in record.question.candidate_texts()]
    paths = TokenPaths(state_ids, question_ids, cands)
    if max_path_tokens is not None:
        longest = max(len(p) for p in paths.flat())
        if longest > max_path_tokens:
            raise ValueError(f"{record.id}: longest candidate path is {longest} tokens > {max_path_tokens}; refusing to truncate")
    return paths


def build_paths_batched(records: list[Record], tokenizer, max_path_tokens: int | None = None) -> list[TokenPaths | None]:
    """Same output as build_paths per record, but with batched tokenizer calls (10-50x faster on large splits).
    Records whose longest path exceeds max_path_tokens get None."""
    state_texts = [STATE_HEADER + state_text(r.state) for r in records]
    q_texts = [QUESTION_HEADER + question_text(r) for r in records]
    cand_texts, spans = [], []
    for r in records:
        texts = r.question.candidate_texts()
        spans.append((len(cand_texts), len(cand_texts) + len(texts)))
        cand_texts.extend(CANDIDATE_HEADER + t.strip() for t in texts)
    readout_ids = _enc(tokenizer, READOUT)

    def enc_many(texts: list[str]) -> list[list[int]]:
        out: list[list[int]] = []
        for s in range(0, len(texts), 4096):
            out.extend(tokenizer(texts[s : s + 4096], add_special_tokens=False)["input_ids"])
        return out

    S, Q, C = enc_many(state_texts), enc_many(q_texts), enc_many(cand_texts)
    result: list[TokenPaths | None] = []
    for i, (a, b) in enumerate(spans):
        paths = TokenPaths(S[i], Q[i], [C[j] + readout_ids for j in range(a, b)])
        if max_path_tokens is not None and max(len(p) for p in paths.flat()) > max_path_tokens:
            result.append(None)
        else:
            result.append(paths)
    return result


if __name__ == "__main__":  # note: one runnable check with a real tokenizer if available
    from .schema import Candidate, Question, Target, one_hot

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as exc:  # noqa: BLE001
        print("tokenizer unavailable, skipping:", exc)
        raise SystemExit(0)
    q = Question(id="q", type="choice", instruction="Pick one.", candidates=[Candidate("a", "Alpha"), Candidate("b", "Beta")])
    rec = Record(id="r", source="demo", source_group="g", family="f", split="train", license_class="A",
                 state={"x": 1, "y": "two"}, question=q, target=Target("label_distribution", "deterministic_truth", one_hot(2, 1), 1))
    p = build_paths(rec, tok)
    flat = p.flat()
    prefix_len = len(p.state_ids) + len(p.question_ids)
    assert all(f[:prefix_len] == flat[0][:prefix_len] for f in flat), "shared prefix not byte-identical"
    assert tok.decode(flat[0]).endswith("### Decision:")
    batched = build_paths_batched([rec, rec], tok)
    assert batched[0] is not None and batched[0].flat() == flat and batched[1].flat() == flat, "batched tokenization differs from per-record"
    print("serialize ok:", [len(f) for f in flat], "tokens per path;", tok.decode(flat[1]))
