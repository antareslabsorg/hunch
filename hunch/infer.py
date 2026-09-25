"""In-process inference: load a checkpoint, answer decision requests (hunch/schema.py schema).

    from hunch.infer import Hunch
    h = Hunch.load("runs/<run>/best", model="Qwen/Qwen3-1.7B")
    out = h.decide([{"id": "s1", "state": {...}, "questions": [{"id": "route", "type": "choice", "instruction": "...",
                    "candidates": [{"id": "a", "label": "Approve"}, ...]}]}])

Every question gets a full distribution or the whole request fails; probabilities are validated to sum to 1.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .batching import TokenizedQuestion, collate, pack_questions
from .model import group_logits, load_scorer, question_probs
from .schema import Candidate, Question, Record, Target
from .serialize import SERIALIZATION_VERSION, build_paths_batched

MAX_STATE_TOKENS = 32768
MAX_QUESTIONS_PER_STATE = 64


def choice_confidence(p: list[float]) -> float:
    k = len(p)
    return 0.0 if k <= 1 else max(0.0, (max(p) - 1 / k) / (1 - 1 / k))


def score_confidence(p: list[float]) -> float:
    k = len(p)
    if k <= 1:
        return 1.0
    m = max(range(k), key=p.__getitem__)
    d = sum(pi * abs(i - m) for i, pi in enumerate(p))
    d_uniform = sum(abs(i - (k - 1) / 2) for i in range(k)) / k
    return max(0.0, 1 - d / d_uniform) if d_uniform > 0 else 1.0


class Hunch:
    def __init__(self, scorer, tokenizer, temperature: float, device, amp_dtype, token_budget: int = 32768):
        self.scorer, self.tok, self.T, self.device, self.amp_dtype, self.token_budget = scorer, tokenizer, temperature, device, amp_dtype, token_budget

    @classmethod
    def load(cls, checkpoint: str | Path, model: str = "Qwen/Qwen3-1.7B", device: str = "cuda", amp: str = "auto", token_budget: int = 32768) -> "Hunch":
        from safetensors.torch import load_file

        dev = torch.device(device)
        scorer, tok = load_scorer(model, dtype=torch.float32)
        ck = Path(checkpoint)
        scorer.load_state_dict(load_file(str(ck / "model.safetensors")), strict=True)
        scorer.to(dev).eval()
        cfg = json.loads((ck / "hunch_config.json").read_text()) if (ck / "hunch_config.json").exists() else {}
        if cfg.get("serialization") not in (None, SERIALIZATION_VERSION):
            raise ValueError(f"checkpoint serialization {cfg.get('serialization')} != runtime {SERIALIZATION_VERSION}")
        if amp == "auto":
            amp_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8 else None
        else:
            amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[amp]
        return cls(scorer, tok, float(cfg.get("temperature", 1.0)), dev, amp_dtype, token_budget)

    @staticmethod
    def _to_records(states: list[dict]) -> list[Record]:
        recs = []
        for si, s in enumerate(states):
            sid = str(s.get("id", si))
            qs = s.get("questions") or []
            if not 1 <= len(qs) <= MAX_QUESTIONS_PER_STATE:
                raise ValueError(f"state {sid}: {len(qs)} questions; allowed 1..{MAX_QUESTIONS_PER_STATE}")
            for qi, q in enumerate(qs):
                qid = str(q.get("id", qi))
                question = Question(
                    id=qid, type=q["type"], instruction=q.get("instruction", ""),
                    candidates=[Candidate(id=str(c["id"]), label=str(c.get("label", c["id"])), description=str(c.get("description", ""))) for c in q.get("candidates", [])],
                    levels=list(q.get("levels", [])), criteria={k: str(v) for k, v in (q.get("criteria") or {}).items()},
                )
                k = question.k
                rec = Record(id=f"{sid}/{qid}", source="request", source_group=sid, family="request", split="test", license_class="A",
                             state=s["state"], question=question, target=Target("label_distribution", "hard_pseudo_label", [1.0 / k] * k))
                rec.validate()
                recs.append(rec)
        return recs

    @torch.no_grad()
    def decide(self, states: list[dict]) -> dict:
        t0 = time.time()
        recs = self._to_records(states)
        paths = build_paths_batched(recs, self.tok)
        qs = []
        for r, p in zip(recs, paths):
            if len(p.state_ids) > MAX_STATE_TOKENS:
                raise ValueError(f"{r.id}: state has {len(p.state_ids)} tokens > {MAX_STATE_TOKENS}")
            qs.append(TokenizedQuestion(r, p.flat(), list(r.target.probs)))
        answers: dict[str, dict] = {}
        n_tokens = 0
        for group in pack_questions(qs, self.token_budget):
            b = collate(group, self.tok.pad_token_id, self.device)
            n_tokens += b["n_tokens"]
            with torch.autocast("cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None):
                z = self.scorer(b["input_ids"], b["attention_mask"], b["readout_idx"])
            zg, valid = group_logits(z / self.T, b["group_sizes"])
            probs = question_probs(zg, valid).cpu()
            for i, q in enumerate(group):
                p = [float(x) for x in probs[i, : q.k]]
                s = math.fsum(p)
                if not math.isfinite(s) or abs(s - 1) > 1e-4:
                    raise RuntimeError(f"{q.record.id}: probabilities sum to {s}")
                p = [x / s for x in p]
                ids = q.record.question.candidate_ids()
                typ = q.record.question.type
                if typ == "choice":
                    ans = {"type": "choice", "probabilities": dict(zip(ids, p)), "choice": ids[max(range(q.k), key=p.__getitem__)], "confidence": choice_confidence(p)}
                elif typ == "boolean":
                    ans = {"type": "boolean", "probability": p[1]}
                else:
                    ans = {"type": "score", "probabilities": p, "score": sum(i * pi for i, pi in enumerate(p)), "level": max(range(q.k), key=p.__getitem__), "confidence": score_confidence(p)}
                answers[q.record.id] = ans
        results = []
        for si, s in enumerate(states):
            sid = str(s.get("id", si))
            results.append({"state_id": sid, "answers": {str(q.get("id", qi)): answers[f"{sid}/{str(q.get('id', qi))}"] for qi, q in enumerate(s["questions"])}})
        return {"serialization": SERIALIZATION_VERSION, "temperature": self.T, "results": results,
                "usage": {"questions": len(recs), "path_tokens": n_tokens}, "timing_ms": {"total": round((time.time() - t0) * 1000, 1)}}
