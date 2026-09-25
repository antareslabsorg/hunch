"""Shared-prefix ("tree") execution for Hunch: each state is read once per request instead of once per candidate.

`hunch.infer.Hunch.decide` scores every candidate as its own path (state + question + candidate), so a request with one
state and twenty candidates reads the state twenty times. The paths share their prefixes byte for byte by construction:
`hunch/serialize.py` tokenizes the state, question and candidate segments separately and concatenates the ids. So they can
run as one packed sequence, the state once, each question once, each candidate once, with a mask that lets every token see
exactly what it sees in its own path (the state, its own question, its own candidate up to itself), and with every token at
the position it has in that path. Each readout is then the same function of the same tokens at the same positions as in
the reference; the two differ only by floating-point rounding, which `scripts/check_fast.py` measures.

    from hunch.fast import FastHunch
    model = FastHunch.load(checkpoint, model="Qwen/Qwen3-0.6B", device="cuda")   # the arguments of Hunch.load
    model.decide(states)                                                          # the output of Hunch.decide

A state whose packed sequence exceeds the token budget falls back to the reference path for that request.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .infer import MAX_STATE_TOKENS, Hunch, choice_confidence, score_confidence
from .model import group_logits, question_probs
from .serialize import SERIALIZATION_VERSION, build_paths_batched


def pack_state(state_ids: list[int], questions: list[tuple[list[int], list[list[int]]]]):
    """One state's paths as a single sequence. `questions` holds (question_ids, [candidate_ids, ...]) in order.

    Returns the token ids, their positions (each token's index in its own reference path), a boolean visibility mask
    [L, L] (row = query, column = key) and the index of each candidate's readout token, candidates in question order."""
    ls = len(state_ids)
    ids, pos = list(state_ids), list(range(ls))
    kind, qid, cid = [0] * ls, [-1] * ls, [-1] * ls        # kind: 0 state, 1 question, 2 candidate
    readout, c = [], 0
    for q, (q_ids, cands) in enumerate(questions):
        ids += q_ids; pos += range(ls, ls + len(q_ids))
        kind += [1] * len(q_ids); qid += [q] * len(q_ids); cid += [-1] * len(q_ids)
        base = ls + len(q_ids)
        for c_ids in cands:
            ids += c_ids; pos += range(base, base + len(c_ids))
            kind += [2] * len(c_ids); qid += [q] * len(c_ids); cid += [c] * len(c_ids)
            readout.append(len(ids) - 1); c += 1
    kind, qid, cid = (torch.tensor(x) for x in (kind, qid, cid))
    t = torch.arange(len(ids))
    causal = t[None, :] <= t[:, None]
    ku, qu, cu = kind[None, :], qid[None, :], cid[None, :]
    visible = (ku == 0) | ((ku == 1) & (qu == qid[:, None])) | ((ku == 2) & (cu == cid[:, None]))
    return ids, pos, causal & visible, readout


def _answer(question, p: list[float]) -> dict:
    """The reference's answer shapes (hunch/infer.py, Hunch.decide), for one question's normalized probabilities."""
    ids, k = question.candidate_ids(), len(p)
    if question.type == "choice":
        return {"type": "choice", "probabilities": dict(zip(ids, p)), "choice": ids[max(range(k), key=p.__getitem__)], "confidence": choice_confidence(p)}
    if question.type == "boolean":
        return {"type": "boolean", "probability": p[1]}
    return {"type": "score", "probabilities": p, "score": sum(i * pi for i, pi in enumerate(p)), "level": max(range(k), key=p.__getitem__), "confidence": score_confidence(p)}


class FastHunch(Hunch):
    """Hunch with shared-prefix execution. Same checkpoint, same arguments, same output as `Hunch.decide`."""

    def _scores(self, ids, pos, mask, readout) -> torch.Tensor:
        """One packed state through the backbone once; the readout logits [len(readout)], fp32. The MLX builds override this."""
        dev = self.device
        with torch.autocast("cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None):
            hidden = self.scorer.backbone(input_ids=torch.tensor([ids], device=dev), position_ids=torch.tensor([pos], device=dev),
                                          attention_mask=mask[None, None].to(dev), use_cache=False).last_hidden_state
        h = hidden[0, torch.tensor(readout, device=dev)].float()
        with torch.autocast(device_type=h.device.type, enabled=False):   # the reference's fp32 readout (hunch/model.py)
            h_norm = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.scorer.norm.eps) * self.scorer.norm.weight.float()
            return F.linear(h_norm, self.scorer.head.weight.float()).squeeze(-1)

    @torch.no_grad()
    def decide(self, states: list[dict]) -> dict:
        if getattr(self.scorer, "set_head", None) is not None:
            raise NotImplementedError("the candidate-set head mixes paths; use hunch.infer.Hunch for such a checkpoint")
        t0 = time.time()
        recs = self._to_records(states)
        paths = build_paths_batched(recs, self.tok)
        by_state: dict[str, list] = {}
        for r, p in zip(recs, paths):
            if len(p.state_ids) > MAX_STATE_TOKENS:
                raise ValueError(f"{r.id}: state has {len(p.state_ids)} tokens > {MAX_STATE_TOKENS}")
            by_state.setdefault(r.source_group, []).append((r, p))
        packed = {}
        for sid, items in by_state.items():
            state_ids = items[0][1].state_ids
            if any(p.state_ids != state_ids for _, p in items):
                raise RuntimeError(f"state {sid}: its questions do not share the state's token ids")
            packed[sid] = pack_state(state_ids, [(p.question_ids, p.candidate_ids) for _, p in items])
            if len(packed[sid][0]) > self.token_budget:
                return super().decide(states)   # note: one oversized state sends the whole request down the reference path
        answers, n_tokens = {}, 0
        for sid, items in by_state.items():
            ids, pos, mask, readout = packed[sid]
            z = self._scores(ids, pos, mask, readout)
            n_tokens += len(ids)
            zg, valid = group_logits(z / self.T, [len(p.candidate_ids) for _, p in items])
            probs = question_probs(zg, valid).cpu()
            for i, (r, p) in enumerate(items):
                k = len(p.candidate_ids)
                v = [float(x) for x in probs[i, :k]]
                s = math.fsum(v)
                if not math.isfinite(s) or abs(s - 1) > 1e-4:
                    raise RuntimeError(f"{r.id}: probabilities sum to {s}")
                answers[r.id] = _answer(r.question, [x / s for x in v])
        results = []
        for si, s in enumerate(states):
            sid = str(s.get("id", si))
            results.append({"state_id": sid, "answers": {str(q.get("id", qi)): answers[f"{sid}/{str(q.get('id', qi))}"] for qi, q in enumerate(s["questions"])}})
        return {"serialization": SERIALIZATION_VERSION, "temperature": self.T, "results": results,
                "usage": {"questions": len(recs), "path_tokens": n_tokens}, "timing_ms": {"total": round((time.time() - t0) * 1000, 1)}}


if __name__ == "__main__":  # note: one runnable check of the mask on a toy layout
    ids, pos, m, ro = pack_state([1, 2, 3], [([4, 5], [[6, 7], [8]]), ([9], [[10, 11]])])
    assert ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] and pos == [0, 1, 2, 3, 4, 5, 6, 5, 3, 4, 5] and ro == [6, 7, 10]
    see = lambda t: [u for u in range(len(ids)) if m[t, u]]
    assert see(6) == [0, 1, 2, 3, 4, 5, 6]          # second token of candidate 1: state, its question, itself so far
    assert see(7) == [0, 1, 2, 3, 4, 7]             # candidate 2 of question 1 never sees candidate 1
    assert see(8) == [0, 1, 2, 8]                   # question 2 sees the state only
    assert see(10) == [0, 1, 2, 8, 9, 10]           # its candidate: the state, question 2, itself
    print("fast pack ok:", len(ids), "tokens packed; each readout sees exactly its own reference path")
