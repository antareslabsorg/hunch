"""Frozen-backbone baseline: no training, read the answer distribution from an instruct LM's logits.

Two readouts:
  letter : list options as "A. text", read next-token logits of the option letters at the answer position (K <= 26).
  cloze  : score each candidate text as a continuation ("Answer: <candidate>"), mean token log-prob, softmax over candidates (any K).

    python -m hunch.baselines.frozen --model Qwen/Qwen3-1.7B --data data/sprint_v0 --split heldout --out results/frozen_heldout.json
"""
from __future__ import annotations

import argparse
import json
import math
import string
import sys
import time
from pathlib import Path

import torch

from ..metrics import base_rates_from_records, summarize
from ..model import _disable_cudnn_sdpa_on_unsupported_arch  # noqa: F401  (runs at import)
from ..schema import Record, read_jsonl
from ..serialize import state_text

LETTERS = list(string.ascii_uppercase)


def build_prompt(rec: Record, readout: str) -> tuple[str, list[str]]:
    q = rec.question
    texts = q.candidate_texts()
    header = f"{state_text(rec.state)}\n\nQuestion: {q.instruction.strip()}\n"
    if q.type == "boolean" and q.criteria:
        header += f"(false: {q.criteria.get('false', '')} | true: {q.criteria.get('true', '')})\n"
    if readout == "letter":
        opts = "\n".join(f"{LETTERS[i]}. {t}" for i, t in enumerate(texts))
        return header + "Options:\n" + opts + "\nAnswer with the letter of the best option.", texts
    return header + "Choose the best answer from the offered options and answer with that option's exact text.", texts


def _left_pad(seqs: list[list[int]], pad_id: int, device):
    """Left-pad so every sequence ends at the last column; returns ids, mask, position_ids."""
    width = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, x in enumerate(seqs):
        ids[i, width - len(x) :] = torch.tensor(x)
        mask[i, width - len(x) :] = True
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


@torch.no_grad()
def score_letter(model, tok, prompts: list[str], ks: list[int], device) -> list[list[float]]:
    letter_ids = [tok.encode(f" {L}", add_special_tokens=False) for L in LETTERS]
    assert all(len(x) == 1 for x in letter_ids[:26]), "letters must be single tokens with a leading space"
    letter_ids = torch.tensor([x[0] for x in letter_ids], device=device)
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "The answer is" for p in prompts]
    ids, mask, pos = _left_pad([tok.encode(c, add_special_tokens=False) for c in chats], tok.pad_token_id, device)
    logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1).logits[:, -1, :]  # last position only
    out = []
    for i, k in enumerate(ks):
        z = logits[i, letter_ids[:k]].float()
        out.append(torch.softmax(z, -1).tolist())
    return out


@torch.no_grad()
def score_cloze(model, tok, prompt: str, texts: list[str], device, batch: int = 8) -> list[float]:
    """Mean log-prob of each candidate text as a continuation. Left-padded so candidates end at the last column and
    only the last (max_candidate_len + 1) positions' logits are materialized."""
    chat = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Answer:"
    prefix = tok.encode(chat, add_special_tokens=False)
    cands = [tok.encode(" " + t.strip(), add_special_tokens=False) for t in texts]
    scores = []
    for s in range(0, len(cands), batch):
        chunk = cands[s : s + batch]
        max_n = max(len(c) for c in chunk)
        ids, mask, pos = _left_pad([prefix + c for c in chunk], tok.pad_token_id, device)
        keep = max_n + 1
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=keep).logits.float(), -1)  # [B, keep, V]
        for i, c in enumerate(chunk):
            n = len(c)
            tgt = torch.tensor(c, device=device)
            lp = logp[i, max_n - n : max_n].gather(-1, tgt[:, None]).squeeze(-1)  # logits at t-1 predict token t
            scores.append(float(lp.mean()))  # length-normalized
    z = torch.tensor(scores)
    return torch.softmax(z, -1).tolist()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data", required=True, help="directory with <split>.jsonl")
    p.add_argument("--split", default="heldout")
    p.add_argument("--train-split-for-base-rates", default="train")
    p.add_argument("--readout", choices=["letter", "cloze", "auto"], default="auto", help="auto = letter when K<=26 else cloze")
    p.add_argument("--max-per-family", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto", help="weights dtype; auto = bf16 on Ampere+, fp32 on older GPUs (fp16 overflows Qwen3)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    per_family: dict[str, int] = {}
    records = []
    for r in read_jsonl(Path(args.data) / f"{args.split}.jsonl"):
        if per_family.get(r.family, 0) >= args.max_per_family:
            continue
        per_family[r.family] = per_family.get(r.family, 0) + 1
        records.append(r)
    base_rates = None
    train_path = Path(args.data) / f"{args.train_split_for_base_rates}.jsonl"
    if train_path.exists():
        base_rates = base_rates_from_records(read_jsonl(train_path))

    preds, t0 = [], time.time()
    letter_batch: list[tuple[Record, str, list[str]]] = []

    def flush_letters():
        if not letter_batch:
            return
        probs = score_letter(model, tok, [pr for _, pr, _ in letter_batch], [len(t) for _, _, t in letter_batch], device)
        for (rec, _, _), pr in zip(letter_batch, probs):
            preds.append({"id": rec.id, "probs": pr, "target": rec.target.probs, "group": rec.source_group, "family": rec.family,
                          "qtype": rec.question.type, "known_q": rec.target.provenance == "programmatic_conditional_distribution"})
        letter_batch.clear()

    for rec in records:
        readout = args.readout if args.readout != "auto" else ("letter" if rec.question.k <= 26 else "cloze")
        prompt, texts = build_prompt(rec, readout)
        if readout == "letter":
            letter_batch.append((rec, prompt, texts))
            if len(letter_batch) >= args.batch:
                flush_letters()
        else:
            pr = score_cloze(model, tok, prompt, texts, device)
            preds.append({"id": rec.id, "probs": pr, "target": rec.target.probs, "group": rec.source_group, "family": rec.family,
                          "qtype": rec.question.type, "known_q": rec.target.provenance == "programmatic_conditional_distribution"})
    flush_letters()
    elapsed = time.time() - t0
    summary = summarize(preds, base_rates)
    summary.update({"model": args.model, "split": args.split, "readout": args.readout, "seconds": round(elapsed, 1),
                    "questions_per_s": round(len(preds) / max(elapsed, 1e-9), 2), "device": str(device), "dtype": str(dtype)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "predictions": preds}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "families"}, indent=1))
    for f, m in summary["families"].items():
        print(f"{f:45s} n={m['n']:5d} acc={m['accuracy']:.3f} nll={m['nll']:.3f} brier={m['brier']:.3f} smece={m['smece']:.3f}")


if __name__ == "__main__":
    main()
