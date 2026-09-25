"""Evaluate a trained scorer checkpoint on a split; same output format as the frozen baseline.

    python -m hunch.evaluate --checkpoint runs/<run>/best --data data/sprint_v3d --split heldout --out results/heldout.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
import time
from pathlib import Path

import torch

from .batching import collate, pack_questions, tokenize_records
from .metrics import base_rates_from_records, summarize
from .model import group_logits, load_scorer, question_probs
from .schema import SPLITS, read_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="directory with model.safetensors (or 'none' for the untrained head)")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="heldout")
    p.add_argument("--max-per-family", type=int, default=500)
    p.add_argument("--token-budget", type=int, default=32768)
    p.add_argument("--max-path-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--amp", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    p.add_argument("--out", required=True)
    p.add_argument("--i-am-unsealing-the-test-split", action="store_true",
                   help="break the test-split seal; only for the single pre-publication read")
    args = p.parse_args()

    # SEAL, ENFORCED HERE (external audit, 2026-09-21). The project claimed the in-family `test`
    # split was "enforced in code"; it was not. The dispatcher refused it and the dashboard *detected*
    # reads after the fact, but this -- the primary evaluator, the thing that actually opens the file --
    # accepted any --split and read it. Detection is not enforcement. The guard belongs at the
    # execution boundary, so it is here.
    # The guard above compared `args.split == "test"` while the split is later interpolated straight into a
    # path (`Path(args.data) / f"{args.split}.jsonl"`). So `--split TEST` and `--split ../test` both missed the
    # comparison and still opened the sealed file; reproduced 2026-09-23. An exact-match denylist over a value
    # that becomes a path is the wrong shape. Allowlist the five real splits FIRST, so anything that is not one
    # of them cannot reach the open at all, and the seal check below then sees a canonical value.
    if args.split not in SPLITS:
        print(f"REFUSED: unknown split {args.split!r}; allowed: {', '.join(SPLITS)}. "
              "Split names are interpolated into a data path, so only exact known names are accepted.",
              file=sys.stderr)
        sys.exit(2)   # a refusal is a failure: nothing was written to --out
    UNSEAL = datetime(2026, 9, 25, 0, 0, tzinfo=timezone(timedelta(hours=2)))  # 2026-09-25 00:00 CEST
    if args.split == "test" and datetime.now(timezone.utc) < UNSEAL:
        if not args.i_am_unsealing_the_test_split:
            print(
                "REFUSED: the in-family `test` split is sealed until 2026-09-25 00:00 CEST and this is "
                f"{datetime.now(timezone.utc).isoformat()}.\n"
                "It is read exactly once, for the selected checkpoint, immediately before publication. "
                "Reading it now destroys that guarantee and cannot be undone.\n"
                "If this is that one read, pass --i-am-unsealing-the-test-split and record it in "
                "the project log (not published) in the same action.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"SEAL BROKEN DELIBERATELY: test split read at {datetime.now(timezone.utc).isoformat()} "
              f"for checkpoint {args.checkpoint}", file=sys.stderr)

    device = torch.device("cuda")
    if args.amp == "auto":
        amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else None
    else:
        amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[args.amp]
    scorer, tok = load_scorer(args.model, dtype=torch.float32)
    if args.checkpoint != "none":
        from safetensors.torch import load_file

        scorer.load_state_dict(load_file(str(Path(args.checkpoint) / "model.safetensors")), strict=True)
    scorer.to(device).eval()

    per_family: dict[str, int] = {}
    records = []
    for r in read_jsonl(Path(args.data) / f"{args.split}.jsonl"):
        if per_family.get(r.family, 0) >= args.max_per_family:
            continue
        per_family[r.family] = per_family.get(r.family, 0) + 1
        records.append(r)
    train_path = Path(args.data) / "train.jsonl"
    base_rates = base_rates_from_records(read_jsonl(train_path)) if train_path.exists() else None
    questions, skipped = tokenize_records(records, tok, args.max_path_tokens)

    preds, t0 = [], time.time()
    # A full-split bake is hours long and used to print nothing until it finished, so "alive" and
    # "wedged" looked identical from outside. Report progress on stderr every 30 s.
    n_target, last_report = len(questions), t0
    print(f"EVAL_BEGIN questions={n_target} split={args.split} data={args.data}", file=sys.stderr, flush=True)
    with torch.no_grad():
        for group in pack_questions(questions, args.token_budget):
            now = time.time()
            if now - last_report >= 30:
                done, rate = len(preds), len(preds) / max(now - t0, 1e-9)
                eta = (n_target - done) / rate if rate > 0 else float("nan")
                print(f"EVAL_PROGRESS {done}/{n_target} ({100 * done / max(n_target, 1):.1f}%) "
                      f"{rate:.1f} q/s eta={eta / 60:.0f}min", file=sys.stderr, flush=True)
                last_report = now
            b = collate(group, tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                z = scorer(b["input_ids"], b["attention_mask"], b["readout_idx"])
            zg_raw, valid = group_logits(z, b["group_sizes"])          # pre-temperature, for the archive
            zg, _ = group_logits(z / args.temperature, b["group_sizes"])
            pr = question_probs(zg, valid).cpu()
            lg = zg_raw.cpu()
            for i, q in enumerate(group):
                r = q.record
                # `logits` is archived because temperature scaling operates on logits: with only
                # post-softmax `probs` saved, no one -- including us -- can re-derive a fitted
                # temperature or any post-temperature calibration number. Three independent replays of
                # the release checkpoint's T gave 1.189 / 1.176 / 1.120 for exactly this reason.
                preds.append({"id": r.id, "probs": pr[i, : q.k].tolist(),
                              "logits": [round(float(v), 6) for v in lg[i, : q.k].tolist()],
                              "target": q.target, "group": r.source_group, "family": r.family,
                              "qtype": r.question.type, "known_q": r.target.provenance == "programmatic_conditional_distribution"})
    elapsed = time.time() - t0
    summary = summarize(preds, base_rates)
    summary.update({"checkpoint": args.checkpoint, "model": args.model, "split": args.split, "temperature": args.temperature, "skipped_too_long": skipped,
                    "seconds": round(elapsed, 1), "questions_per_s": round(len(preds) / max(elapsed, 1e-9), 2), "amp_dtype": str(amp_dtype)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "predictions": preds}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "families"}, indent=1))
    for f, m in sorted(summary["families"].items()):
        print(f"{f:45s} n={m['n']:5d} acc={m['accuracy']:.3f} nll={m['nll']:.3f} brier={m['brier']:.3f} smece={m['smece']:.3f}")


if __name__ == "__main__":
    main()
