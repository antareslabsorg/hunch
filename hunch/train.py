"""Train the flat decision scorer (sprint recipe, design notes, unpublished with sprint simplifications).

Single GPU:   python -m hunch.train --data DATA --out RUN [...]
Multi GPU:    torchrun --standalone --nproc_per_node=6 -m hunch.train --data DATA --out RUN [...]

- complete questions only; one global denominator per optimizer step; soft CE (+ optional CRPS for Score)
- fp32 master weights; fp16 (Turing) or bf16 (Ampere+) autocast; FSDP full-shard when world_size > 1
- deterministic per-epoch permutation sharded across ranks; resume reproduces the data order
- dev evaluation every --eval-every steps; checkpoint every --ckpt-every; best-by-dev-NLL kept
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .batching import cache_name, collate, pack_questions, subsample_candidates, tokenize_records
from .losses import brier, crps_ordinal, soft_cross_entropy
from .metrics import base_rates_from_records, summarize
from .model import group_logits, load_scorer, question_probs
from .schema import read_jsonl


def is_dist() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup() -> tuple[int, int, torch.device]:
    if is_dist():
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, torch.device("cuda", local)


def log0(rank: int, *a) -> None:
    if rank == 0:
        print(*a, flush=True)


def amp_dtype_for(choice: str, device):
    """auto -> bf16 on Ampere+, fp32 (no autocast) on older cards. fp16 overflows Qwen3 activations; use only deliberately."""
    if choice == "auto":
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else None
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[choice]


def autocast(amp_dtype):
    return torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None)


def wrap_fsdp(scorer, amp_dtype, device):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    import functools

    layer_cls = type(scorer.backbone.layers[0])
    return FSDP(
        scorer,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}),
        mixed_precision=MixedPrecision(param_dtype=amp_dtype, reduce_dtype=torch.float32, buffer_dtype=amp_dtype) if amp_dtype else None,
        sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=device, use_orig_params=True,
    )


def lr_at(step: int, total: int, warmup: int, peak: float, floor_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


@torch.no_grad()
def fits_budget(q, token_budget: int) -> bool:
    """pack_questions refuses a single question whose padded footprint exceeds the budget (by design: inference must
    never split one). Training and evaluation skip such questions instead, and count them, rather than crash a run."""
    return q.k * max(len(p) for p in q.paths) <= token_budget


@torch.no_grad()  # no backward follows: without this, eval mode (which disables gradient checkpointing) stores every
                  # layer's activations and a long dev microbatch exhausts the card; outputs are identical either way
def evaluate(model, questions, pad_id, device, amp_dtype, token_budget, rank, world, base_rates, max_questions=None):
    model.eval()
    shard = questions[rank::world]
    if max_questions:
        shard = shard[: max(1, max_questions // world)]
    over = sum(1 for q in shard if not fits_budget(q, token_budget))
    if over:
        shard = [q for q in shard if fits_budget(q, token_budget)]
        if rank == 0:
            print(json.dumps({"eval_skipped_over_budget": over, "token_budget": token_budget}), flush=True)
    preds = []
    groups = pack_questions(shard, token_budget)
    n_groups = torch.tensor(len(groups), device=device)
    if world > 1:
        dist.all_reduce(n_groups, op=dist.ReduceOp.MAX)  # equal forward counts across ranks (FSDP all-gathers)
    n_real = len(groups)
    groups = groups + [[shard[0]]] * (int(n_groups) - n_real)
    for gi, group in enumerate(groups):
        b = collate(group, pad_id, device)
        is_choice = [q.record.question.type == "choice" for q in group]
        with autocast(amp_dtype):
            z = model(b["input_ids"], b["attention_mask"], b["readout_idx"], b["group_sizes"], is_choice)
        if gi >= n_real:
            continue  # dummy pass, output discarded
        zg, valid = group_logits(z, b["group_sizes"])
        p = question_probs(zg, valid).cpu()
        for i, q in enumerate(group):
            r = q.record
            preds.append({"id": r.id, "probs": p[i, : q.k].tolist(), "target": q.target, "group": r.source_group, "family": r.family,
                          "qtype": r.question.type, "known_q": r.target.provenance == "programmatic_conditional_distribution"})
    if world > 1:
        gathered: list = [None] * world
        dist.all_gather_object(gathered, preds)
        preds = [x for part in gathered for x in part]
    model.train()
    return summarize(preds, base_rates, n_boot=0), preds


def save_checkpoint(model, opt, out: Path, tag: str, state: dict, rank: int, world: int) -> None:
    from safetensors.torch import save_file

    if world > 1:
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            sd = model.state_dict()
        osd = FSDP.optim_state_dict(model, opt) if state.get("save_optimizer") else None
    else:
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        osd = opt.state_dict() if state.get("save_optimizer") else None
    if rank == 0:
        d = out / tag
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in sd.items()}, str(d / "model.safetensors"))
        if osd is not None:
            torch.save(osd, d / "optimizer.pt")
        (d / "train_state.json").write_text(json.dumps({k: v for k, v in state.items() if k != "save_optimizer"}, indent=1))
    if world > 1:
        dist.barrier()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--questions-per-step", type=int, default=256, help="global logical batch (complete questions)")
    p.add_argument("--token-budget", type=int, default=32768, help="padded path tokens per microbatch per GPU")
    p.add_argument("--max-path-tokens", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--crps-weight", type=float, default=0.5, help="added CRPS weight for Score questions")
    p.add_argument("--brier-weight", type=float, default=0.0, help="added Brier score loss weight for calibration regularization")
    p.add_argument("--use-set-head", action="store_true", help="enable permutation-equivariant Choice set head")
    p.add_argument("--family-alpha", type=float, default=0.5, help="family sampling exponent: 1 = natural frequency, 0 = uniform over families")
    p.add_argument("--max-train-k", type=int, default=16, help="cap Choice candidates per training question via sampled negatives (0 = full K); eval always uses full K")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--amp", choices=["auto", "fp16", "bf16", "fp32"], default="auto", help="autocast dtype; auto = bf16 on Ampere+, fp32 on older GPUs")
    p.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw", help="adamw8bit (bitsandbytes) fits a 1.7B full fine-tune with fp32 master weights on one 32 GB card")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-max", type=int, default=4000)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="checkpoint dir to resume from")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--eval-token-budget", type=int, default=None, help="padded tokens per evaluation microbatch (default: --token-budget); evaluation holds no activations for backward, so it can afford a larger budget and score full-K questions whole")
    p.add_argument("--freeze-embeddings", action="store_true", help="keep the input embedding matrix fixed: no grads or optimizer state for it (~1.5 GB less for a 1.7B), for fine-tuning on a shared card; the scorer has no output head tied to it")
    args = p.parse_args()

    rank, world, device = setup()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    amp_dtype = amp_dtype_for(args.amp, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # rank 0 downloads/loads first so concurrent ranks never race on a half-written Hub cache
    if world > 1 and rank != 0:
        dist.barrier(device_ids=[device.index])
    scorer, tok = load_scorer(args.model, dtype=torch.float32, use_set_head=args.use_set_head)
    if world > 1 and rank == 0:
        dist.barrier(device_ids=[device.index])
    if args.grad_ckpt:
        scorer.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.resume:
        from safetensors.torch import load_file

        scorer.load_state_dict(load_file(str(Path(args.resume) / "model.safetensors")), strict=True)
    if args.freeze_embeddings:
        scorer.backbone.get_input_embeddings().weight.requires_grad_(False)
    scorer.to(device)
    model = wrap_fsdp(scorer, amp_dtype, device) if world > 1 else scorer
    model.train()

    # the scorer has exactly three parts: backbone, readout norm, head; everything outside the backbone is "head"
    head_params = [prm for n, prm in model.named_parameters() if "backbone" not in n]
    head_ids = {id(x) for x in head_params}
    body_params = [prm for prm in model.parameters() if id(prm) not in head_ids and prm.requires_grad]
    assert head_params and body_params, "parameter grouping failed"
    param_groups = [{"params": body_params, "lr": args.lr, "weight_decay": args.weight_decay},
                    {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0}]
    if args.optimizer == "adamw8bit":
        assert world == 1, "8-bit AdamW is for single-GPU runs (no FSDP); it keeps fp32 master weights with 8-bit moments"
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(param_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    if world > 1:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        scaler = ShardedGradScaler(enabled=amp_dtype == torch.float16)
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    # data
    t0 = time.time()
    train_records = list(read_jsonl(Path(args.data) / "train.jsonl"))
    dev_records = list(read_jsonl(Path(args.data) / "dev.jsonl"))
    base_rates = base_rates_from_records(train_records)
    caches = {s: cache_name(args.data, s, args.model, args.max_path_tokens) for s in ("train", "dev")}
    if rank == 0:  # tokenize once, share through the cache
        train_q, skipped_tr = tokenize_records(train_records, tok, args.max_path_tokens, caches["train"])
        dev_q, skipped_dev = tokenize_records(dev_records, tok, args.max_path_tokens, caches["dev"])
    if world > 1:
        dist.barrier()
    if rank != 0:
        train_q, skipped_tr = tokenize_records(train_records, tok, args.max_path_tokens, caches["train"])
        dev_q, skipped_dev = tokenize_records(dev_records, tok, args.max_path_tokens, caches["dev"])
    log0(rank, json.dumps({"train_questions": len(train_q), "skipped_train_too_long": skipped_tr, "dev_questions": len(dev_q),
                           "skipped_dev_too_long": skipped_dev, "tokenize_s": round(time.time() - t0, 1), "amp_dtype": str(amp_dtype), "world": world}))
    per_rank = max(1, args.questions_per_step // world)
    steps_per_epoch = len(train_q) // (per_rank * world)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    warmup = int(args.warmup_frac * total_steps)

    start_step = 0
    if args.resume and (Path(args.resume) / "train_state.json").exists():
        st = json.loads((Path(args.resume) / "train_state.json").read_text())
        start_step = st["step"]
        if (Path(args.resume) / "optimizer.pt").exists():
            osd = torch.load(Path(args.resume) / "optimizer.pt", map_location="cpu", weights_only=True)
            if world > 1:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                osd = FSDP.optim_state_dict_to_load(model, opt, osd)
            opt.load_state_dict(osd)
        log0(rank, f"resumed at step {start_step}")

    # family-balanced sampling: P(family) ∝ n_f^alpha (alpha=1 is natural frequency, 0 is uniform over families),
    # then uniform within the family; deterministic per (seed, step, rank) so resume reproduces the batches.
    fam_index: dict[str, list[int]] = {}
    for i, q in enumerate(train_q):
        fam_index.setdefault(q.record.family, []).append(i)
    fam_names = sorted(fam_index)
    fam_weights = torch.tensor([len(fam_index[f]) ** args.family_alpha for f in fam_names], dtype=torch.float64)
    fam_weights = fam_weights / fam_weights.sum()
    log0(rank, json.dumps({"family_sampling_probs": {f: round(float(w), 4) for f, w in zip(fam_names, fam_weights)}}))

    def batches_for_step(step: int) -> list:
        g = torch.Generator().manual_seed((args.seed * 1_000_003 + step) * 64 + rank)
        fams = torch.multinomial(fam_weights, per_rank, replacement=True, generator=g).tolist()
        out = []
        for f in fams:
            idxs = fam_index[fam_names[f]]
            q = train_q[idxs[int(torch.randint(len(idxs), (1,), generator=g))]]
            out.append(subsample_candidates(q, args.max_train_k, g) if args.max_train_k else q)
        return out

    best_dev, bad_streak, logf = float("inf"), 0, (out / "train_log.jsonl").open("a")
    if rank == 0:
        (out / "config.json").write_text(json.dumps({**vars(args), "total_steps": total_steps, "steps_per_epoch": steps_per_epoch, "world": world,
                                                     "amp_dtype": str(amp_dtype), "gpu": torch.cuda.get_device_name(device)}, indent=1))
    t_start, tokens_seen = time.time(), 0
    for step in range(start_step, total_steps):
        step_t0 = time.time()
        lr_scale = lr_at(step, total_steps, warmup, 1.0)
        for grp, base in zip(opt.param_groups, (args.lr, args.head_lr)):
            grp["lr"] = base * lr_scale
        mine = batches_for_step(step)
        n_local = torch.tensor(len(mine), device=device, dtype=torch.float32)
        if world > 1:
            dist.all_reduce(n_local)
        n_global = float(n_local)
        over = sum(1 for q in mine if not fits_budget(q, args.token_budget))
        if over:
            mine = [q for q in mine if fits_budget(q, args.token_budget)]
            if rank == 0:
                print(json.dumps({"step": step + 1, "train_skipped_over_budget": over}), flush=True)
        groups = pack_questions(mine, args.token_budget)
        # every rank must run the same number of forward/backward passes or FSDP collectives deadlock:
        # pad with zero-weight dummy microbatches (smallest question of this rank, loss multiplied by 0)
        n_groups = torch.tensor(len(groups), device=device)
        if world > 1:
            dist.all_reduce(n_groups, op=dist.ReduceOp.MAX)
        n_real = len(groups)
        dummy = [min(mine, key=lambda q: q.padded_tokens)]
        groups = groups + [dummy] * (int(n_groups) - n_real)
        loss_sum, step_tokens = 0.0, 0
        for gi, group in enumerate(groups):
            b = collate(group, tok.pad_token_id, device)
            weight = 1.0 if gi < n_real else 0.0
            step_tokens += b["n_tokens"] if weight else 0
            sync = gi == len(groups) - 1
            ctx = model.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            is_choice = [q.record.question.type == "choice" for q in group]
            with ctx:
                with autocast(amp_dtype):
                    z = model(b["input_ids"], b["attention_mask"], b["readout_idx"], b["group_sizes"], is_choice)
                zg, valid = group_logits(z, b["group_sizes"])
                losses = soft_cross_entropy(zg, b["target"], valid)
                if args.brier_weight > 0:
                    losses = losses + args.brier_weight * brier(zg, b["target"], valid)
                if args.crps_weight > 0:
                    is_score = torch.tensor([q.record.question.type == "score" for q in group], device=device)
                    if is_score.any():
                        losses = losses + args.crps_weight * crps_ordinal(zg, b["target"], valid) * is_score
                loss = weight * (world / n_global) * losses.sum()
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
        scaler.unscale_(opt)
        gnorm = model.clip_grad_norm_(1.0) if world > 1 else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(gnorm)) and amp_dtype != torch.float16:
            bad_streak += 1
            opt.zero_grad(set_to_none=True)
            log0(rank, f"step {step}: non-finite gradient, skipped ({bad_streak})")
            if bad_streak >= 3:
                raise RuntimeError("three consecutive non-finite steps")
            continue
        bad_streak = 0
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step_tokens_global = torch.tensor(step_tokens, device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(step_tokens_global)
        tokens_seen += int(step_tokens_global)
        now = time.time()
        step_s = now - step_t0
        if rank == 0:
            rec = {"step": step + 1, "loss": round(loss_sum, 5), "lr": opt.param_groups[0]["lr"], "grad_norm": round(float(gnorm), 4),
                   "tokens_step_local": step_tokens, "tokens_step_global": int(step_tokens_global), "microbatches": len(groups), "step_s": round(step_s, 2),
                   "tokens_per_s_step": round(int(step_tokens_global) / max(step_s, 1e-9)), "elapsed_s": round(now - t_start, 1),
                   "tokens_per_s_total": round(tokens_seen / max(1e-9, now - t_start)), "scale": float(scaler.get_scale()) if amp_dtype == torch.float16 else None}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if (step + 1) % 10 == 0 or step == start_step:
                print(json.dumps(rec), flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == total_steps:
            summ, preds = evaluate(model, dev_q, tok.pad_token_id, device, amp_dtype, args.eval_token_budget or args.token_budget, rank, world, base_rates, args.eval_max)
            if rank == 0:
                summ["step"] = step + 1
                (out / f"dev_step{step + 1}.json").write_text(json.dumps(summ, indent=1))
                print(json.dumps({"dev": {k: round(v, 4) for k, v in summ["overall"].items() if isinstance(v, float)}, "step": step + 1}), flush=True)
                for f, m in sorted(summ["families"].items()):
                    print(f"  {f:45s} n={m['n']:5d} acc={m['accuracy']:.3f} nll={m['nll']:.3f} brier={m['brier']:.3f} smece={m['smece']:.3f}", flush=True)
            dev_nll = summ["overall"]["nll"]
            if dev_nll < best_dev:
                best_dev = dev_nll
                save_checkpoint(model, opt, out, "best", {"step": step + 1, "dev_nll": dev_nll, "save_optimizer": False}, rank, world)
        if (step + 1) % args.ckpt_every == 0 or step + 1 == total_steps:
            save_checkpoint(model, opt, out, "last", {"step": step + 1, "save_optimizer": True}, rank, world)
    log0(rank, json.dumps({"done": True, "steps": total_steps, "best_dev_nll": best_dev, "hours": round((time.time() - t_start) / 3600, 2)}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
