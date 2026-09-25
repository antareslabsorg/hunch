#!/usr/bin/env python3
"""Score a Hunch checkpoint on `pngwn/typed-decisions-v2-system-one`.

    python scripts/bench_td_v2_systemone.py --checkpoint <dir> --model <hf id> --out results.json

A second external suite, chosen because its schema maps onto our interface without interpretation:
`state` is free text, `question` is the instruction, `options` are the candidates, `answer_index`
is the gold. 5,214 test rows, every one `question_type: choice`, `ordered: False`, 2 to 10 options.

Unlike `typed-decisions`, the gold here is a **hard index**, not a distribution from a teacher. KL
against a soft gold is therefore undefined and is not reported; accuracy and Brier against the
one-hot are. The majority-class baseline on this split is **0.3506** and is printed beside our score,
because an accuracy without its base rate says nothing.
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

def to_request(row, i):
    opts=list(row["options"])
    return {"id": f"r{i}", "state": str(row["state"]),
            "questions": [{"id": "q", "type": "choice", "instruction": str(row["question"]),
                           "candidates": [{"id": str(j), "label": str(o), "description": str(o)} for j,o in enumerate(opts)]}]}

V2SO_REV = "4af44cdb3cb6e7a4374ef6a2e86bbc656e93d29e"


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--amp", default="bf16")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backbone-dtype", default="keep", choices=["keep","bf16"],
                    help="Hunch.load keeps weights in fp32 and autocasts compute, so a large backbone may not fit a smaller card. "
                         "'bf16' casts the backbone after load and leaves the fp32 readout alone. "
                         "Recorded in the output so the row is not read as the same regime as the others.")
    a=ap.parse_args()
    from huggingface_hub import hf_hub_download
    # pinned 2026-09-24: every earlier read was unpinned, but the test parquet has not changed since 2026-09-16 (3b618d9d uploaded
    # it; 4af44cdb, the only later commit, added row_map.jsonl), so all reads saw this revision's data
    df=pd.read_parquet(hf_hub_download("pngwn/typed-decisions-v2-system-one","data/test-00000-of-00001.parquet",repo_type="dataset",revision=V2SO_REV))
    if a.limit: df=df.head(a.limit)
    if a.dry_run:   # build every request with no model, so a schema bug cannot surface on a GPU
        n=sum(len(to_request(r,i)["questions"]) for i,(_,r) in enumerate(df.iterrows()))
        print(f"dry run ok: {len(df)} rows, {n} questions, no errors"); return
    from hunch.infer import Hunch
    h=Hunch.load(a.checkpoint, model=a.model, amp=a.amp, device=a.device); h.T=1.0; h.token_budget=a.token_budget
    if a.backbone_dtype=="bf16":
        import torch
        h.scorer.backbone.to(torch.bfloat16)   # norm + head stay fp32
    rows=[]; t0=time.time()
    for i,(_,r) in enumerate(df.iterrows()):
        ans=h.decide([to_request(r,i)])["results"][0]["answers"]["q"]
        k=len(r["options"]); p=np.array([float(ans["probabilities"][str(j)]) for j in range(k)])
        gold=int(r["answer_index"]); oh=np.zeros(k); oh[gold]=1.0
        rows.append({"task": str(r["task"]), "k": k, "correct": float(int(np.argmax(p))==gold),
                     "brier": float(((p-oh)**2).sum()), "nll": float(-np.log(max(p[gold],1e-12)))})
        if (i+1)%500==0: print(f"  {i+1}/{len(df)} {time.time()-t0:.0f}s", flush=True)
    def agg(sel):
        s=[r for r in rows if sel(r)]
        return None if not s else {"n":len(s),"accuracy":float(np.mean([x["correct"] for x in s])),
                                   "brier":float(np.mean([x["brier"] for x in s])),"nll":float(np.mean([x["nll"] for x in s]))}
    cnt=defaultdict(int)
    for _,r in df.iterrows(): cnt[int(r["answer_index"])]+=1
    out={"dataset":"pngwn/typed-decisions-v2-system-one","revision":V2SO_REV,"split":"test","checkpoint":a.checkpoint,
         "model":a.model,"temperature":1.0,"kind":"general, zero-shot","n":len(rows),
         "backbone_dtype":"bf16 weights (cast after load; fp32 readout)" if a.backbone_dtype=="bf16" else "fp32 weights, bf16 autocast (same as every other row)",
         "gold":"hard index (no soft distribution; KL not defined)",
         "majority_class_baseline": max(cnt.values())/len(df),
         "overall":agg(lambda r:True),
         "by_task":{t:agg(lambda r,t=t:r["task"]==t) for t in sorted({r["task"] for r in rows})},
         "by_k":{str(k):agg(lambda r,k=k:r["k"]==k) for k in sorted({r["k"] for r in rows})},
         "seconds":time.time()-t0}
    Path(a.out).write_text(json.dumps(out,indent=1)+"\n")
    print(json.dumps({"overall":out["overall"],"majority":out["majority_class_baseline"]},indent=1))

if __name__=="__main__": main()
