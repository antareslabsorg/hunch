"""The deliverable: does each on-device artifact reproduce the reference distributions?

    python -m hunch.formats.equivalence_test --reference --checkpoint <checkpoint dir> [--device cpu|mps] [--limit 201|0]
    python -m hunch.formats.equivalence_test --gguf hunch/formats/out/hunch-0.6b-preview.f16.gguf [--limit N]
    python -m hunch.formats.equivalence_test --mlx hunch/formats/out/hunch-0.6b-preview-mlx-f16 [--limit N]
    python -m hunch.formats.equivalence_test --report
(HUNCH_BACKBONE picks the size; see hunch/formats/common.py.)

Ground truth is the archive of 6,000 held-out predictions of the release checkpoint (bf16 autocast on a GPU, T = 1),
plus a PyTorch fp32 run of the same checkpoint made here: the noise floor, and the source of logits and hidden vectors
for localising a disagreement. Every scored run writes per-question logits and probs to out/preds/<name>.jsonl
(resumable) and the readout vectors of a fixed family-balanced 201-question subset to out/preds/<name>.hidden.npz;
--report recomputes the report from those files. Failures are printed, never filtered.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from ..batching import collate, pack_questions, tokenize_records
from ..infer import Hunch
from ..model import group_logits, question_probs
from ..schema import Record, read_jsonl
from ..serialize import CANDIDATE_HEADER, QUESTION_HEADER, READOUT, STATE_HEADER, question_text, state_text
from .common import ARCHIVE, BACKBONE, NAME, OUT, REPO, TEMPERATURE, Backbone, readout, split_path

# smooth_ece / calibration_pairs / temper / metrics are the gate script's own, imported rather than re-implemented
_spec = importlib.util.spec_from_file_location("compare_results", REPO / "scripts" / "compare_results.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

# Predictions are scoped BY BACKBONE. Without this a 0.6B run finds the 1.7B's `gguf-f16.jsonl`,
# treats those question ids as already scored, writes nothing, and the report then presents 1.7B
# numbers as the 0.6B's. Seen 2026-09-23 03:14: every artifact "done; 0 questions in 0s".
PREDS = OUT / ("preds" if NAME == "hunch-1.7b-preview" else f"preds-{NAME.split(chr(45))[1]}")
BARS = {  # level -> (max TV, min argmax agreement): the original fixed bars, reported but superseded by floor_gate
    "f16": (1e-3, 1.0), "Q8_0": (5e-3, 1.0), "8bit": (5e-3, 1.0), "Q4_K_M": (2e-2, 0.995), "4bit": (2e-2, 0.995)}
CAL_TOL = NLL_TOL = 0.005
TEMPS = (1.0, TEMPERATURE)
SUBSET_N, SUBSET_SEED = 201, 0  # 67 questions per family keep their readout vectors, for localisation
MAX_PATH_TOKENS = 2048          # hunch.evaluate's default, which produced the archive


# ----------------------------------------------------------------------------- data

def archive() -> list[dict]:
    rows = json.loads(ARCHIVE.read_text())["predictions"]
    if len(rows) != 6000 or len({r["id"] for r in rows}) != 6000:
        raise SystemExit(f"{ARCHIVE}: expected 6000 unique predictions, found {len(rows)}")
    return rows


def subset(rows: list[dict], n: int = SUBSET_N, seed: int = SUBSET_SEED) -> list[str]:
    """A fixed, family-balanced sample of archived question ids (n rounded up to a multiple of the family count)."""
    rng = np.random.default_rng(seed)
    fams = sorted({r["family"] for r in rows})
    per = -(-n // len(fams))
    out = []
    for f in fams:
        ids = [r["id"] for r in rows if r["family"] == f]
        out += [ids[j] for j in sorted(rng.permutation(len(ids))[:per])]
    return out


def records_for(ids: list[str]) -> list[Record]:
    want, found = set(ids), {}
    for r in read_jsonl(split_path("heldout")):
        if r.id in want:
            found[r.id] = r
    missing = want - found.keys()
    if missing:
        raise SystemExit(f"{len(missing)} archived ids are not in heldout.jsonl, e.g. {sorted(missing)[:3]}")
    return [found[i] for i in ids]


def questions_for(ids: list[str], tok):
    qs, skipped = tokenize_records(records_for(ids), tok, MAX_PATH_TOKENS)
    if skipped:
        raise SystemExit(f"{skipped} archived questions exceed {MAX_PATH_TOKENS} path tokens; the archive scored them, so this cannot be")
    return qs


# ----------------------------------------------------------------------------- scoring

class TorchBackbone(Backbone):
    """The reference itself (hunch.infer.Hunch in fp32), instrumented to expose the hidden state before and after the
    backbone's final norm. Logits come from the unmodified DecisionScorer.forward; `selfcheck_max` is how far this
    package's fp32 readout is from it."""

    def __init__(self, checkpoint_dir: str, device: str):
        self.h = Hunch.load(checkpoint_dir, model=BACKBONE, device=device, amp="fp32")
        sc = self.h.scorer
        self.norm_w = sc.norm.weight.detach().float().cpu()
        self.head_w = sc.head.weight.detach().float().cpu().reshape(-1)
        self.device = torch.device(device)
        self.meta = {"backbone": BACKBONE, "temperature": TEMPERATURE}
        self.settings = {"device": device, "dtype": "fp32", "torch": torch.__version__}
        self.selfcheck_max = 0.0
        self._pre = self._post = None
        sc.backbone.norm.register_forward_hook(lambda m, inp, out: setattr(self, "_pre", inp[0]) or setattr(self, "_post", out))

    @torch.no_grad()
    def hidden(self, input_ids, attention_mask, readout_idx):
        ids, mask, ro = input_ids.to(self.device), attention_mask.to(self.device), readout_idx.to(self.device)
        z = self.h.scorer(ids, mask, ro).float().cpu()  # the reference forward, unmodified
        ar = torch.arange(ids.shape[0], device=self.device)
        h, self.prenorm = self._post[ar, ro].float().cpu(), self._pre[ar, ro].float().cpu()
        self.selfcheck_max = max(self.selfcheck_max, float((readout(h, self.norm_w, self.head_w) - z).abs().max()))
        return h


def score(name: str, backbone: Backbone, qs, budget: int, keep: set[str], pad_id: int) -> None:
    """Score questions, appending per-question logits/probs to out/preds/<name>.jsonl; resumes; keeps hidden vectors for `keep`."""
    PREDS.mkdir(parents=True, exist_ok=True)
    path, hid = PREDS / f"{name}.jsonl", PREDS / f"{name}.hidden.npz"
    done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
    todo = [q for q in qs if q.record.id not in done]
    H: dict[str, list] = {"ids": [], "sizes": [], "h": [], "pre": []}
    if hid.exists():
        old = np.load(hid)
        H["ids"], H["sizes"], H["h"] = list(old["ids"]), list(old["sizes"]), [old["h"]]
        H["pre"] = [old["pre"]] if "pre" in old else []
    print(f"{name}: {len(done)} questions cached, {len(todo)} to score", file=sys.stderr, flush=True)
    t0 = last = time.time()
    n_q = n_p = n_t = n_bad = 0
    with path.open("a") as f:
        for group in pack_questions(todo, budget):
            b = collate(group, pad_id, "cpu")
            h = backbone.hidden(b["input_ids"], b["attention_mask"], b["readout_idx"])
            pre = getattr(backbone, "prenorm", None)
            z = readout(h, backbone.norm_w, backbone.head_w)
            zg, valid = group_logits(z, b["group_sizes"])
            p = question_probs(zg, valid)
            off = 0
            for i, q in enumerate(group):
                k, qid = q.k, q.record.id
                zi = z[off:off + k]
                row = {"id": qid, "logits": zi.tolist(), "probs": p[i, :k].tolist()}
                if not torch.isfinite(zi).all():
                    row["nonfinite"] = True
                    n_bad += 1
                f.write(json.dumps(row) + "\n")
                if qid in keep:
                    H["ids"].append(qid); H["sizes"].append(k); H["h"].append(h[off:off + k].numpy())
                    if pre is not None:
                        H["pre"].append(pre[off:off + k].numpy())
                off += k
            f.flush()
            n_q += len(group); n_p += len(z); n_t += int(b["n_tokens"])
            if time.time() - last >= 30:
                el = time.time() - t0
                print(f"{name}: {n_q}/{len(todo)} q  {n_p / el:.1f} paths/s  {n_t / el:.0f} tok/s  "
                      f"eta {(len(todo) - n_q) / max(n_q / el, 1e-9) / 60:.0f} min  non-finite {n_bad}", file=sys.stderr, flush=True)
                last = time.time()
    if H["ids"]:
        arrays = {"ids": np.array(H["ids"]), "sizes": np.array(H["sizes"]), "h": np.concatenate(H["h"]),
                  "norm_w": backbone.norm_w.numpy(), "head_w": backbone.head_w.numpy()}
        if H["pre"]:
            arrays["pre"] = np.concatenate(H["pre"])
        np.savez(hid, **arrays)
    meta_path = PREDS / f"{name}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"settings": getattr(backbone, "settings", {}), "questions_scored_this_run": n_q, "paths": meta.get("paths", 0) + n_p,
                 "tokens": meta.get("tokens", 0) + n_t, "seconds": round(meta.get("seconds", 0) + time.time() - t0, 1), "nonfinite": meta.get("nonfinite", 0) + n_bad})
    if hasattr(backbone, "selfcheck_max"):
        meta["readout_selfcheck_max_abs"] = backbone.selfcheck_max
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"{name}: done; {n_q} questions in {time.time() - t0:.0f}s; non-finite logits in {n_bad} questions", file=sys.stderr, flush=True)


def tokenizer_parity(backbone, records: list[Record], tok) -> dict:
    """llama.cpp's tokenizer on the GGUF against the HF tokenizer, segment by segment as hunch/serialize.py tokenizes."""
    n = bad = 0
    examples = []
    for r in records:
        segs = [STATE_HEADER + state_text(r.state), QUESTION_HEADER + question_text(r), READOUT]
        segs += [CANDIDATE_HEADER + t.strip() for t in r.question.candidate_texts()]
        for s in segs:
            n += 1
            a, b = tok.encode(s, add_special_tokens=False), backbone.tokenize(s)
            if a != b:
                bad += 1
                if len(examples) < 5:
                    examples.append({"text": s[:80], "hf": a[:12], "llama_cpp": b[:12]})
    return {"segments": n, "mismatched": bad, "examples": examples}


# ----------------------------------------------------------------------------- report

def artifact_name(path: Path, tag: str = "") -> str:
    """gguf-<level> / mlx-<level>, plus '+<tag>' for a run on another backend (the untagged name is the laptop backend)."""
    if path.suffix == ".gguf":
        base = "gguf-" + path.name.removeprefix(NAME + ".").removesuffix(".gguf")
    else:
        base = "mlx-" + path.name.split("-mlx-")[-1]
    return f"{base}+{tag}" if tag else base


def level_of(name: str) -> str | None:
    fmt, _, rest = name.partition("-")
    return rest.split("+")[0] if fmt in ("gguf", "mlx") else None


def load_rows(name: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, (PREDS / f"{name}.jsonl").open())}


def matrix(rows: dict[str, dict], ids: list[str], kmax: int, key: str = "probs") -> np.ndarray:
    m = np.zeros((len(ids), kmax))
    for i, q in enumerate(ids):
        p = rows[q][key]
        m[i, :len(p)] = p
    return m


def compare(p_hat: np.ndarray, p_ref: np.ndarray, t: np.ndarray, fam: np.ndarray, qt: list[str], ids: list[str]) -> dict:
    """Statistics of p_hat against p_ref on the same questions; non-finite rows are counted, then excluded."""
    finite = np.isfinite(p_hat).all(-1)
    out = {"n": int(len(ids)), "n_nonfinite": int((~finite).sum())}
    ph, pr, tt, ff = p_hat[finite], p_ref[finite], t[finite], fam[finite]
    qq, ii = [q for q, m in zip(qt, finite) if m], [q for q, m in zip(ids, finite) if m]
    tv = 0.5 * np.abs(ph - pr).sum(-1)
    agree = ph.argmax(-1) == pr.argmax(-1)
    out.update(tv_max=float(tv.max()), tv_mean=float(tv.mean()), tv_p99=float(np.percentile(tv, 99)),
               argmax_agree=float(agree.mean()), n_argmax_diff=int((~agree).sum()),
               worst=[{"id": ii[i], "tv": float(tv[i]), "p_hat": ph[i].round(4).tolist(), "p_ref": pr[i].round(4).tolist()} for i in np.argsort(-tv)[:3]],
               argmax_diff_ids=[ii[i] for i in np.where(~agree)[0][:10]])
    cal, fams = {}, sorted(set(ff))
    for T in TEMPS:
        qh, qr = cr.temper(ph, T), cr.temper(pr, T)
        for name, mask in [("pooled", np.ones(len(ph), bool))] + [(f, ff == f) for f in fams]:
            qm = [q for q, m in zip(qq, mask) if m]
            sm_h = cr.smooth_ece(*cr.calibration_pairs(qh[mask], tt[mask], qm))
            sm_r = cr.smooth_ece(*cr.calibration_pairs(qr[mask], tt[mask], qm))
            cal[f"{name}@T={T}"] = {"smece": float(sm_h), "smece_ref": float(sm_r), "delta": float(sm_h - sm_r)}
        nll_h, nll_r = cr.metrics(qh, tt)[0].mean(), cr.metrics(qr, tt)[0].mean()
        out[f"nll@T={T}"] = {"nll": float(nll_h), "nll_ref": float(nll_r), "delta": float(nll_h - nll_r)}
    out["calibration"] = cal
    out["max_abs_dsmece"] = max(abs(v["delta"]) for v in cal.values())
    return out


def bars_for(level: str) -> tuple[float, float] | None:
    """Fixed bars exist for f16, Q8/8-bit and Q4/4-bit; variants such as Q4_K_M-imat inherit their family's bar, Q5/Q6 have none."""
    return next((v for k, v in BARS.items() if level.startswith(k)), None)


def floor_gate(c: dict, floor: dict) -> dict:
    """FORMATS.md's rule: an artifact passes if its max TV against fp32 is within the floor (bf16 against fp32 on the same
    checkpoint) and its disagreement rate is not significantly above the floor's (two-proportion z, one-sided 95 %).
    "pass" = significantly below the floor rate, "at floor" = indistinguishable from it, "no" = above it on either statistic."""
    n, d, fn, fd = c["n"] - c["n_nonfinite"], c["n_argmax_diff"], floor["n"] - floor["n_nonfinite"], floor["n_argmax_diff"]
    pool = (d + fd) / (n + fn)
    se = (pool * (1 - pool) * (1 / n + 1 / fn)) ** 0.5
    z = (d / n - fd / fn) / se if se else float("inf")
    ok = c["tv_max"] <= floor["tv_max"] and z <= 1.645 and c["n_nonfinite"] == 0
    return {"floor_tv_max": floor["tv_max"], "floor_rate": fd / fn, "x_floor_tv": c["tv_max"] / floor["tv_max"], "rate": d / n,
            "x_floor_rate": (d / n) / (fd / fn) if fd else float("inf"), "z": z, "verdict": "no" if not ok else ("pass" if z <= -1.645 else "at floor")}


def verdict(level: str, c: dict) -> dict | None:
    bars = bars_for(level)
    if bars is None:
        return None
    tv_bar, arg_bar = bars
    v = {"tv": c["tv_max"] <= tv_bar, "argmax": c["argmax_agree"] >= arg_bar, "calibration": c["max_abs_dsmece"] <= CAL_TOL,
         "nll": all(abs(c[f"nll@T={T}"]["delta"]) <= NLL_TOL for T in TEMPS), "finite": c["n_nonfinite"] == 0}
    v["all_fixed_bars"] = all(v.values())
    return v


def regroup_tv_max(rows: dict[str, dict]) -> float:
    """Stored probs against the reference softmax of the stored logits: a non-zero value means the wrapper grouped wrongly."""
    worst = 0.0
    for r in rows.values():
        z = torch.tensor(r["logits"])
        if torch.isfinite(z).all():
            p = question_probs(*group_logits(z, [len(z)]))[0]
            worst = max(worst, float(0.5 * (p - torch.tensor(r["probs"])).abs().sum()))
    return worst


def _index(d) -> dict[str, tuple[int, int]]:
    out, off = {}, 0
    for i, k in zip(d["ids"], d["sizes"]):
        out[str(i)] = (off, off + int(k))
        off += int(k)
    return out


def localise(name: str, ref: str) -> dict:
    """Where an artifact departs from the fp32 reference on the subset: readout vector, then logit."""
    if not ((PREDS / f"{name}.hidden.npz").exists() and (PREDS / f"{ref}.hidden.npz").exists()):
        return {}
    a, r = np.load(PREDS / f"{name}.hidden.npz"), np.load(PREDS / f"{ref}.hidden.npz")
    ia, ir = _index(a), _index(r)
    common = [q for q in ia if q in ir]
    if not common:
        return {}
    ha = np.concatenate([a["h"][slice(*ia[q])] for q in common])
    hr = np.concatenate([r["h"][slice(*ir[q])] for q in common])
    nw, hw = torch.from_numpy(r["norm_w"]), torch.from_numpy(r["head_w"])
    za, zr = readout(torch.from_numpy(ha), nw, hw).numpy(), readout(torch.from_numpy(hr), nw, hw).numpy()
    rel = np.linalg.norm(ha - hr, axis=-1) / np.linalg.norm(hr, axis=-1)
    out = {"n_questions": len(common), "n_paths": int(len(rel)),
           "hidden_rel_l2_max": float(rel.max()), "hidden_rel_l2_mean": float(rel.mean()),
           "logit_abs_max": float(np.abs(za - zr).max()), "logit_abs_mean": float(np.abs(za - zr).mean())}
    if "pre" in r:  # is the artifact's vector the post-norm hidden state (small) or the pre-norm one (large)?
        pre = np.concatenate([r["pre"][slice(*ir[q])] for q in common])
        out["hidden_rel_l2_mean_vs_prenorm"] = float((np.linalg.norm(ha - pre, axis=-1) / np.linalg.norm(pre, axis=-1)).mean())
    return out


def design_a(ref: str, rows_ref: dict[str, dict]) -> dict:
    """What design (a) as written would score: Hunch's readout on the vector BEFORE the backbone's final norm."""
    if not (PREDS / f"{ref}.hidden.npz").exists():
        return {}
    r = np.load(PREDS / f"{ref}.hidden.npz")
    if "pre" not in r:
        return {}
    nw, hw = torch.from_numpy(r["norm_w"]), torch.from_numpy(r["head_w"])
    z = readout(torch.from_numpy(r["pre"]), nw, hw)
    tvs, agree, off = [], [], 0
    for qid, k in zip(r["ids"], r["sizes"]):
        k = int(k)
        p = question_probs(*group_logits(z[off:off + k], [k]))[0].numpy()
        p_ref = np.array(rows_ref[str(qid)]["probs"])
        tvs.append(0.5 * np.abs(p - p_ref).sum()); agree.append(p.argmax() == p_ref.argmax())
        off += k
    return {"n_questions": len(tvs), "tv_max": float(max(tvs)), "tv_mean": float(np.mean(tvs)), "argmax_agree": float(np.mean(agree))}


def report(rows: list[dict]) -> dict:
    ids = [r["id"] for r in rows]
    kmax = max(len(r["probs"]) for r in rows)
    by_id = {r["id"]: r for r in rows}
    P_arch, T = matrix(by_id, ids, kmax), matrix(by_id, ids, kmax, key="target")
    fam, qt = np.array([r["family"] for r in rows]), [r["qtype"] for r in rows]
    names = sorted(p.stem for p in PREDS.glob("*.jsonl"))
    loaded = {n: load_rows(n) for n in names}
    manifest = json.loads((OUT / "manifest.json").read_text()) if (OUT / "manifest.json").exists() else {}
    manifest = {f: m for f, m in manifest.items() if f.startswith(NAME + ".") or f.startswith(NAME + "-")}   # this size's artifacts only
    rep = {"archive": str(ARCHIVE.relative_to(REPO)), "n": len(ids), "bars": BARS, "tolerances": {"smece": CAL_TOL, "nll": NLL_TOL},
           "manifest": manifest, "noise_floor": {}, "artifacts": {}}

    def cov(name):
        idx = [i for i, q in enumerate(ids) if q in loaded[name]]
        return idx, [ids[i] for i in idx]

    def cmp(name, idx, sub_ids, p_ref):
        return compare(matrix(loaded[name], sub_ids, kmax), p_ref, T[idx], fam[idx], [qt[i] for i in idx], sub_ids)

    rep["archive_calibration"] = {}  # the archive's own smooth ECE on all 6000, per family and pooled, at both temperatures
    for Tm in TEMPS:
        qa = cr.temper(P_arch, Tm)
        for name, mask in [("pooled", np.ones(len(ids), bool))] + [(f, fam == f) for f in sorted(set(fam))]:
            rep["archive_calibration"][f"{name}@T={Tm}"] = float(cr.smooth_ece(*cr.calibration_pairs(qa[mask], T[mask], [q for q, m in zip(qt, mask) if m])))

    refs = [n for n in names if n.startswith("pytorch-fp32")]
    anchor = max(refs, key=lambda n: len(loaded[n]), default=None)
    rep["anchor"] = anchor
    for n in refs:
        idx, sub_ids = cov(n)
        rep["noise_floor"][f"{n} vs archive"] = cmp(n, idx, sub_ids, P_arch[idx])
        meta = PREDS / f"{n}.meta.json"
        rep["noise_floor"][f"{n} vs archive"]["readout_selfcheck_max_abs"] = json.loads(meta.read_text()).get("readout_selfcheck_max_abs") if meta.exists() else None
    if len(refs) > 1:
        a, b = refs[0], refs[1]
        common = [q for q in ids if q in loaded[a] and q in loaded[b]]
        idx = [i for i, q in enumerate(ids) if q in set(common)]
        rep["noise_floor"][f"{a} vs {b}"] = compare(matrix(loaded[a], common, kmax), matrix(loaded[b], common, kmax), T[idx], fam[idx], [qt[i] for i in idx], common)
    if anchor:
        rep["design_a"] = design_a(anchor, loaded[anchor])

    for n in names:
        level = level_of(n)
        if level is None:
            continue
        idx, sub_ids = cov(n)
        vs_arch = cmp(n, idx, sub_ids, P_arch[idx])
        meta = PREDS / f"{n}.meta.json"
        entry = {"level": level, "n_scored": len(idx), "vs_archive": vs_arch, "verdict_vs_archive": verdict(level, vs_arch),
                 "regroup_tv_max": regroup_tv_max(loaded[n]), "run": json.loads(meta.read_text()) if meta.exists() else {}}
        if anchor:
            common_set = set(loaded[anchor]) & set(sub_ids)
            cidx = [i for i in idx if ids[i] in common_set]
            cids = [ids[i] for i in cidx]
            if cids:
                vs_fp32 = compare(matrix(loaded[n], cids, kmax), matrix(loaded[anchor], cids, kmax), T[cidx], fam[cidx], [qt[i] for i in cidx], cids)
                entry.update(vs_fp32=vs_fp32, verdict_vs_fp32=verdict(level, vs_fp32))
                if f"{anchor} vs archive" in rep["noise_floor"]:
                    entry["gate_vs_floor"] = floor_gate(vs_fp32, rep["noise_floor"][f"{anchor} vs archive"])
            entry["localisation"] = localise(n, anchor)
        parity = PREDS / f"{n}.tokenizer.json"
        if parity.exists():
            entry["tokenizer_parity"] = json.loads(parity.read_text())
        rep["artifacts"][n] = entry

    rep["backend_agreement"] = {}  # the same file scored on two backends (e.g. gguf-Q8_0 on Metal vs gguf-Q8_0+cuda)
    by_file: dict[str, list[str]] = {}
    for n in rep["artifacts"]:
        by_file.setdefault(n.split("+")[0], []).append(n)
    for ns in by_file.values():
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                a, b = ns[i], ns[j]
                cs = set(loaded[a]) & set(loaded[b])
                cidx = [k for k, q in enumerate(ids) if q in cs]
                common = [ids[k] for k in cidx]
                if common:
                    rep["backend_agreement"][f"{a} vs {b}"] = compare(matrix(loaded[a], common, kmax), matrix(loaded[b], common, kmax), T[cidx], fam[cidx], [qt[k] for k in cidx], common)

    md = render(rep)
    OUT.mkdir(parents=True, exist_ok=True)
    # One report per size (equivalence-0.6b, equivalence-1.7b): a report must never overwrite another size's, and reading
    # the wrong one silently presents another model's fidelity as this model's.
    suffix = "-" + NAME.split(chr(45))[1]
    (OUT / f"equivalence{suffix}.json").write_text(json.dumps(rep, indent=1))
    (OUT / f"equivalence{suffix}.md").write_text(md)
    print(md)
    return rep


def render(rep: dict) -> str:
    ok = lambda b: "pass" if b else "**FAIL**"
    arts = rep["artifacts"]
    L = [f"# Equivalence report: {NAME}", "",
         f"Reference archive `{rep['archive']}`: {rep['n']} held-out questions, bf16 autocast, T = 1. fp32 anchor: `{rep.get('anchor')}`.", "",
         "**The pass rule** (FORMATS.md): an artifact passes if its max TV against the fp32 anchor is within the floor (bf16 against "
         "fp32 on the same checkpoint) and its disagreement rate is not significantly above the floor's (two-proportion z, one-sided "
         "95 %). \"pass\" = significantly below the floor rate, \"at floor\" = indistinguishable from it, \"no\" = above it.", ""]
    floor = next((c for k, c in rep["noise_floor"].items() if k == f"{rep.get('anchor')} vs archive"), None)
    if floor and any("gate_vs_floor" in e for e in arts.values()):
        L += [f"## The floor gate against `{rep.get('anchor')}`", "",
              f"Floor: max TV {floor['tv_max']:.2e}, {floor['n_argmax_diff']} of {floor['n'] - floor['n_nonfinite']} answers differ.", "",
              "| artifact | n | max TV | × floor | answers differing | × floor | z | verdict |", "|---|---|---|---|---|---|---|---|"]
        for n, e in arts.items():
            g, c = e.get("gate_vs_floor"), e.get("vs_fp32")
            if g and c:
                L.append(f"| {n} | {c['n']} | {c['tv_max']:.2e} | {g['x_floor_tv']:.2f} | {c['n_argmax_diff']} ({100 * g['rate']:.3f} %) | "
                         f"{g['x_floor_rate']:.2f} | {g['z']:+.2f} | **{g['verdict']}** |")
        L.append("")
    L += ["The tables below also show the original fixed bars (max TV f16 ≤ 1e-3, Q8/8-bit ≤ 5e-3, Q4 ≤ 2e-2; argmax 100 % for "
          f"f16/Q8, ≥ 99.5 % for Q4; smooth ECE at T = 1 and T = {TEMPERATURE} within 0.005; pooled NLL within 0.005). They are "
          "tighter than bf16 itself and are not the pass rule.", ""]

    def stat_cols(c, v=None, with_verdict=False):
        cols = [str(c["n"]), f"{c['tv_max']:.2e}", f"{c['tv_mean']:.2e}", f"{100 * c['argmax_agree']:.2f} % ({c['n_argmax_diff']})",
                f"{c['max_abs_dsmece']:.4f}", f"{c['nll@T=1.0']['delta']:+.4f}", str(c["n_nonfinite"])]
        if with_verdict:
            cols += ([ok(v["tv"]), ok(v["argmax"]), ok(v["calibration"]), ok(v["nll"]), "yes" if v["all_fixed_bars"] else "no"] if v
                     else ["no bar"] * 4 + ["n/a"])
        return cols

    head = "| artifact | n | max TV | mean TV | argmax agree (differ) | max abs ΔsmECE | pooled ΔNLL (T=1) | non-finite |"
    head_v = head + " fixed TV bar | fixed argmax bar | smECE ≤ 0.005 | NLL ≤ 0.005 | all fixed bars |"
    sep = lambda h: "|" + "|".join("---" for _ in h.split("|")[1:-1]) + "|"

    L += ["## Noise floor: the reference against itself", "", head, sep(head)]
    for k, c in rep["noise_floor"].items():
        L.append("| " + " | ".join([k] + stat_cols(c)) + " |")
    for k, c in rep["noise_floor"].items():
        if c.get("readout_selfcheck_max_abs") is not None:
            L.append(f"\n`{k.split(' vs ')[0]}`: max |readout(hidden) − DecisionScorer.forward| = {c['readout_selfcheck_max_abs']:.2e} (this package's fp32 readout against the reference's own).")
    if rep.get("design_a"):
        d = rep["design_a"]
        L.append(f"\nThe rejected GGUF design (readout norm folded into `output_norm`, dropping `backbone.norm`), scored from the reference's pre-norm hidden state on {d['n_questions']} questions: "
                 f"max TV {d['tv_max']:.3f}, mean TV {d['tv_mean']:.3f}, argmax agreement {100 * d['argmax_agree']:.1f} %.")

    if rep.get("backend_agreement"):
        L += ["", "## Same file, different backend (common questions)", "", head, sep(head)]
        for k, c in rep["backend_agreement"].items():
            L.append("| " + " | ".join([k] + stat_cols(c)) + " |")

    L += ["", "## Artifacts against the archived bf16 reference (all scored questions)", "", head_v, sep(head_v)]
    for n, e in arts.items():
        L.append("| " + " | ".join([n] + stat_cols(e["vs_archive"], e["verdict_vs_archive"], True)) + " |")
    if any("vs_fp32" in e for e in arts.values()):
        L += ["", f"## Artifacts against the PyTorch fp32 reference (`{rep['anchor']}`, common questions)", "", head_v, sep(head_v)]
        for n, e in arts.items():
            if "vs_fp32" in e:
                L.append("| " + " | ".join([n] + stat_cols(e["vs_fp32"], e["verdict_vs_fp32"], True)) + " |")

    fams = sorted({k.split("@")[0] for e in arts.values() for k in e["vs_archive"]["calibration"]} | {k.split("@")[0] for c in rep["noise_floor"].values() for k in c["calibration"]})
    if fams:
        fams = [f for f in fams if f != "pooled"] + ["pooled"]
        L += ["", f"## Calibration: smooth ECE per family, T = 1 / T = {TEMPERATURE}", "",
              "Each cell: the run's smooth ECE on the questions it scored, with (Δ) against the archive on the same questions. "
              "smECE is sample-size dependent, so only the deltas are comparable across rows with different n.", ""]
        h = "| run (n) | " + " | ".join(f.replace("heldout.", "") for f in fams) + " |"
        L += [h, sep(h)]
        if rep.get("archive_calibration"):
            a = rep["archive_calibration"]
            L.append(f"| archive, bf16 ({rep['n']}) | " + " | ".join(f"{a[f'{f}@T=1.0']:.4f} / {a[f'{f}@T={TEMPERATURE}']:.4f}" for f in fams) + " |")
        cell = lambda c, f: " / ".join(f"{c[f'{f}@T={T}']['smece']:.4f} ({c[f'{f}@T={T}']['delta']:+.4f})" for T in TEMPS)
        for k, c in rep["noise_floor"].items():
            if " vs archive" in k:
                L.append(f"| {k.split(' vs ')[0]} ({c['n']}) | " + " | ".join(cell(c["calibration"], f) for f in fams) + " |")
        for n, e in arts.items():
            c = e["vs_archive"]
            L.append(f"| {n} ({c['n']}) | " + " | ".join(cell(c["calibration"], f) for f in fams) + " |")

    if any("localisation" in e for e in arts.values()):
        L += ["", f"## Localisation on the {SUBSET_N}-question subset, against `{rep['anchor']}`", "",
              "| artifact | paths | max rel ‖Δh‖ (post-norm) | mean rel ‖Δh‖ | mean rel ‖Δh‖ vs PRE-norm | max abs Δz | mean abs Δz | max TV vs fp32 | regroup TV | tokenizer segments mismatched |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for n, e in arts.items():
            l = e.get("localisation") or {}
            if not l:
                continue
            tp = e.get("tokenizer_parity")
            L.append(f"| {n} | {l['n_paths']} | {l['hidden_rel_l2_max']:.2e} | {l['hidden_rel_l2_mean']:.2e} | {l.get('hidden_rel_l2_mean_vs_prenorm', float('nan')):.2f} | "
                     f"{l['logit_abs_max']:.2e} | {l['logit_abs_mean']:.2e} | {e['vs_fp32']['tv_max']:.2e} | {e['regroup_tv_max']:.1e} | "
                     + (f"{tp['mismatched']} / {tp['segments']}" if tp else "n/a") + " |")

    L += ["", "## Worst questions per artifact (against the archive)", ""]
    for n, e in arts.items():
        for w in e["vs_archive"]["worst"][:2]:
            L.append(f"- `{n}` `{w['id']}` TV {w['tv']:.4f}: artifact {w['p_hat']} vs reference {w['p_ref']}")

    if rep.get("manifest"):
        L += ["", "## Artifacts (sha256 recorded at conversion)", "", "| file | format | level | bytes | sha256 |", "|---|---|---|---|---|"]
        for f, m in sorted(rep["manifest"].items()):
            q = m.get("quantization")
            level = m["level"] + (f" (affine, group {q['group_size']}, {q['bits']}-bit)" if q else "") + (f", {m['dtype']}" if m.get("dtype") and not q else "")
            L.append(f"| `{f}` | {m['format']} | {level} | {m['bytes']:,} | `{m['sha256']}` |")
    runs = {n: e.get("run", {}) for n, e in arts.items()} | {k.split(" vs ")[0]: {} for k in rep["noise_floor"]}
    L += ["", "## Run settings", "", "| run | settings | paths | tokens | seconds |", "|---|---|---|---|---|"]
    for n in sorted(runs):
        meta = PREDS / f"{n}.meta.json"
        r = json.loads(meta.read_text()) if meta.exists() else {}
        L.append(f"| {n} | `{json.dumps(r.get('settings', {}), sort_keys=True)}` | {r.get('paths', '')} | {r.get('tokens', '')} | {r.get('seconds', '')} |")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- cli

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="score every artifact in hunch/formats/out (and the fp32 reference subset if --checkpoint is given), then report")
    ap.add_argument("--reference", action="store_true", help="score the PyTorch fp32 reference (needs --checkpoint)")
    ap.add_argument("--checkpoint", help="directory holding the release model.safetensors")
    ap.add_argument("--device", default="cpu", help="device for --reference: cpu (the fp32 reference) or mps")
    ap.add_argument("--limit", type=int, default=None, help="score a family-balanced subset of this many questions; 0 = all 6000 (default: 201 for --reference, all otherwise)")
    ap.add_argument("--gguf", action="append", default=[], help="GGUF artifact to score (repeatable)")
    ap.add_argument("--mlx", action="append", default=[], help="MLX artifact directory to score (repeatable)")
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--report", action="store_true", help="only recompute the report from out/preds")
    ap.add_argument("--no-report", action="store_true", help="score without reporting (another scoring process may still be appending)")
    ap.add_argument("--tag", default="", help="backend tag appended to artifact run names, e.g. cuda -> gguf-Q8_0+cuda; untagged = the laptop backend")
    ap.add_argument("--mlx-activations", default="float32", choices=["float32", "float16"], help="MLX activation dtype; float16 runs are named mlx-<level>+act16")
    ap.add_argument("--gguf-gpu-layers", type=int, default=-1, help="llama.cpp layers to offload; 0 = CPU backend (f32 accumulation), -1 = all")
    ap.add_argument("--gguf-threads", type=int, default=None, help="llama.cpp CPU threads (default: all cores)")
    ap.add_argument("--shard", default="", help="K/N: score only every N-th question of the ordered list starting at K (0-based), for fanning one artifact out across machines; concatenate the shards' preds afterwards")
    args = ap.parse_args()

    rows = archive()
    order = [r["id"] for r in rows]
    keep = set(subset(rows))
    jobs = []
    ref_name = f"pytorch-fp32-{args.device}"
    if args.reference or (args.all and args.checkpoint and not (PREDS / f"{ref_name}.jsonl").exists()):
        if not args.checkpoint:
            raise SystemExit("--reference needs --checkpoint <directory with model.safetensors>")
        n = SUBSET_N if args.limit is None else args.limit
        jobs.append((ref_name, lambda: TorchBackbone(args.checkpoint, args.device), subset(rows, n) if n else order))
    ids = subset(rows, args.limit) if args.limit else order
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not 0 <= k < n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= K < N")
        ids = ids[k::n]
    ggufs = [Path(p) for p in args.gguf] + (sorted(OUT.glob("*.gguf")) if args.all else [])
    mlxs = [Path(p) for p in args.mlx] + (sorted(d for d in OUT.glob(f"{NAME}-mlx-*") if (d / "config.json").exists()) if args.all else [])
    for g in ggufs:
        jobs.append((artifact_name(g, args.tag), lambda g=g: __import__("hunch.formats.hunch_gguf", fromlist=["GGUFBackbone"]).GGUFBackbone(
            g, n_ctx=args.token_budget, n_gpu_layers=args.gguf_gpu_layers, n_threads=args.gguf_threads), ids))
    mlx_tag = "+".join(t for t in (args.tag, "act16" if args.mlx_activations == "float16" else "") if t)
    for m in mlxs:
        jobs.append((artifact_name(m, mlx_tag), lambda m=m: __import__("hunch.formats.hunch_mlx", fromlist=["MLXBackbone"]).MLXBackbone(m, args.mlx_activations), ids))

    if jobs:
        tok = AutoTokenizer.from_pretrained(BACKBONE)
        cache: dict[tuple, list] = {}
        for name, make, job_ids in jobs:
            key = tuple(job_ids)
            if key not in cache:
                cache[key] = questions_for(job_ids, tok)
            backbone = make()
            score(name, backbone, cache[key], args.token_budget, keep, tok.pad_token_id)
            if hasattr(backbone, "tokenize"):
                parity = tokenizer_parity(backbone, records_for(sorted(keep)), tok)
                (PREDS / f"{name}.tokenizer.json").write_text(json.dumps(parity, indent=1))
                print(f"{name}: tokenizer parity {parity['segments'] - parity['mismatched']}/{parity['segments']} segments identical", file=sys.stderr)
            del backbone
    if (args.report or args.all or jobs) and not args.no_report:
        report(rows)


if __name__ == "__main__":
    main()
