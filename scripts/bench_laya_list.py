#!/usr/bin/env python3
"""Every public suite Laya reports on, rebuilt row for row from Laya's published benchmark code, scored with Hunch.

    python scripts/bench_laya_list.py --dry-run --out-dir results/laya        # build + tokenise + price, no model
    python scripts/bench_laya_list.py --checkpoint <dir> --model <hf id> --release-T 1.0 --out-dir results/laya

Construction follows `research/scripts/build_benchmark_nb.py`, `research/eval/laya_eval.py` and
`research/scripts/bench_apps.py` of github.com/NandhaKishorM/laya (Apache-2.0; copies under .cache/laya/,
fetched 2026-09-23): the same datasets, splits, row slices, seeds, option construction, label rendering and
instruction strings. Every dataset revision is recorded at build time. Where our schema differs from Laya's
(`criteria` maps a key to a description or None), the mapping is: description None -> candidate label = key;
value equal to the rendered key -> label = value; otherwise label = key, description = value. `noul` is our
Boolean question; a `score` question carries Laya's level texts as levels.

Metrics are Laya's: accuracy, macro-F1, ECE over 15 equal-width bins (first bin closed at the bottom), Brier
(sum over classes of (p - onehot)^2), NLL (-log p_gold); score suites add the MAE of the expected level.
Reported at T = 1 and at the release temperature (--release-T), both stated. Option-order robustness follows
their section 7: the first 200 cases of massive_intent.en, en.emotion and xnli.en with options permuted by
random.Random(99); the flip rate is how often the chosen option changes.

Overlap with Hunch's training mixture (marked on the row, never hidden): Banking77, MASSIVE intent English
(and scenario: same utterances), BoolQ. Laya's own overlaps (AG News, BoolQ, spam, phishing, triage, RAG)
are carried over from their notes. The sealed test split of this project is never touched.
"""
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SEED, N_OPTS, PER_LANG, N_THEME = 13, 20, 300, 400
MASSIVE_LANGS = ["en", "de", "fr", "es", "pt", "ru", "tr", "ar", "hi", "ta", "zh-CN", "ja", "ko", "sw"]
XNLI_LANGS = ["en", "de", "fr", "es", "ru", "tr", "ar", "hi", "ur", "vi", "th", "el", "bg", "zh", "sw"]
NLI_CRIT = {"entailment": "the premise implies the hypothesis is true",
            "neutral": "the premise neither implies nor contradicts the hypothesis",
            "contradiction": "the premise implies the hypothesis is false"}
HUNCH_TRAINED = {"en.banking77_full", "jev.banking77_full", "massive_intent.en", "massive_scenario.en", "en.boolq"}
LAYA_TRAINED = {"en.ag_news", "jev.ag_news", "en.boolq", "app.support_triage", "app.email_spam", "app.phishing", "app.rag_relevance"}
REV, FAIL = {}, {}


def render(key):  # laya_eval.render_label / build_choice_questions
    return key.replace("_", " ").replace(".", ": ")


def load(ds, *cfg, split):
    from datasets import load_dataset
    if ds not in REV:
        try:
            from huggingface_hub import HfApi
            REV[ds] = HfApi().dataset_info(ds).sha
        except Exception as e:
            REV[ds] = f"unpinned:{type(e).__name__}"
    kw = {} if REV[ds].startswith("unpinned") else {"revision": REV[ds]}
    return load_dataset(ds, *cfg, split=split, **kw)


def case(state, qid, qtype, instruction, options, gold, gold_score=None, criteria=None):
    return {"state": state, "qid": qid, "qtype": qtype, "instruction": instruction, "options": options,
            "gold": gold, "gold_score": gold_score, "criteria": criteria}


def opts_from_crit(crit):
    """Laya's criteria dict -> [(key, label, description)] in insertion order."""
    out = []
    for k, v in crit.items():
        if v is None: out.append((k, k, ""))
        elif v == render(k): out.append((k, v, ""))
        else: out.append((k, k, str(v)))
    return out


# ------------------------------------------------------------------ suites, in Laya's construction order
def build_multilingual(S):
    for task, short, instr in [("mteb/amazon_massive_intent", "massive_intent", "What is the user asking for in `utterance`?"),
                               ("mteb/amazon_massive_scenario", "massive_scenario", "Which domain does `utterance` belong to?")]:
        for lg in MASSIVE_LANGS:
            name = f"{short}.{lg}"
            try:
                d = load(task, lg, split="test"); labels = sorted(set(d["label_text"])); rng = random.Random(SEED); cases = []
                for r in list(d)[:PER_LANG]:
                    pool = [x for x in labels if x != r["label_text"]]
                    keys = [r["label_text"]] + rng.sample(pool, min(N_OPTS - 1, len(pool))); rng.shuffle(keys)
                    cases.append(case({"utterance": r["text"]}, "label", "choice", instr, [(k, render(k), "") for k in keys], keys.index(r["label_text"])))
                S[name] = {"cases": cases, "meta": {"lang": lg, "n_options": N_OPTS, "label_space": len(labels), "family": short}}
            except Exception as e:
                FAIL[name] = f"{type(e).__name__}: {str(e)[:120]}"
    for lg in XNLI_LANGS:
        name = f"xnli.{lg}"
        try:
            d = load("facebook/xnli", lg, split="test"); cases = []
            for r in list(d)[:PER_LANG]:
                cases.append(case({"premise": r["premise"], "hypothesis": r["hypothesis"]}, "relation", "choice",
                                  "What is the relationship between `premise` and `hypothesis`?", opts_from_crit(NLI_CRIT), int(r["label"])))
            S[name] = {"cases": cases, "meta": {"lang": lg, "n_options": 3, "family": "xnli"}}
        except Exception as e:
            FAIL[name] = f"{type(e).__name__}: {str(e)[:120]}"


def build_english(S):
    def add(name, fn, **meta):
        try:
            S[name] = {"cases": fn(), "meta": dict(meta, family="english")}
        except Exception as e:
            FAIL[name] = f"{type(e).__name__}: {str(e)[:120]}"

    def sst5():
        d = load("SetFit/sst5", split="test"); crit = ["very negative", "negative", "neutral", "positive", "very positive"]
        return [case({"text": r["text"]}, "sentiment", "score", "How positive is the sentiment of `text`?",
                     [(c, c, "") for c in crit], int(r["label"]), float(r["label"])) for r in list(d)[:600]]

    def emotion():
        d = load("dair-ai/emotion", "split", split="test"); names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        return [case({"text": r["text"]}, "emotion", "choice", "Which emotion is most strongly expressed in `text`?",
                     [(n, n, "") for n in names], int(r["label"])) for r in list(d)[:600]]

    def injections():
        d = load("deepset/prompt-injections", split="test")
        return [case({"text": r["text"]}, "injection", "boolean", "Does `text` try to inject or override instructions given to an AI system?",
                     [("false", "no", ""), ("true", "yes", "")], int(r["label"])) for r in list(d)]

    def banking77():
        try:
            d = load("PolyAI/banking77", split="test"); names = d.features["label"].names; rows = [(r["text"], int(r["label"])) for r in list(d)[:500]]
        except Exception:  # the script-based dataset may not load on a current `datasets`; mteb/banking77 has the same rows and label ids
            d = load("mteb/banking77", split="test"); names = [None] * 77
            for r in d: names[int(r["label"])] = r["label_text"]
            rows = [(r["text"], int(r["label"])) for r in list(d)[:500]]
        opts = [(n, n.replace("_", " "), "") for n in names]
        return [case({"message": t}, "intent", "choice", "Which banking intent does `message` express?", opts, g) for t, g in rows]

    def ag_news():
        d = load("fancyzhx/ag_news", split="test")
        crit = {"world": "world news and international politics", "sports": "sports", "business": "business and economy", "sci_tech": "science and technology"}
        return [case({"article": r["text"]}, "topic", "choice", "What is the topic of `article`?", opts_from_crit(crit), int(r["label"])) for r in list(d)[:600]]

    def boolq():
        d = load("google/boolq", split="validation")
        return [case({"passage": r["passage"], "question": r["question"]}, "answer", "boolean", "Based on `passage`, is the answer to `question` yes?",
                     [("false", "no", ""), ("true", "yes", "")], int(bool(r["answer"]))) for r in list(d)[:600]]

    add("en.sst5", sst5, note="zero-shot, ordinal", held_out_for_laya=True)
    add("en.emotion", emotion, note="zero-shot", held_out_for_laya=True)
    add("en.prompt_injections", injections, note="zero-shot", held_out_for_laya=True)
    add("en.banking77_full", banking77, note="77 options at once", held_out_for_laya=True)
    add("en.ag_news", ag_news, note="in Laya's training", held_out_for_laya=False)
    add("en.boolq", boolq, note="in Laya's training", held_out_for_laya=False)


def build_themes(S):
    """bench_apps.py, N = 400; one random.Random(13) shared across the builds in this exact order."""
    rng = random.Random(SEED)

    def add(name, fn, **meta):
        try:
            S[name] = {"cases": fn(), "meta": dict(meta, family="app")}
        except Exception as e:
            FAIL[name] = f"{type(e).__name__}: {str(e)[:120]}"

    def ag():
        d = load("fancyzhx/ag_news", split="test")
        crit = {"world": "world news and international politics", "sports": "sports", "business": "business and economy", "sci_tech": "science and technology"}
        return [case({"article": r["text"]}, "topic", "choice", "What is the topic of `article`?", opts_from_crit(crit), int(r["label"])) for r in list(d)[:N_THEME]]

    def emo():
        d = load("dair-ai/emotion", "split", split="test"); names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        return [case({"text": r["text"]}, "emotion", "choice", "Which emotion is most strongly expressed in `text`?", [(n, n, "") for n in names], int(r["label"])) for r in list(d)[:N_THEME]]

    def b77():
        d = load("mteb/banking77", split="test"); labels = sorted(set(d["label_text"])); keys = [x.replace("_", " ") for x in labels]
        return [case({"message": r["text"]}, "intent", "choice", "Which banking intent does `message` express?", [(k, k, "") for k in keys],
                     keys.index(r["label_text"].replace("_", " "))) for r in list(d)[:N_THEME]]

    def triage():
        d = load("Tobi-Bueck/customer-support-tickets", split="train")
        Q = {"Technical Support": "technical problems, bugs, outages, integrations", "Product Support": "help using a product or feature",
             "Customer Service": "general account or service questions", "IT Support": "internal IT, devices, access, networks",
             "Billing and Payments": "invoices, charges, refunds, payment methods", "Returns and Exchanges": "returning or exchanging an item",
             "Service Outages and Maintenance": "downtime, outages, scheduled maintenance", "Sales and Pre-Sales": "pricing, quotes, buying",
             "Human Resources": "employment, payroll, leave, hiring", "General Inquiry": "anything else"}
        keys = list(Q); out = []
        for r in d:
            if r.get("language") != "en" or r.get("queue") not in Q or not r.get("body"): continue
            out.append(case({"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]}, "queue", "choice",
                            "Which support queue should handle this ticket?", opts_from_crit(Q), keys.index(r["queue"])))
            if len(out) >= N_THEME: break
        return out

    def spam():
        from _laya_email import email_state
        d = load("SetFit/enron_spam", split="test")
        return [case(email_state(r.get("subject") or "", (r.get("message") or "")[:3000]), "is_spam", "boolean",
                     "Is this email unsolicited spam or bulk marketing?", [("false", "no", ""), ("true", "yes", "")], int(r["label"])) for r in list(d)[:N_THEME]]

    def phishing():
        d = load("zefang-liu/phishing-email-dataset", split="train")
        rows = [r for r in list(d)[:6000] if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")]
        rng.shuffle(rows)
        return [case({"email": r["Email Text"][:3000]}, "is_phishing", "boolean",
                     "Is this email a phishing or scam attempt to steal money, credentials, or personal data?", [("false", "no", ""), ("true", "yes", "")],
                     int(r["Email Type"] == "Phishing Email"), criteria={"true": "phishing, scam, or fraud", "false": "a legitimate email (even if promotional)"})
                for r in rows[:N_THEME]]

    tc = {}
    def toxic_rows():
        if "rows" not in tc:
            d = load("lmsys/toxic-chat", "toxicchat0124", split="test"); tc["rows"] = [r for r in d if (r.get("user_input") or "").strip()]
        return tc["rows"]

    def jailbreak():
        rows = toxic_rows(); jb = [r for r in rows if int(r.get("jailbreaking", 0)) == 1][:N_THEME // 2]; nj = [r for r in rows if int(r.get("jailbreaking", 0)) == 0][:N_THEME - len(jb)]
        mix = jb + nj; rng.shuffle(mix)
        return [case({"prompt": r["user_input"][:3000]}, "jailbreak", "boolean", "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?",
                     [("false", "no", ""), ("true", "yes", "")], int(r["jailbreaking"])) for r in mix]

    def toxicity():
        rows = toxic_rows(); tox = [r for r in rows if int(r.get("toxicity", 0)) == 1][:N_THEME // 2]; ntox = [r for r in rows if int(r.get("toxicity", 0)) == 0][:N_THEME - len(tox)]
        mix = tox + ntox; rng.shuffle(mix)
        return [case({"post": r["user_input"][:3000]}, "toxic", "boolean", "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?",
                     [("false", "no", ""), ("true", "yes", "")], int(r["toxicity"])) for r in mix]

    def rag():
        d = load("microsoft/ms_marco", "v1.1", split="validation"); out = []
        for r in d:
            texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
            pos = [t for t, s in zip(texts, sel) if s == 1]; neg = [t for t, s in zip(texts, sel) if s == 0]
            if not pos or not neg: continue
            take_pos = len(out) % 2 == 0; p = rng.choice(pos if take_pos else neg)
            out.append(case({"query": r["query"], "passage": p}, "relevant", "boolean", "Does `passage` help answer `query`?", [("false", "no", ""), ("true", "yes", "")], 1 if take_pos else 0))
            if len(out) >= N_THEME: break
        return out

    def routing():
        DOM = {"code": "software engineering, programming, refactoring, architecture, debugging", "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
               "writing": "creative writing, essays, emails, blog posts, copywriting", "factual_lookup": "facts, definitions, trivia, history",
               "data_analysis": "statistics, SQL, data manipulation, metrics", "chitchat": "casual conversation, greetings, small talk"}
        keys = list(DOM); pool = []
        g = load("openai/gsm8k", "main", split="test"); pool += [(r["question"], "math_or_logic") for r in list(g)[:N_THEME // 3]]
        m = load("google-research-datasets/mbpp", "full", split="test"); pool += [(r["text"], "code") for r in list(m)[:N_THEME // 3]]
        t = load("fancyzhx/ag_news", split="test"); pool += [(r["text"][:400], "factual_lookup") for r in list(t)[:N_THEME // 3]]
        rng.shuffle(pool)
        return [case({"request": text}, "domain", "choice", "What domain does `request` belong to?", opts_from_crit(DOM), keys.index(dom)) for text, dom in pool[:N_THEME]]

    add("jev.ag_news", ag, note="Jev published 0.910 (n=100, indicative)")
    add("jev.emotion", emo, note="Jev published 0.480 (n=100, indicative)")
    add("jev.banking77_full", b77, note="Jev published 0.870 on 72 labels (indicative)")
    add("app.support_triage", triage, note="10-way queue routing")
    add("app.email_spam", spam, note="enron spam")
    add("app.phishing", phishing, note="phishing emails")
    add("app.guardrails_jailbreak", jailbreak, note="toxic-chat jailbreaking")
    add("app.moderation_toxicity", toxicity, note="toxic-chat toxicity")
    add("app.rag_relevance", rag, note="MS MARCO relevance")
    add("app.model_routing_domain", routing, note="gsm8k/mbpp/ag_news -> domain")


def build_all():
    S = {}; build_multilingual(S); build_english(S); build_themes(S); return S


# ------------------------------------------------------------------ requests, scoring, metrics
def to_request(c, rid):
    q = {"id": c["qid"], "type": c["qtype"], "instruction": c["instruction"]}
    if c["qtype"] == "choice":
        q["candidates"] = [{"id": k, "label": lab, "description": desc} for k, lab, desc in c["options"]]
    elif c["qtype"] == "score":
        q["levels"] = [lab for _, lab, _ in c["options"]]
    else:
        q["candidates"] = []
        if c.get("criteria"): q["criteria"] = c["criteria"]
    return {"id": rid, "state": c["state"], "questions": [q]}


def score_cases(h, cases, batch):
    P = [None] * len(cases); reqs = [(i, to_request(c, f"c{i}")) for i, c in enumerate(cases)]
    for s in range(0, len(reqs), batch):
        chunk = reqs[s:s + batch]
        try:
            res = h.decide([r for _, r in chunk])["results"]
        except Exception:  # fall back to one at a time so one bad row does not take the batch
            res = []
            for _, r in chunk:
                try: res.append(h.decide([r])["results"][0])
                except Exception as e: res.append({"error": str(e)[:120]})
        for (i, r), rr in zip(chunk, res):
            if "answers" not in rr: continue
            ans = rr["answers"][r["questions"][0]["id"]]; c = cases[i]
            if c["qtype"] == "boolean": p = [1 - ans["probability"], ans["probability"]]
            elif c["qtype"] == "choice": p = [ans["probabilities"][k] for k, _, _ in c["options"]]
            else: p = list(ans["probabilities"])
            P[i] = np.array(p, float)
    return P


def temper(p, T):
    q = np.power(np.clip(p, 1e-12, 1), 1 / T); return q / q.sum()


def ece_score(conf, corr, bins=15):  # bench_local.ece_score, verbatim semantics
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    if not len(conf): return float("nan")
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if s.any(): e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def macro_f1(g, p):  # bench_local.macro_f1
    g, p = np.asarray(g), np.asarray(p); f = []
    for c in sorted(set(g.tolist()) | set(p.tolist())):
        tp = int(((p == c) & (g == c)).sum()); fp = int(((p == c) & (g != c)).sum()); fn = int(((p != c) & (g == c)).sum())
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def metrics(cases, P, T):
    g, pred, conf, brier, nll, mae = [], [], [], [], [], []
    for c, p in zip(cases, P):
        if p is None: continue
        q = temper(p, T) if T != 1.0 else p; a = int(q.argmax()); oh = np.zeros(len(q)); oh[c["gold"]] = 1
        g.append(c["gold"]); pred.append(a); conf.append(float(q.max())); brier.append(float(((q - oh) ** 2).sum())); nll.append(float(-np.log(max(q[c["gold"]], 1e-12))))
        if c["gold_score"] is not None: mae.append(abs(float((np.arange(len(q)) * q).sum()) - c["gold_score"]))
    if not g: return {"n": 0}
    g, pred = np.array(g), np.array(pred)
    out = {"n": int(len(g)), "accuracy": float((g == pred).mean()), "macro_f1": macro_f1(g, pred), "ece": ece_score(conf, (g == pred).astype(float)),
           "brier": float(np.mean(brier)), "nll": float(np.mean(nll)), "mean_confidence": float(np.mean(conf))}
    if mae: out["score_mae"] = float(np.mean(mae))
    return out


def dry_run(S, out_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); rows = []; tot_paths = tot_tok = 0
    for name, s in S.items():
        cs = s["cases"]; paths = toks = over = 0
        for c in cs:
            st = json.dumps(c["state"], ensure_ascii=False) if isinstance(c["state"], dict) else c["state"]
            n_st = len(tok(st, add_special_tokens=False)["input_ids"]); k = max(2, len(c["options"]))
            if n_st > 32768: over += 1
            paths += k; toks += k * (n_st + 8) + sum(len(tok(f"{lab} {desc}".strip(), add_special_tokens=False)["input_ids"]) for _, lab, desc in c["options"])
        tot_paths += paths; tot_tok += toks
        rows.append({"suite": name, "cases": len(cs), "k": max(2, len(cs[0]["options"])) if cs else 0, "paths": paths, "tokens": toks, "states_over_cap": over,
                     "hunch_trained": name in HUNCH_TRAINED, "laya_trained": name in LAYA_TRAINED, **s["meta"]})
        print(f"  {name:28s} cases {len(cs):4d}  k {rows[-1]['k']:3d}  paths {paths:7d}  tokens {toks:9d}  over-cap {over}", flush=True)
    man = {"seed": SEED, "n_opts": N_OPTS, "per_lang": PER_LANG, "n_theme": N_THEME, "revisions": REV, "failed": FAIL, "suites": rows,
           "total_paths": tot_paths, "total_tokens": tot_tok, "est_minutes_at_1500_tok_per_s": round(tot_tok / 1500 / 60, 1)}
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "laya_list_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"\n{len(S)} suites built, {len(FAIL)} failed: {FAIL}\ntotal paths {tot_paths:,}, tokens {tot_tok:,} (~{man['est_minutes_at_1500_tok_per_s']} min at 1,500 tok/s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint"); ap.add_argument("--model"); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--release-T", type=float, default=1.0); ap.add_argument("--amp", default="bf16"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--token-budget", type=int, default=16384); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--suites", default="", help="comma-separated subset (default: all built)"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); out_dir = Path(a.out_dir)
    t0 = time.time(); S = build_all(); print(f"built {len(S)} suites in {time.time()-t0:.0f}s; failed: {list(FAIL)}", flush=True)
    if a.suites: S = {k: v for k, v in S.items() if k in set(a.suites.split(","))}
    if a.dry_run: dry_run(S, out_dir); return
    if not (a.checkpoint and a.model): sys.exit("--checkpoint and --model are required to score")
    from hunch.infer import Hunch
    h = Hunch.load(a.checkpoint, model=a.model, amp=a.amp, device=a.device); h.T = 1.0; h.token_budget = a.token_budget
    tag = Path(a.checkpoint.rstrip("/")).name
    if tag in ("best", "last"): tag = Path(a.checkpoint.rstrip("/")).parent.name   # runs/<run>/best -> <run>; every run's dir is "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"checkpoint": a.checkpoint, "model": a.model, "release_T": a.release_T, "seed": SEED, "n_opts": N_OPTS, "per_lang": PER_LANG, "n_theme": N_THEME,
           "revisions": REV, "failed_to_build": FAIL, "suites": {}, "option_order_robustness": {}}
    for name, s in S.items():
        t1 = time.time(); P = score_cases(h, s["cases"], a.batch); scored = sum(p is not None for p in P)
        res["suites"][name] = {"meta": dict(s["meta"], hunch_trained=name in HUNCH_TRAINED, laya_trained=name in LAYA_TRAINED), "n_cases": len(s["cases"]),
                               "n_scored": scored, "raw": metrics(s["cases"], P, 1.0), "at_release_T": metrics(s["cases"], P, a.release_T), "seconds": round(time.time() - t1, 1),
                               "rows": [{"gold": c["gold"], "argmax": (int(p.argmax()) if p is not None else None), "p_gold": (float(p[c["gold"]]) if p is not None else None)} for c, p in zip(s["cases"], P)]}
        r = res["suites"][name]["raw"]; print(f"  {name:28s} n {r.get('n',0):4d}  acc {r.get('accuracy',float('nan')):.3f}  f1 {r.get('macro_f1',float('nan')):.3f}  ece {r.get('ece',float('nan')):.3f}  ({res['suites'][name]['seconds']} s)", flush=True)
        (out_dir / f"laya_list_{tag}.json").write_text(json.dumps(res, indent=1))
        # option-order robustness, Laya section 7: first 200 cases, options permuted with random.Random(99)
        if name in ("massive_intent.en", "en.emotion", "xnli.en") and scored:
            base = s["cases"][:200]; rng = random.Random(99); perm = []
            for c in base:
                order = list(range(len(c["options"]))); rng.shuffle(order)
                perm.append(dict(c, options=[c["options"][i] for i in order], gold=order.index(c["gold"])))
            Pp = score_cases(h, perm, a.batch); flips = n = 0
            for c, cp, p, pp in zip(base, perm, P[:200], Pp):
                if p is None or pp is None: continue
                n += 1; flips += int(c["options"][int(p.argmax())][0] != cp["options"][int(pp.argmax())][0])
            res["option_order_robustness"][name] = {"n": n, "flip_rate": (flips / n if n else None)}
            (out_dir / f"laya_list_{tag}.json").write_text(json.dumps(res, indent=1))
    print("done:", out_dir / f"laya_list_{tag}.json")


if __name__ == "__main__":
    main()
