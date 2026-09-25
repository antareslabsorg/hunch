"""Adapters from public Hugging Face datasets to `hunch-data-v1` records for the sprint families.

Every adapter declares its license class and yields Records with candidate *descriptions* (never bare ids),
a `source_group` for split assignment, and a tagged target. Field names are checked up front so a schema
change in the upstream dataset fails loudly instead of producing garbage.

Candidate descriptions for large intent sets are auto-humanized from label names for the sprint; the
the authored specs in docs/research/sources replace them with authored descriptions.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterator

from ..schema import Candidate, Question, Record, Target, one_hot
from . import authored
from .authored import pick_instruction

# ----------------------------------------------------------------------------------------------------------
# helpers


def group_hash(text: str, seed: int = 0) -> float:
    h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / 16**12


def assign_split(group: str, has_official_eval: bool, seed: int = 0) -> str:
    """Deterministic split by source group. With an official eval split: train 90 / dev 5 / cal 5.
    Without one: train 85 / dev 5 / cal 5 / test 5 (test carved by group)."""
    u = group_hash(group, seed)
    if has_official_eval:
        return "train" if u < 0.90 else ("dev" if u < 0.95 else "calibration")
    return "train" if u < 0.85 else ("dev" if u < 0.90 else ("calibration" if u < 0.95 else "test"))


def humanize(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").replace(".", " ").strip().lower()


def _require(ds, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in ds.column_names]
    if missing:
        raise ValueError(f"{name}: expected columns {missing} not found; have {ds.column_names}")


def label_names(ds, col: str) -> tuple[list[str], dict | None]:
    """Class names for an int ClassLabel column, or (sorted unique strings, str->index) for a string label column."""
    feat = ds.features[col]
    names = getattr(feat, "names", None)
    if names is not None:
        return list(names), None
    values = sorted({str(v) for v in ds[col]})
    if not values or len(values) > 1000:
        raise ValueError(f"label column {col!r} is neither ClassLabel nor a small string set ({len(values)} values)")
    return values, {v: i for i, v in enumerate(values)}


@dataclass
class Source:
    name: str  # HF id
    family: str
    license_class: str
    build: Callable[..., Iterator[Record]]
    note: str = ""
    revision: str | None = None  # e.g. "refs/convert/parquet" for script-based repos
    admitted: bool = True  # may back a released checkpoint (license class A/B AND labels not produced by a closed model)


def _hf(name: str, config: str | None = None, split: str | None = None, revision: str | None = None, **kw):
    from datasets import load_dataset

    return load_dataset(name, config, split=split, revision=revision, **kw)


def _dist_record(source: Source, idx: str, group: str, split: str, state, question: Question, probs: list[float], hard: int | None,
                 semantic_kind: str, provenance: str, meta: dict | None = None) -> Record:
    return Record(
        id=f"{source.name}:{idx}:{question.id}", source=source.name, source_group=group, family=source.family, split=split,
        license_class=source.license_class, state=state, question=question,
        target=Target(semantic_kind=semantic_kind, provenance=provenance, probs=probs, hard=hard), meta=meta or {},
    )


def _hard_record(source: Source, idx: str, group: str, split: str, state, question: Question, hard: int, meta: dict | None = None,
                 provenance: str = "adjudicated_human_label") -> Record:
    """A one-hot target from a dataset label. Human labels are adjudicated labels, not deterministic truth;
    generators pass provenance='deterministic_truth' explicitly."""
    return _dist_record(source, idx, group, split, state, question, one_hot(question.k, hard), hard, "label_distribution", provenance, meta)


# ----------------------------------------------------------------------------------------------------------
# support routing: intent classification (Choice, large K)


def _intent_source(source: Source, text_col: str, label_col: str, instruction: str, config: str | None, eval_splits: list[str],
                   max_per_split: int | None, descriptions: dict[str, str] | None = None, drop_labels: set[str] | None = None,
                   label_text_col: str | None = None, spec_name: str | None = None, spec_section: str | None = None):
    def build(max_per_split=max_per_split):
        sp = authored.spec(spec_name) if spec_name else None
        authored_desc = sp.descriptions(spec_section) if sp else {}
        bank = sp.instructions(spec_section) if sp else None
        for hf_split in ["train", *eval_splits]:
            ds = _hf(source.name, config, split=hf_split, revision=source.revision)
            _require(ds, [text_col, label_col] + ([label_text_col] if label_text_col else []), source.name)
            if label_text_col:  # int label + separate name column (mirror datasets without ClassLabel metadata)
                pairs = sorted({(int(a), str(b)) for a, b in zip(ds[label_col], ds[label_text_col])})
                if [a for a, _ in pairs] != list(range(len(pairs))):
                    raise ValueError(f"{source.name}: label ids are not 0..K-1: {[a for a, _ in pairs][:10]}")
                names, str_index = [b for _, b in pairs], None
            else:
                names, str_index = label_names(ds, label_col)
            desc = {**(descriptions or {}), **authored_desc}
            cands = [Candidate(id=n, label=humanize(n), description=desc.get(n, "")) for n in names if n not in (drop_labels or set())]
            keep = [i for i, n in enumerate(names) if n not in (drop_labels or set())]
            index_of = {orig: new for new, orig in enumerate(keep)}
            n = 0
            for i, row in enumerate(ds):
                raw = row[label_col]
                lab = str_index.get(str(raw)) if str_index is not None else raw
                if lab not in index_of:
                    continue
                row = {**row, label_col: lab}
                group = f"{source.name}:{hf_split}:{i}"
                split = "test" if hf_split != "train" else assign_split(group, has_official_eval=bool(eval_splits))
                q = Question(id="intent", type="choice", instruction=pick_instruction(bank, instruction, group, split), candidates=cands)
                yield _hard_record(source, f"{hf_split}:{i}", group, split, {"message": row[text_col]}, q, index_of[row[label_col]])
                n += 1
                if max_per_split and n >= max_per_split:
                    break
    source.build = build
    return source


BANKING77 = _intent_source(
    Source("mteb/banking77", "support.routing.intent.banking77", "B", None, "77 fine-grained banking intents (parquet mirror of PolyAI/banking77, CC-BY-4.0)"),
    "text", "label",
    "A customer of an online bank sent the message in `message`. Which single intent best describes what the customer wants?",
    None, ["test"], 20000, label_text_col="label_text", spec_name="banking77")

CLINC150 = _intent_source(
    Source("clinc/clinc_oos", "support.routing.intent.clinc150", "B", None, "150 intents + out-of-scope"),
    "text", "intent",
    "A user of a voice assistant said the utterance in `message`. Which single intent best describes it? Choose the out-of-scope option if none applies.",
    "plus", ["validation", "test"], 20000, spec_name="clinc150")

MASSIVE_EN = _intent_source(
    Source("mteb/amazon_massive_intent", "support.routing.intent.massive", "B", None, "60 assistant intents, en (parquet mirror of MASSIVE; upstream is script-based)"),
    "text", "label",
    "A user of a smart assistant said the utterance in `message`. Which single intent best describes the request?",
    "en", ["validation", "test"], 12000, spec_name="massive-en")


def _bitext_build(max_per_split=20000):
    src = BITEXT
    ds = _hf(src.name, split="train")
    _require(ds, ["instruction", "intent", "category"], src.name)
    intents = sorted(set(ds["intent"]))
    categories = sorted(set(ds["category"]))
    sp = authored.spec("bitext")
    idesc = sp.descriptions("intent candidates") if sp else {}
    cdesc = sp.descriptions("category candidates") if sp else {}
    ibank = sp.instructions("intent instruction") if sp else None
    cbank = sp.instructions("category instruction") if sp else None
    ic = [Candidate(id=n, label=humanize(n), description=idesc.get(n, "")) for n in intents]
    cc = [Candidate(id=n, label=humanize(n), description=cdesc.get(n, "")) for n in categories]
    n = 0
    for i, row in enumerate(ds):
        group = f"{src.name}:{i}"
        split = assign_split(group, has_official_eval=False)
        state = {"message": row["instruction"]}
        q1 = Question(id="intent", type="choice", candidates=ic,
                      instruction=pick_instruction(ibank, "A customer wrote the support message in `message`. Which single intent best describes the request?", group + ":i", split))
        yield _hard_record(src, str(i), group, split, state, q1, intents.index(row["intent"]))
        q2 = Question(id="category", type="choice", candidates=cc,
                      instruction=pick_instruction(cbank, "A customer wrote the support message in `message`. Which support category should handle it?", group + ":c", split))
        yield _hard_record(src, str(i), group, split, state, q2, categories.index(row["category"]))
        n += 1
        if max_per_split and n >= max_per_split:
            break


BITEXT = Source("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "support.routing.bitext", "B", None, "27 intents, 11 categories")
BITEXT.build = _bitext_build


# ----------------------------------------------------------------------------------------------------------
# evidence verification (Choice 3 / Boolean)

NLI_CANDS = [
    Candidate("entailment", "entailment", "The premise implies that the hypothesis is true."),
    Candidate("neutral", "neutral", "The premise neither implies nor rules out the hypothesis."),
    Candidate("contradiction", "contradiction", "The premise implies that the hypothesis is false."),
]
NLI_INSTRUCTION = "Read `premise` and `hypothesis`. Assuming the premise is true, what is the relationship of the hypothesis to it?"


def _mnli_build(max_per_split=20000):
    src = MNLI
    for hf_split in ["train", "validation_matched", "validation_mismatched"]:
        ds = _hf(src.name, split=hf_split)
        _require(ds, ["premise", "hypothesis", "label"], src.name)
        n = 0
        for i, row in enumerate(ds):
            if row["label"] not in (0, 1, 2):
                continue
            group = f"{src.name}:premise:{hashlib.sha1(row['premise'].encode()).hexdigest()[:16]}"
            split = "test" if hf_split != "train" else assign_split(group, True)
            q = Question(id="nli", type="choice", instruction=NLI_INSTRUCTION, candidates=NLI_CANDS)
            yield _hard_record(src, f"{hf_split}:{i}", group, split, {"premise": row["premise"], "hypothesis": row["hypothesis"]}, q, row["label"],
                               meta={"genre": row.get("genre")})
            n += 1
            if max_per_split and n >= max_per_split:
                break


MNLI = Source("nyu-mll/multi_nli", "verification.nli.mnli", "D", None, "licensing memo: unscreened genre mixture, exclude from release training", admitted=False)
MNLI.build = _mnli_build


def _snli_build(max_per_split=20000):
    """SNLI (CC BY-SA 4.0): the admitted NLI family. Added after the release mixture v3, which has no NLI source, lost
    the Circa transfer that the MNLI-containing research mixture had. Rows without a gold label (-1) are skipped."""
    src = SNLI
    for hf_split in ["train", "validation", "test"]:
        ds = _hf(src.name, "plain_text", split=hf_split)
        _require(ds, ["premise", "hypothesis", "label"], src.name)
        n = 0
        for i, row in enumerate(ds):
            if row["label"] not in (0, 1, 2):
                continue
            group = f"{src.name}:premise:{hashlib.sha1(row['premise'].encode()).hexdigest()[:16]}"
            split = "test" if hf_split != "train" else assign_split(group, True)
            q = Question(id="nli", type="choice", instruction=NLI_INSTRUCTION, candidates=NLI_CANDS)
            yield _hard_record(src, f"{hf_split}:{i}", group, split, {"premise": row["premise"], "hypothesis": row["hypothesis"]}, q, row["label"])
            n += 1
            if max_per_split and n >= max_per_split:
                break


SNLI = Source("stanfordnlp/snli", "verification.nli.snli", "B", None, "CC BY-SA 4.0 (Bowman et al. 2015); caption-derived sentence pairs; opt-in via --extra-sources, not part of v3")
SNLI.build = _snli_build

FACT_CANDS = [
    Candidate("supports", "supported", "The evidence supports the claim: given the evidence, the claim is true."),
    Candidate("refutes", "refuted", "The evidence refutes the claim: given the evidence, the claim is false."),
    Candidate("nei", "not enough information", "The evidence is insufficient to decide whether the claim is true or false."),
]
FACT_LABELS = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


def _vitaminc_build(max_per_split=20000):
    src = VITAMINC
    sp = authored.spec("vitaminc")
    bank = sp.instructions() if sp else None
    cands = FACT_CANDS
    if sp and sp.descriptions():
        d = sp.descriptions()  # keyed by upstream labels SUPPORTS / REFUTES / NOT ENOUGH INFO
        cands = [Candidate(c.id, c.label, d.get(up, c.description)) for c, up in zip(FACT_CANDS, ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"))]
    for hf_split in ["train", "validation", "test"]:
        ds = _hf(src.name, split=hf_split)
        _require(ds, ["claim", "evidence", "label"], src.name)
        n = 0
        for i, row in enumerate(ds):
            if row["label"] not in FACT_LABELS:
                continue
            page = row.get("page") or row["claim"]
            group = f"{src.name}:page:{hashlib.sha1(str(page).encode()).hexdigest()[:16]}"
            split = "test" if hf_split != "train" else assign_split(group, True)
            q = Question(id="verdict", type="choice", candidates=cands,
                         instruction=pick_instruction(bank, "Does the text in `evidence` support or refute the statement in `claim`, or is it insufficient to tell?", group, split))
            yield _hard_record(src, f"{hf_split}:{i}", group, split, {"claim": row["claim"], "evidence": row["evidence"]}, q, FACT_LABELS[row["label"]])
            n += 1
            if max_per_split and n >= max_per_split:
                break


VITAMINC = Source("tals/vitaminc", "verification.fact.vitaminc", "B", None, "contrastive evidence, 3-way")
VITAMINC.build = _vitaminc_build


def _boolq_build(max_per_split=20000):
    src = BOOLQ
    sp = authored.spec("boolq")
    bank = sp.instructions() if sp else None
    crit = (sp.criteria() if sp else None) or {"true": "The passage supports answering yes.", "false": "The passage supports answering no."}
    for hf_split in ["train", "validation"]:
        ds = _hf(src.name, split=hf_split)
        _require(ds, ["question", "passage", "answer"], src.name)
        n = 0
        for i, row in enumerate(ds):
            group = f"{src.name}:passage:{hashlib.sha1(row['passage'].encode()).hexdigest()[:16]}"
            split = "test" if hf_split != "train" else assign_split(group, True)
            q = Question(id="answer", type="boolean", criteria=crit,
                         instruction=pick_instruction(bank, "Based only on `passage`, is the answer to `question` yes?", group, split))
            yield _hard_record(src, f"{hf_split}:{i}", group, split, {"passage": row["passage"], "question": row["question"]}, q, int(bool(row["answer"])))
            n += 1
            if max_per_split and n >= max_per_split:
                break


BOOLQ = Source("google/boolq", "verification.boolq", "B", None)
BOOLQ.build = _boolq_build


def _halueval_build(max_per_split=10000):
    src = HALUEVAL
    ds = _hf(src.name, "qa", split="data")
    _require(ds, ["knowledge", "question", "right_answer", "hallucinated_answer"], src.name)
    n = 0
    for i, row in enumerate(ds):
        group = f"{src.name}:{i}"
        split = assign_split(group, has_official_eval=False)
        for tag, ans, hard in (("right", row["right_answer"], 1), ("halu", row["hallucinated_answer"], 0)):
            q = Question(id="grounded", type="boolean",
                         instruction="Given `knowledge` and `question`, is `answer` fully supported by the knowledge, with no invented or contradicted facts?",
                         criteria={"true": "The answer is grounded in the knowledge.", "false": "The answer contains hallucinated or contradicted content."})
            yield _hard_record(src, f"{i}:{tag}", group, split, {"knowledge": row["knowledge"], "question": row["question"], "answer": ans}, q, hard)
        n += 1
        if max_per_split and n >= max_per_split:
            break


HALUEVAL = Source("pminervini/HaluEval", "verification.grounding.halueval", "A", None, "ChatGPT-generated answers: research-only under the open-teacher rule", admitted=False)
HALUEVAL.build = _halueval_build


def _paws_build(max_per_split=20000):
    src = PAWS
    sp = authored.spec("paws")
    bank = sp.instructions() if sp else None
    crit = (sp.criteria() if sp else None) or {"true": "They are paraphrases.", "false": "They differ in meaning, even if they share most words."}
    for hf_split in ["train", "validation", "test"]:
        ds = _hf(src.name, "labeled_final", split=hf_split)
        _require(ds, ["sentence1", "sentence2", "label"], src.name)
        n = 0
        for i, row in enumerate(ds):
            group = f"{src.name}:{hf_split}:{i}"
            split = "test" if hf_split != "train" else assign_split(group, True)
            q = Question(id="paraphrase", type="boolean", criteria=crit,
                         instruction=pick_instruction(bank, "Do `sentence1` and `sentence2` have the same meaning?", group, split))
            yield _hard_record(src, f"{hf_split}:{i}", group, split, {"sentence1": row["sentence1"], "sentence2": row["sentence2"]}, q, int(row["label"]))
            n += 1
            if max_per_split and n >= max_per_split:
                break


PAWS = Source("google-research-datasets/paws", "verification.paraphrase.paws", "B", None, "PAWS-Wiki (labeled_final): 'may be freely used for any purpose', Wikipedia attribution/share-alike retained")
PAWS.build = _paws_build


# ----------------------------------------------------------------------------------------------------------
# content moderation (Boolean with soft human labels; Choice over categories)

CIVIL_ATTRS = {
    "toxicity": "rude, disrespectful, or unreasonable enough to make someone leave a discussion",
    "insult": "insulting or inflammatory toward a person or group",
    "threat": "describing an intention to inflict harm",
    "identity_attack": "attacking a person or group on the basis of identity",
    "obscene": "obscene or profane",
}


CIVIL_SPEC_ORDER = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]  # block order in the spec


def _civil_build(max_per_split=20000):
    src = CIVIL_COMMENTS
    ds = _hf(src.name, split="train")
    _require(ds, ["text", *CIVIL_ATTRS], src.name)
    sp = authored.spec("civil-comments")
    banks = {a: (sp.instructions(f"{a}:") if sp else None) for a in CIVIL_ATTRS}
    crits = {a: ((sp.criteria("Candidate semantics", CIVIL_SPEC_ORDER.index(a)) if sp else None)
                 or {"true": f"Yes, the comment is {a.replace('_', ' ')}.", "false": f"No, the comment is not {a.replace('_', ' ')}."}) for a in CIVIL_ATTRS}
    n = 0
    for i, row in enumerate(ds):
        if not row["text"] or not row["text"].strip():
            continue
        group = f"{src.name}:{i}"
        split = assign_split(group, has_official_eval=False)
        for attr, desc in CIVIL_ATTRS.items():
            p = float(row[attr])
            if not 0.0 <= p <= 1.0:
                continue
            q = Question(id=attr, type="boolean", criteria=crits[attr],
                         instruction=pick_instruction(banks[attr], f"Would a typical reader consider the comment in `comment` {desc}?", f"{group}:{attr}", split))
            yield _dist_record(src, f"{i}", group, split, {"comment": row["text"]}, q, [1.0 - p, p], int(p >= 0.5),
                               "label_distribution", "human_annotation_distribution")
        n += 1
        if max_per_split and n >= max_per_split:
            break


CIVIL_COMMENTS = Source("google/civil_comments", "moderation.civil_comments", "A", None, "annotator fractions -> soft Booleans")
CIVIL_COMMENTS.build = _civil_build


def _aegis_build(max_per_split=20000):
    src = AEGIS
    ds = _hf(src.name, split="train")
    cols = set(ds.column_names)
    prompt_col = "prompt" if "prompt" in cols else None
    label_col = next((c for c in ("prompt_label", "prompt_safety", "label") if c in cols), None)
    if not prompt_col or not label_col:
        raise ValueError(f"{src.name}: unexpected columns {sorted(cols)}")
    sp = authored.spec("aegis-2.0")
    bank = sp.instructions("prompt safety") if sp else None
    crit = (sp.criteria("prompt_label") if sp else None) or {"true": "The request is unsafe and should be refused or escalated.", "false": "The request is safe."}
    n = 0
    for i, row in enumerate(ds):
        text = row[prompt_col]
        lab = str(row[label_col]).lower()
        if not text or lab not in ("safe", "unsafe"):
            continue
        group = f"{src.name}:{i}"
        split = assign_split(group, has_official_eval=False)
        q = Question(id="unsafe", type="boolean", criteria=crit,
                     instruction=pick_instruction(bank, "Under a general content safety policy (violence, sexual content, harassment, hate, self-harm, criminal planning, weapons, privacy violations), is the user request in `prompt` unsafe?", group, split))
        yield _hard_record(src, str(i), group, split, {"prompt": text}, q, int(lab == "unsafe"))
        n += 1
        if max_per_split and n >= max_per_split:
            break


AEGIS = Source("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "moderation.aegis2", "B", None, "licensing memo: model-derived strata unverified, hold out of release training", admitted=False)
AEGIS.build = _aegis_build


# ----------------------------------------------------------------------------------------------------------
# ordered Score data

def _ultrafeedback_build(max_per_split=20000):
    src = ULTRAFEEDBACK
    ds = _hf(src.name, split="train")
    _require(ds, ["instruction", "completions"], src.name)
    levels = [
        "Not helpful: the response is off-topic, wrong, or fails to address the request.",
        "Slightly helpful: touches the request but with major gaps or errors.",
        "Moderately helpful: addresses the request with noticeable omissions or inaccuracies.",
        "Helpful: addresses the request accurately with minor issues.",
        "Highly helpful: complete, accurate, and well suited to the request.",
    ]
    n = 0
    for i, row in enumerate(ds):
        group = f"{src.name}:{i}"
        split = assign_split(group, has_official_eval=False)
        for j, comp in enumerate(row["completions"] or []):
            ann = (comp.get("annotations") or {}).get("helpfulness") or {}
            rating = ann.get("Rating")
            if rating is None or str(rating).strip() not in {"1", "2", "3", "4", "5"}:
                continue
            q = Question(id="helpfulness", type="score", instruction="How helpful is `response` as an answer to `instruction`?", levels=levels)
            yield _hard_record(src, f"{i}:{j}", group, split, {"instruction": row["instruction"], "response": comp.get("response", "")}, q, int(rating) - 1)
        n += 1
        if max_per_split and n >= max_per_split:
            break


ULTRAFEEDBACK = Source("openbmb/UltraFeedback", "quality.score.ultrafeedback_helpfulness", "A", None, "1-5 helpfulness -> Score; ratings are GPT-4 generated: research-only under the open-teacher rule", admitted=False)
ULTRAFEEDBACK.build = _ultrafeedback_build


def _sst5_build(max_per_split=20000):
    src = SST5
    levels = ["Very negative.", "Negative.", "Neutral.", "Positive.", "Very positive."]
    sp = authored.spec("sst-5")
    bank = sp.instructions() if sp else None
    if sp and len(sp.descriptions()) == 5:
        levels = [sp.descriptions()[str(k)] for k in range(5)]
    for hf_split in ["train", "validation", "test"]:
        ds = _hf(src.name, split=hf_split)
        _require(ds, ["text", "label"], src.name)
        n = 0
        for i, row in enumerate(ds):
            group = f"{src.name}:{hf_split}:{i}"
            split = "test" if hf_split != "train" else assign_split(group, True)
            q = Question(id="sentiment", type="score", levels=levels,
                         instruction=pick_instruction(bank, "What is the sentiment expressed in the movie review sentence in `text`?", group, split))
            yield _hard_record(src, f"{hf_split}:{i}", group, split, {"text": row["text"]}, q, int(row["label"]))
            n += 1
            if max_per_split and n >= max_per_split:
                break


SST5 = Source("SetFit/sst5", "sentiment.score.sst5", "D", None, "licensing memo: exclude; sprint-only", admitted=False)
SST5.build = _sst5_build


# ----------------------------------------------------------------------------------------------------------
# release-admitted Score data (human-derived): Civil Comments toxicity as 5 ordered levels, HelpSteer2 helpfulness

TOX_LEVELS = [
    "No toxicity: essentially no reader would call the comment rude or disrespectful.",
    "Slight toxicity: a small minority of readers would find it rude or disrespectful.",
    "Moderate toxicity: readers are split on whether the comment is rude or disrespectful.",
    "High toxicity: most readers would find the comment rude, disrespectful, or unreasonable.",
    "Severe toxicity: nearly every reader would call the comment rude, hateful, or abusive.",
]


def ordinal_soft_target(fraction: float, k: int) -> list[float]:
    """Encode an annotator fraction in [0,1] as a soft distribution over k ordered levels by linear interpolation
    between the two adjacent levels (mass sums to 1; exact at level centers)."""
    x = min(max(fraction, 0.0), 1.0) * (k - 1)
    lo = int(x); hi = min(lo + 1, k - 1); w = x - lo
    t = [0.0] * k
    t[lo] += 1.0 - w
    t[hi] += w
    return t


def _civil_score_build(max_per_split=20000):
    src = CIVIL_SCORE
    ds = _hf("google/civil_comments", split="train")
    _require(ds, ["text", "toxicity"], src.name)
    n = 0
    for i, row in enumerate(ds):
        if not row["text"] or not row["text"].strip():
            continue
        p = float(row["toxicity"])
        if not 0.0 <= p <= 1.0:
            continue
        group = f"google/civil_comments:{i}"  # same group key as the Boolean adapter -> same split
        split = assign_split(group, has_official_eval=False)
        q = Question(id="toxicity_level", type="score", levels=TOX_LEVELS,
                     instruction="How toxic would typical readers consider the comment in `comment`? Choose the level that best matches the share of readers who would call it rude, disrespectful, or unreasonable.")
        t = ordinal_soft_target(p, 5)
        yield _dist_record(src, f"{i}:score", group, split, {"comment": row["text"]}, q, t, int(max(range(5), key=t.__getitem__)),
                           "label_distribution", "human_annotation_distribution", meta={"encoding": "ordinal interpolation of annotator fraction"})
        n += 1
        if max_per_split and n >= max_per_split:
            break


CIVIL_SCORE = Source("google/civil_comments", "moderation.score.civil_toxicity_level", "A", None, "annotator fraction -> 5-level Score (human-derived)")
CIVIL_SCORE.build = _civil_score_build


def _helpsteer2_train_build(max_per_split=20000):
    """HelpSteer2 helpfulness (0-4, human ratings, CC-BY-4.0) as an admitted Score training family; splits by prompt group,
    official validation -> test. Used instead of the held-out variant in release builds."""
    src = HELPSTEER2_TRAIN
    levels = ["Not helpful at all.", "Slightly helpful.", "Partially helpful.", "Mostly helpful.", "Perfectly helpful."]
    for hf_split in ["train", "validation"]:
        ds = _hf("nvidia/HelpSteer2", split=hf_split)
        _require(ds, ["prompt", "response", "helpfulness"], src.name)
        n = 0
        for i, row in enumerate(ds):
            h = row["helpfulness"]
            if h is None or not 0 <= int(h) <= 4:
                continue
            group = f"nvidia/HelpSteer2:prompt:{hashlib.sha1(row['prompt'].encode()).hexdigest()[:16]}"
            split = "test" if hf_split != "train" else assign_split(group, True)
            q = Question(id="helpfulness", type="score", instruction="How helpful is `response` as an answer to `prompt`?", levels=levels)
            yield _hard_record(src, f"{hf_split}:{i}", group, split, {"prompt": row["prompt"], "response": row["response"]}, q, int(h))
            n += 1
            if max_per_split and n >= max_per_split:
                break


HELPSTEER2_TRAIN = Source("nvidia/HelpSteer2", "quality.score.helpsteer2", "B", None, "human 0-4 helpfulness ratings -> Score (release-admitted training variant)")
HELPSTEER2_TRAIN.build = _helpsteer2_train_build


# ----------------------------------------------------------------------------------------------------------
# held-out families (never trained in the sprint)

def _circa_build(max_per_split=None):
    src = CIRCA
    ds = _hf(src.name, split="train")
    _require(ds, ["context", "question-X", "answer-Y", "goldstandard1"], src.name)
    names, _ = label_names(ds, "goldstandard1")
    cands = [Candidate(id=str(k), label=n, description="") for k, n in enumerate(names)]
    for i, row in enumerate(ds):
        lab = row["goldstandard1"]
        if lab is None or lab < 0:
            continue
        group = f"{src.name}:{i}"
        q = Question(id="meaning", type="choice",
                     instruction="In the situation described in `context`, X asks Y the question in `question` and Y replies with `answer`. What does Y's indirect answer mean?", candidates=cands)
        yield _hard_record(src, str(i), group, "heldout", {"context": row["context"], "question": row["question-X"], "answer": row["answer-Y"]}, q, int(lab))


CIRCA = Source("google-research-datasets/circa", "heldout.pragmatics.circa", "B", None, "indirect yes/no answers")
CIRCA.build = _circa_build


def _casehold_build(max_per_split=None):
    src = CASEHOLD
    for hf_split in ["validation", "test"]:
        ds = _hf(src.name, "case_hold", split=hf_split)
        _require(ds, ["context", "endings", "label"], src.name)
        for i, row in enumerate(ds):
            cands = [Candidate(id=str(k), label=e) for k, e in enumerate(row["endings"])]
            q = Question(id="holding", type="choice", instruction="The legal excerpt in `context` cites a case and ends where the holding should be stated. Which candidate is the correct holding?", candidates=cands)
            yield _hard_record(src, f"{hf_split}:{i}", f"{src.name}:{hf_split}:{i}", "heldout", {"context": row["context"]}, q, int(row["label"]))


CASEHOLD = Source("coastalcph/lex_glue", "heldout.legal.casehold", "B", None)
CASEHOLD.build = _casehold_build


def _goemotions_build(max_per_split=3000):
    src = GOEMOTIONS
    ds = _hf(src.name, "simplified", split="test")
    _require(ds, ["text", "labels"], src.name)
    names = ds.features["labels"].feature.names
    for i, row in enumerate(ds):
        if i >= (max_per_split or 10**9):
            break
        present = set(row["labels"])
        for k in (present | set(range(0, len(names), 7))):  # every present emotion plus a fixed sample of absent ones
            name = names[k]
            q = Question(id=f"emotion_{name}", type="boolean", instruction=f"Does the Reddit comment in `text` express the emotion '{name}'?")
            yield _hard_record(src, f"{i}", f"{src.name}:{i}", "heldout", {"text": row["text"]}, q, int(k in present))


GOEMOTIONS = Source("google-research-datasets/go_emotions", "heldout.emotion.goemotions", "A", None)
GOEMOTIONS.build = _goemotions_build


def _helpsteer2_build(max_per_split=3000):
    src = HELPSTEER2
    ds = _hf(src.name, split="validation")
    _require(ds, ["prompt", "response", "helpfulness"], src.name)
    levels = ["Not helpful at all.", "Slightly helpful.", "Partially helpful.", "Mostly helpful.", "Perfectly helpful."]
    for i, row in enumerate(ds):
        if i >= (max_per_split or 10**9):
            break
        h = row["helpfulness"]
        if h is None or not 0 <= int(h) <= 4:
            continue
        q = Question(id="helpfulness", type="score", instruction="How helpful is `response` as an answer to `prompt`?", levels=levels)
        yield _hard_record(src, str(i), f"{src.name}:{i}", "heldout", {"prompt": row["prompt"], "response": row["response"]}, q, int(h))


HELPSTEER2 = Source("nvidia/HelpSteer2", "heldout.quality.helpsteer2", "B", None)
HELPSTEER2.build = _helpsteer2_build


SPRINT_TRAIN_SOURCES: list[Source] = [BANKING77, CLINC150, MASSIVE_EN, BITEXT, MNLI, VITAMINC, BOOLQ, HALUEVAL, PAWS, CIVIL_COMMENTS, AEGIS, ULTRAFEEDBACK, SST5]
SPRINT_HELDOUT_SOURCES: list[Source] = [CIRCA, CASEHOLD, GOEMOTIONS, HELPSTEER2]
# release build: admitted train sources only, HelpSteer2 moved into training as the human-rated Score family,
# Civil Comments toxicity added as an ordinal Score; held-out = Circa, CaseHOLD, GoEmotions
RELEASE_TRAIN_SOURCES: list[Source] = [s for s in SPRINT_TRAIN_SOURCES if s.admitted] + [CIVIL_SCORE, HELPSTEER2_TRAIN]
RELEASE_HELDOUT_SOURCES: list[Source] = [CIRCA, CASEHOLD, GOEMOTIONS]
ALL_SOURCES = {s.name: s for s in SPRINT_TRAIN_SOURCES + SPRINT_HELDOUT_SOURCES}
ALL_SOURCES.update({f"{CIVIL_SCORE.name}#score": CIVIL_SCORE, f"{HELPSTEER2_TRAIN.name}#train": HELPSTEER2_TRAIN})
ALL_SOURCES[SNLI.name] = SNLI  # opt-in extra source (see build --extra-sources)
