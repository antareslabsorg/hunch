# Contamination report

**Yes — evaluation items appear in training** for: heldout, dev, calibration. Sets with zero exact-match, zero 13-gram Jaccard≥0.8, and zero MinHash near-duplicate≥0.8 among those scanned: typed-decisions, typed-decisions-v2-system-one.

Date: **2026-09-24**. Runtime: **4477.8s**. Scan: **full (not a sample)** — `--max-train` / `--max-eval` were unset.

## Method

- **State field**: each row's `state` (the request schema, `hunch/schema.py`). A structured state is compared by its string values only, in sorted-key order, so the same text under different field names matches; JSON strings are parsed first when they decode as objects/arrays; plain strings are kept.
- **Normalization** (exact match, n-grams, MinHash): Unicode NFKC, lowercase, collapse runs of whitespace to a single space, strip.
- **Exact match**: equality of normalized state text. Rate = (# eval rows whose normalized state equals ≥1 train row) / n_eval.
- **13-gram Jaccard**: character 13-grams over normalized text. If either side has length < 13, Jaccard is 1.0 iff the strings are equal else 0.0. Overlap hit if max Jaccard vs train ≥ **0.8**.
- **MinHash near-duplicates**: vendored MinHash (datasketch not required), **128** permutations, seed **42**, LSH **32** bands × 4 rows. Near-dup hit if estimated MinHash similarity ≥ **0.8** or verified Jaccard ≥ **0.8**.
- **Candidates** for Jaccard / near-dup / top-10: MinHash LSH hits ∪ exact matches ∪ train rows sharing a rare character 13-gram (train document frequency in 1..200; at most 512 rare-gram candidates per eval, ranked by shared rare-gram count). Rates are verified Jaccard / MinHash on those candidates — not estimated-only.
- **Top-10 pairs**: highest verified Jaccard (tie-break MinHash sim); listed by id and similarity only; this version quotes no dataset text.
- **Train**: `sprint_v3d/train.jsonl`, `v4t/train.jsonl` (n=556933). Never reads a `test.jsonl`.
- **Hub pins**: `LocalLLaMA/typed-decisions` revision `c76749ec58bd`; `pngwn/typed-decisions-v2-system-one` revision `4af44cdb3cb6e7a4374ef6a2e86bbc656e93d29e`.
- **Hardware**: CPU only; no GPU.

## Summary

| Eval set | n_eval | exact-match rate | 13-gram Jaccard≥0.8 | MinHash near-dup≥0.8 |
|---|---:|---:|---:|---:|
| heldout | 66180 | 49/66180 = 0.000740 | 54/66180 = 0.000816 | 54/66180 = 0.000816 |
| dev | 13945 | 573/13945 = 0.041090 | 1435/13945 = 0.102904 | 1551/13945 = 0.111223 |
| calibration | 14002 | 469/14002 = 0.033495 | 1318/14002 = 0.094129 | 1465/14002 = 0.104628 |
| typed-decisions | 400 | 0/400 = 0.000000 | 0/400 = 0.000000 | 0/400 = 0.000000 |
| typed-decisions-v2-system-one | 5214 | 0/5214 = 0.000000 | 0/5214 = 0.000000 | 0/5214 = 0.000000 |

n_train = **556933**.

## What the exact matches are

**heldout**: 49 exact matches; 0 evaluation ids occur in train. 49 of the matched states are under 80 characters (median 10); 49 match a training row from a different source dataset.

| source of the evaluation row | exact matches |
|---|---:|
| `google-research-datasets/go_emotions` | 49 |

**dev**: 573 exact matches; 0 evaluation ids occur in train. 324 of the matched states are under 80 characters (median 55); 17 match a training row from a different source dataset.

| source of the evaluation row | exact matches |
|---|---:|
| `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | 250 |
| `gen.urn` | 158 |
| `google/civil_comments` | 150 |
| `mteb/amazon_massive_intent` | 10 |
| `clinc/clinc_oos` | 4 |
| `google-research-datasets/paws` | 1 |

**calibration**: 469 exact matches; 0 evaluation ids occur in train. 295 of the matched states are under 80 characters (median 52); 17 match a training row from a different source dataset.

| source of the evaluation row | exact matches |
|---|---:|
| `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | 262 |
| `gen.urn` | 130 |
| `google/civil_comments` | 60 |
| `mteb/amazon_massive_intent` | 10 |
| `clinc/clinc_oos` | 3 |
| `google-research-datasets/paws` | 3 |
| `nvidia/HelpSteer2` | 1 |

## heldout

- n_eval = 66180, n_train = 556933
- exact-match: 49/66180 = 0.000740
- 13-gram Jaccard≥0.8: 54/66180 = 0.000816
- MinHash near-dup≥0.8: 54/66180 = 0.000816

### Top 10 most-similar pairs

| rank | Jaccard | MinHash | exact | evaluation id | training id |
|---:|---:|---:|---|---|---|
| 1 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:5398:emotion_fear` | `mteb/amazon_massive_intent:train:97:intent` |
| 2 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:5398:emotion_curiosity` | `mteb/amazon_massive_intent:train:97:intent` |
| 3 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:5398:emotion_pride` | `mteb/amazon_massive_intent:train:97:intent` |
| 4 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:5398:emotion_love` | `mteb/amazon_massive_intent:train:97:intent` |
| 5 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:5398:emotion_admiration` | `mteb/amazon_massive_intent:train:97:intent` |
| 6 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:4715:emotion_fear` | `google/civil_comments:14333:score:toxicity_level` |
| 7 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:4715:emotion_neutral` | `google/civil_comments:14333:score:toxicity_level` |
| 8 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:4715:emotion_curiosity` | `google/civil_comments:14333:score:toxicity_level` |
| 9 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:4715:emotion_pride` | `google/civil_comments:14333:score:toxicity_level` |
| 10 | 1.0000 | 1.0000 | True | `google-research-datasets/go_emotions:4715:emotion_admiration` | `google/civil_comments:14333:score:toxicity_level` |

## dev

- n_eval = 13945, n_train = 556933
- exact-match: 573/13945 = 0.041090
- 13-gram Jaccard≥0.8: 1435/13945 = 0.102904
- MinHash near-dup≥0.8: 1551/13945 = 0.111223

### Top 10 most-similar pairs

| rank | Jaccard | MinHash | exact | evaluation id | training id |
|---:|---:|---:|---|---|---|
| 1 | 1.0000 | 1.0000 | True | `gen.urn:0:4948:b` | `gen.urn:0:2015` |
| 2 | 1.0000 | 1.0000 | True | `gen.urn:0:4948` | `gen.urn:0:2015` |
| 3 | 1.0000 | 1.0000 | True | `gen.urn:0:4884:b` | `gen.urn:0:223` |
| 4 | 1.0000 | 1.0000 | True | `gen.urn:0:4884` | `gen.urn:0:223` |
| 5 | 1.0000 | 1.0000 | True | `gen.urn:0:4828:b` | `gen.urn:0:69` |
| 6 | 1.0000 | 1.0000 | True | `gen.urn:0:4828` | `gen.urn:0:69` |
| 7 | 1.0000 | 1.0000 | True | `gen.urn:0:4823:b` | `gen.urn:0:154` |
| 8 | 1.0000 | 1.0000 | True | `gen.urn:0:4823` | `gen.urn:0:154` |
| 9 | 1.0000 | 1.0000 | True | `gen.urn:0:4752:b` | `gen.urn:0:254` |
| 10 | 1.0000 | 1.0000 | True | `gen.urn:0:4752` | `gen.urn:0:254` |

## calibration

- n_eval = 14002, n_train = 556933
- exact-match: 469/14002 = 0.033495
- 13-gram Jaccard≥0.8: 1318/14002 = 0.094129
- MinHash near-dup≥0.8: 1465/14002 = 0.104628

### Top 10 most-similar pairs

| rank | Jaccard | MinHash | exact | evaluation id | training id |
|---:|---:|---:|---|---|---|
| 1 | 1.0000 | 1.0000 | True | `gen.urn:0:4973:b` | `gen.urn:0:184` |
| 2 | 1.0000 | 1.0000 | True | `gen.urn:0:4973` | `gen.urn:0:184` |
| 3 | 1.0000 | 1.0000 | True | `gen.urn:0:4949:b` | `gen.urn:0:355` |
| 4 | 1.0000 | 1.0000 | True | `gen.urn:0:4949` | `gen.urn:0:355` |
| 5 | 1.0000 | 1.0000 | True | `gen.urn:0:4899:b` | `gen.urn:0:4341` |
| 6 | 1.0000 | 1.0000 | True | `gen.urn:0:4899` | `gen.urn:0:4341` |
| 7 | 1.0000 | 1.0000 | True | `gen.urn:0:4795:b` | `gen.urn:0:1631` |
| 8 | 1.0000 | 1.0000 | True | `gen.urn:0:4795` | `gen.urn:0:1631` |
| 9 | 1.0000 | 1.0000 | True | `gen.urn:0:4777:b` | `gen.urn:0:3744:b` |
| 10 | 1.0000 | 1.0000 | True | `gen.urn:0:4777` | `gen.urn:0:3744:b` |

## typed-decisions

- n_eval = 400, n_train = 556933
- exact-match: 0/400 = 0.000000
- 13-gram Jaccard≥0.8: 0/400 = 0.000000
- MinHash near-dup≥0.8: 0/400 = 0.000000

### Top 10 most-similar pairs

| rank | Jaccard | MinHash | exact | evaluation id | training id |
|---:|---:|---:|---|---|---|
| 1 | 0.0977 | 0.0938 | False | `customer_service_000047` | `mteb/banking77:train:4884:intent` |
| 2 | 0.0656 | 0.0469 | False | `customer_service_000036` | `bitext/Bitext-customer-support-llm-chatbot-training-dataset:11046:intent` |
| 3 | 0.0531 | 0.0469 | False | `customer_service_000069` | `teacher:custom_questions:189:0` |
| 4 | 0.0508 | 0.0312 | False | `customer_service_000089` | `mteb/banking77:train:7784:intent` |
| 5 | 0.0500 | 0.0391 | False | `agent_trace_observability_000091` | `teacher:routing_messages:4741:0` |
| 6 | 0.0488 | 0.0469 | False | `agent_trace_observability_000065` | `teacher:routing_messages:4741:0` |
| 7 | 0.0487 | 0.0312 | False | `customer_service_000028` | `mteb/banking77:train:8803:intent` |
| 8 | 0.0476 | 0.0391 | False | `agent_trace_observability_000029` | `teacher:routing_messages:4741:0` |
| 9 | 0.0442 | 0.0547 | False | `agent_trace_observability_000010` | `teacher:routing_messages:4741:0` |
| 10 | 0.0439 | 0.0625 | False | `customer_service_000041` | `bitext/Bitext-customer-support-llm-chatbot-training-dataset:17317:intent` |

## typed-decisions-v2-system-one

- n_eval = 5214, n_train = 556933
- exact-match: 0/5214 = 0.000000
- 13-gram Jaccard≥0.8: 0/5214 = 0.000000
- MinHash near-dup≥0.8: 0/5214 = 0.000000

### Top 10 most-similar pairs

| rank | Jaccard | MinHash | exact | evaluation id | training id |
|---:|---:|---:|---|---|---|
| 1 | 0.2278 | 0.2422 | False | `typed-decisions-v2-system-one:4671` | `mteb/amazon_massive_intent:train:245:intent` |
| 2 | 0.2245 | 0.2500 | False | `typed-decisions-v2-system-one:4125` | `mteb/amazon_massive_intent:train:8836:intent` |
| 3 | 0.1930 | 0.1797 | False | `typed-decisions-v2-system-one:4788` | `mteb/amazon_massive_intent:train:9834:intent` |
| 4 | 0.1892 | 0.2656 | False | `typed-decisions-v2-system-one:3884` | `google/boolq:train:9221:answer` |
| 5 | 0.1731 | 0.1406 | False | `typed-decisions-v2-system-one:4851` | `mteb/amazon_massive_intent:train:9729:intent` |
| 6 | 0.1724 | 0.1953 | False | `typed-decisions-v2-system-one:4910` | `mteb/banking77:train:6584:intent` |
| 7 | 0.1560 | 0.1484 | False | `typed-decisions-v2-system-one:4677` | `mteb/amazon_massive_intent:train:245:intent` |
| 8 | 0.1560 | 0.1484 | False | `typed-decisions-v2-system-one:4676` | `mteb/amazon_massive_intent:train:245:intent` |
| 9 | 0.1429 | 0.1172 | False | `typed-decisions-v2-system-one:4736` | `teacher:routing_terse:2998:0` |
| 10 | 0.1263 | 0.1328 | False | `typed-decisions-v2-system-one:4513` | `clinc/clinc_oos:train:8864:intent` |

## Reproduce

```bash
python scripts/contamination_scan.py --train data/sprint_v3d/train.jsonl data/v4t/train.jsonl --write results/contamination-full.md \
  --public docs/21-contamination-report.md --dump-ids docs/results/launch/contamination_ids.json
```

Requires `huggingface_hub`, `numpy` and `pyarrow`. MinHash is vendored in the script.
