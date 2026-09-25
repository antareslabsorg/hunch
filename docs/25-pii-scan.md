# PII scan of sprint_v3d bundles — 2026-09-23

**Headline:** Live-looking contact patterns are present on email, phone, street. Zero looks-real hits on iban, card.

## Verdict

Direct answer to “is anything that looks like real PII present?”:

**Yes — live-looking contact PII patterns are present:** 361 hit(s) labelled `looks-real` on email / phone / IBAN / card / street. Also 24843 full/titled-name and 103340 given-name token hit(s) labelled `looks-real` (these include public-figure and dialogue names in published source text). Contact placeholders: 27. See § Examples.

## Method

Scanned explicit paths only (never a directory glob; never `data/sprint_v3d/test.jsonl`). Each JSONL row is parsed; every string value is walked deterministically (`dict` keys sorted). Source family is the row field `family` (present on all inspected rows; `source` / `source_group` also exist but families are the training/evaluation taxonomy used here).

Detectors:

- **email** — `local@domain.tld` with word boundaries.
- **phone** — NANP `(NNN) NNN-NNNN` / `NNN-NNN-NNNN` or `+` international; ≥10 digits. Bare `NNN NNNN` is excluded (collides with legal reporter pins). Windows matching citation markers (`led2d`, `s.ct.`, …) are skipped.
- **IBAN** — `[A-Z]{2}[0-9]{2}` + 11–30 alphanumerics; classified with ISO 13616 mod-97 when parseable.
- **card** — 13–19 digits **with** space/dash separators and Luhn pass; ISBN-13 prefixes `978`/`979` excluded; separator-free digit runs ignored.
- **street** — house number + ≤3 name tokens + suffix. Full-word suffixes must be Capitalized (`Street`, `Avenue`, …); abbreviations (`St.`, `Ave.`, …) allowed. Rejects time/distance collocations (`minute`/`hour`/`people`/… before the suffix).
- **names** — (a) `name_full`: Capitalized First+Last where First ∈ given list and Last ∈ surname list; (b) `name_titled`: Mr/Ms/Dr/… + name on a list; (c) `name_given`: standalone Capitalized given-name token after removing ambiguous English/calendar/color words and skipping tokens that are the first half of a First Last pair.

**Name lists:** US Census 2010 top surnames (2000; https://www2.census.gov/topics/genealogy/2010surnames/names.zip); SSA-derived given names via Hadley baby-names (2000; https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv, 1880-2008). Cached under /tmp/pii_scan_names.

### Placeholder vs looks-real rules

These rules are explicit so a reader can disagree. A hit is `synthetic-placeholder` when any of the following hold; otherwise `looks-real`:

- **email:** domain in {example.com/org/net/edu, test.com/org, localhost, invalid, mailinator.com, fake.com, email.com, domain.com, company.com, yourcompany.com, sample.com} or ends with `.example`/`.test`; or local-part in {user, username, name, email, test, example, sample, foo, bar, baz, noreply, john.doe, jane.doe, …}; or local matches `(user|test|sample|foo|bar)\d*`.
- **phone:** contains NANP fictional `555-01xx` / `555-555` / all-zero / `123-456-7890` / `111-111-1111` patterns; or ≤2 distinct digits.
- **card:** known test PANs; ISBN-13 `978`/`979`; ≤2 distinct digits; or fails Luhn.
- **IBAN:** country `XX`, contains EXAMPLE, low digit diversity, or fails mod-97.
- **street:** contains example/placeholder/lorem/fake; or body tokens in the time/distance stop list; or `N` + {main,first,oak,…} without apt/suite; or `123 Main …`.
- **names:** Alice/Bob/… crypto-persona set; John/Jane Doe; John/Jane Smith; or all tokens in the placeholder-name set.

Redaction in examples: emails keep first character and last two of the local-part and mask the rest; phones/IBAN/cards replace leading digits with `X` (keep last two); streets digit-mask; names keep first letter per token.

## Counts per file

| file | rows | email | phone | iban | card | street | name_full | name_titled | name_given | looks_real | synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train.jsonl` | 244624 | 29 | 45 | 0 | 0 | 190 | 19140 | 2199 | 90423 | 108462 | 3564 |
| `heldout.jsonl` | 66180 | 0 | 13 | 0 | 0 | 82 | 763 | 421 | 6772 | 7893 | 158 |
| `dev.jsonl` | 13945 | 2 | 3 | 0 | 0 | 8 | 1142 | 85 | 5259 | 6362 | 137 |
| `calibration.jsonl` | 14002 | 4 | 3 | 0 | 0 | 9 | 1096 | 113 | 4798 | 5827 | 196 |
| **total** | 338751 | 35 | 64 | 0 | 0 | 289 | 22141 | 2818 | 107252 | 128544 | 4055 |

Column notes: channel columns are total hits (placeholder + looks-real). `looks_real` / `synthetic` sum labelled hits across channels (a row may contribute multiple hits).

### Contact-channel label split (all files)

| channel | looks-real | synthetic-placeholder | total |
| --- | --- | --- | --- |
| `email` | 21 | 14 | 35 |
| `phone` | 63 | 1 | 64 |
| `iban` | 0 | 0 | 0 |
| `card` | 0 | 0 | 0 |
| `street` | 277 | 12 | 289 |

## Counts per source family

| family | email | phone | iban | card | street | name_full | name_titled | name_given | looks_real | synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `heldout.emotion.goemotions` | 0 | 5 | 0 | 0 | 0 | 20 | 0 | 361 | 382 | 4 |
| `heldout.legal.casehold` | 0 | 8 | 0 | 0 | 82 | 301 | 415 | 5794 | 6494 | 106 |
| `heldout.pragmatics.circa` | 0 | 0 | 0 | 0 | 0 | 442 | 6 | 617 | 1017 | 48 |
| `moderation.civil_comments` | 0 | 0 | 0 | 0 | 0 | 4600 | 1470 | 29590 | 34945 | 715 |
| `moderation.score.civil_toxicity_level` | 0 | 0 | 0 | 0 | 0 | 920 | 294 | 5918 | 6989 | 143 |
| `quality.score.helpsteer2` | 35 | 51 | 0 | 0 | 102 | 2479 | 415 | 18053 | 19911 | 1224 |
| `support.routing.intent.banking77` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 |
| `verification.boolq` | 0 | 0 | 0 | 0 | 24 | 3096 | 84 | 12080 | 14840 | 444 |
| `verification.fact.vitaminc` | 0 | 0 | 0 | 0 | 16 | 4899 | 29 | 14565 | 19072 | 437 |
| `verification.paraphrase.paws` | 0 | 0 | 0 | 0 | 65 | 5384 | 105 | 20268 | 24888 | 934 |

## Examples (20, redacted)

Showing 20 unique redacted example(s) (deduped by channel+label+redacted form; selected for channel coverage; budget 20). Labels are `synthetic-placeholder` or `looks-real`.

1. **looks-real** · `email` · file `calibration.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:19080:helpfulness` → `p***ry@p***.org`
2. **looks-real** · `email` · file `calibration.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:18750:helpfulness` → `s**es@s***.com`
3. **looks-real** · `email` · file `dev.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:884:helpfulness` → `w**********11@g***.com`
4. **looks-real** · `email` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:10036:helpfulness` → `B***********pp@s***.gov`
5. **synthetic-placeholder** · `email` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.response` · id `nvidia/HelpSteer2:train:11828:helpfulness` → `E**IL@E***.COM`
6. **synthetic-placeholder** · `email` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.response` · id `nvidia/HelpSteer2:train:7669:helpfulness` → `e*********rk@e***.com`
7. **synthetic-placeholder** · `email` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.response` · id `nvidia/HelpSteer2:train:7669:helpfulness` → `e****le@e***.com`
8. **synthetic-placeholder** · `email` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:18574:helpfulness` → `j******th@e***.com`
9. **looks-real** · `phone` · file `calibration.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:19080:helpfulness` → `(XXX) XXX-XX11`
10. **looks-real** · `phone` · file `dev.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:4622:helpfulness` → `(XXX) XXX-XX83`
11. **looks-real** · `phone` · file `dev.jsonl` · family `quality.score.helpsteer2` · field `state.response` · id `nvidia/HelpSteer2:train:15496:helpfulness` → `XXX-XXX-XX50`
12. **looks-real** · `phone` · file `heldout.jsonl` · family `heldout.emotion.goemotions` · field `state.text` · id `google-research-datasets/go_emotions:3171:emotion_admiration` → `XXX-XXX-XX91`
13. **synthetic-placeholder** · `phone` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.response` · id `nvidia/HelpSteer2:train:9276:helpfulness` → `(XXX) XXX-XX90`
14. **looks-real** · `street` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:8716:helpfulness` → `X Hoosac St.`
15. **looks-real** · `street` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:8716:helpfulness` → `XX Crossway St.`
16. **looks-real** · `street` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:8716:helpfulness` → `XX Walker St.`
17. **looks-real** · `street` · file `train.jsonl` · family `quality.score.helpsteer2` · field `state.prompt` · id `nvidia/HelpSteer2:train:8716:helpfulness` → `XXX Ashland St.`
18. **looks-real** · `name_full` · file `train.jsonl` · family `verification.fact.vitaminc` · field `state.claim` · id `tals/vitaminc:train:706:verdict` → `A*** C****`
19. **looks-real** · `name_full` · file `train.jsonl` · family `verification.fact.vitaminc` · field `state.evidence` · id `tals/vitaminc:train:40:verdict` → `A**** H*****`
20. **looks-real** · `name_full` · file `train.jsonl` · family `verification.fact.vitaminc` · field `state.claim` · id `tals/vitaminc:train:823:verdict` → `B*** J****`

## The 1.7B release's teacher data (scanned 2026-09-24)

The 1.7B's second stage adds 40,612 training questions and 2,175 dev questions, converted from the open Decider's teacher data
(written by `Qwen/Qwen3.5-27B`, see THIRD_PARTY_NOTICES). They were scanned with the same script and name lists:
`python scripts/pii_scan.py --files data/teacher_v1/train.jsonl data/teacher_v1/dev.jsonl`.

| file | rows | email | phone | iban | card | street | name_full | name_titled | name_given | looks_real | synthetic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `train.jsonl` | 40612 | 706 | 74 | 0 | 0 | 534 | 10999 | 2077 | 15478 | 27871 | 1997 |
| `dev.jsonl` | 2175 | 10 | 0 | 0 | 0 | 22 | 572 | 90 | 851 | 1442 | 103 |
| **total** | 42787 | 716 | 74 | 0 | 0 | 556 | 11571 | 2167 | 16329 | 29313 | 2100 |

| channel | looks-real | synthetic-placeholder | total |
| --- | --- | --- | --- |
| `email` | 452 | 264 | 716 |
| `phone` | 23 | 51 | 74 |
| `iban` | 0 | 0 | 0 |
| `card` | 0 | 0 | 0 |
| `street` | 133 | 423 | 556 |

In total it has more live-looking contact patterns than the whole base mixture above, 608 looks-real hits against 361, and
nearly all of the difference is e-mail: 452 against 21. It has fewer live-looking phone numbers (23 against 63) and streets
(133 against 277). The counts cover the train and dev rows together. The text is written by a model, and a model can produce contact details that look real.
This heuristic cannot tell invented details from copied ones, so no claim is made either way. No examples are quoted.

## Reproduce

From the repo root (no GPU; streams train.jsonl):

```bash
python3 scripts/pii_scan.py --write results/pii-base.md                        # the base mixture: every section above the teacher one
python3 scripts/pii_scan.py --files data/teacher_v1/train.jsonl data/teacher_v1/dev.jsonl --write results/pii-teacher.md   # its tables
```

Optional: `--limit N` smoke-tests the first N rows per file; omit for the full scan. Default files: `data/sprint_v3d/train.jsonl`, `data/sprint_v3d/heldout.jsonl`, `data/sprint_v3d/dev.jsonl`, `data/sprint_v3d/calibration.jsonl`.

Report date: 2026-09-23. Script: `scripts/pii_scan.py`. Name-list cache: `/tmp/pii_scan_names`.

