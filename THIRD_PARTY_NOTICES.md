# Third-party notices

Hunch's own code, model cards, evaluation harness, data-building scripts and generated data are Apache-2.0 (`LICENSE`). This
file records what the two released checkpoints are built from, what this repository evaluates them on, and the obligations
those grants carry.

Licenses are recorded **as declared on the artifact we read** (the Hugging Face dataset or model, as it stood when the data
was built on 2026-09-20; the builder reads each source's current revision, see REPRODUCE), with the upstream grant named
separately where the two differ. Reading a dataset to train a model and redistributing derived rows are
different acts. This repository redistributes no training or evaluation rows: `hunch/data/build.py` rebuilds them from the
public sources on your machine.

## Backbone

| Artifact | License | Obligation we carry |
|---|---|---|
| `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B` (fine-tuned) | Apache-2.0 | Retain the license and notice, and state that the weights are a derivative work. Released Hunch weights are Apache-2.0, which Apache-2.0 permits for derivatives. |

## What labels and selection used

No output of a closed model is a training target or a label. Every source must carry the `admitted` flag in
`hunch/data/families.py` (license class A or B, and labels not produced by a closed model), and a release build
(`--release-only`) refuses to run with any other source. The one teacher whose outputs are training targets is named
below, with its licence.

Checkpoint selection is a different matter, and we say so plainly. Choosing the 1.7B release used, among four
measurements, accuracy on `LocalLLaMA/typed-decisions` ([MODEL_CARD](MODEL_CARD.md)). That suite's gold labels are the
averaged answers of a model endpoint the suite does not name.

## Training sources of both released checkpoints

| Source we read | Declared on that artifact | Upstream grant, where it differs | Class | Obligation |
|---|---|---|---|---|
| `mteb/banking77` | MIT | **PolyAI Banking77 is CC BY 4.0.** We honour the upstream grant, not the mirror's re-declaration. | B | Attribution to PolyAI; see the discrepancy note below. |
| `clinc/clinc_oos` | CC BY 3.0 | — | B | Attribution. |
| `mteb/amazon_massive_intent` | Apache-2.0 | **Amazon MASSIVE is CC BY 4.0.** We honour the upstream grant. | B | Attribution to Amazon; see the discrepancy note below. |
| `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | CDLA-Sharing 1.0 | — | B | Share-alike on any redistribution of derived rows (this repository redistributes none). |
| `tals/vitaminc` | CC BY-SA 3.0 | Built on Wikipedia revisions, themselves CC BY-SA. | B | Share-alike, plus Wikipedia attribution. |
| `google/boolq` | CC BY-SA 3.0 | Primary grant verified: "BoolQ is released under the Creative Commons Share-Alike 3.0 license". | B | Share-alike, plus question/passage provenance and Wikipedia attribution. |
| `google-research-datasets/paws`, Wiki configuration only | "other" | Primary grant verified: the dataset "may be freely used for any purpose"; the underlying text is Wikipedia. **PAWS-QQP is excluded** because it requires reconstruction from the separately licensed QQP corpus. | B | Wikipedia attribution and share-alike retained on the sentence text. |
| `google/civil_comments` | CC0-1.0 | — | A | None beyond good practice; we still credit the source. |
| `nvidia/HelpSteer2` | CC BY 4.0 | — | B | Attribution to NVIDIA. |

The instructions, paraphrases and candidate descriptions used with these sources were written for Hunch
(`docs/research/sources/`). They are not quoted upstream prompts.

## Generated in this repository

- **Rule-engine and urn families** (`hunch/data/generators.py`): produced from a seed with exact ground-truth
  distributions. They contain no third-party text.
- **The indirect-answer family** (`hunch/data/indirect_gen.py`, 27,073 training questions, in both released checkpoints' data):
  produced by a rule engine over a hand-written topic graph. No model and no Circa item is involved. It reuses **Circa's
  question instruction and its eight answer labels** verbatim, which is why Circa is not a held-out measurement for these
  checkpoints (the cards say so). Circa (`google-research-datasets/circa`) is CC BY 4.0, and we attribute that wording to
  its authors. (The generator labels its rows licence class A, since they contain no third-party sentence; the attribution
  applies regardless.)
- **Prior augmentation** (`hunch/data/prior_aug.py`): for 10 % of training questions, the state is replaced by another state from
  the same family or by a truncated or shuffled copy of its own, and the target by the family's base rate. It adds no text.

## Additional training data of the 1.7B release

| Source we read | Declared on that artifact | Written by | Class | Obligation |
|---|---|---|---|---|
| `github.com/Mapika/decider`, `teacher_data/` at commit `b44b4c9880a6` | Apache-2.0 | `Qwen/Qwen3.5-27B` (Apache-2.0), per that repository's own generation scripts | B | Attribution and the Apache-2.0 notice. 40,612 questions after dropping those the teacher had re-answered differently; converted by `scripts/build_v1_data.py teacher`, which also removes any state matching a typed-decisions test state or one of our held-out states. Used by the 1.7B release only; not redistributed. |

Qwen3.5's licence permits training other models on its outputs. The per-item generating-model revision is not recorded
upstream, so it is not recorded here either.

## Held-out evaluation sources (no item trained on)

| Source we read | Declared on that artifact | Class | Obligation |
|---|---|---|---|
| `google-research-datasets/circa` | CC BY 4.0 | B | Attribution. Evaluation only. Its instruction and label wording are also used by the generated family above. |
| `coastalcph/lex_glue`, CaseHOLD subset | CC BY 4.0 | B | Attribution to the LexGLUE authors and to the CaseHOLD authors. Evaluation only. The CaseHOLD subset's own primary instrument has not been inspected, so this repository redistributes none of its rows. |
| `google-research-datasets/go_emotions` | Apache-2.0 | A | Attribution. Evaluation only. |

## Public benchmark suites (evaluation only; no split trained on by the released checkpoints)

| Suite | Revision | Declared licence | Use |
|---|---|---|---|
| `LocalLLaMA/typed-decisions` | `c76749ec58bd` | Apache-2.0 | test split, zero-shot; the published leaderboard rows are quoted, not re-run |
| `pngwn/typed-decisions-v2-system-one` | `4af44cdb3cb6` (test data unchanged since 2026-09-16) | none declared on the dataset card | test split, zero-shot |
| The Laya benchmark list (59 suites rebuilt from Laya's published code) | per suite, in each result file's metadata | per source dataset | zero-shot scoring only; no suite is redistributed |

The open `Mapika/decider-0.8b` and `Mapika/decider-2b` models (Apache-2.0) were run through their own published interface to
measure the open bar on typed-decisions. Their code and weights are not redistributed here.

## Two mirror discrepancies, stated plainly

`mteb/banking77` and `mteb/amazon_massive_intent` are convenience mirrors that re-declare their licenses as MIT and
Apache-2.0 respectively, while the corpora they mirror are released under CC BY 4.0 by PolyAI and by Amazon. A mirror cannot
enlarge the rights granted by the original publisher, so we treat both as CC BY 4.0 and carry the attribution obligation.
Anyone rebuilding the data from these mirrors inherits the same obligation, whatever the mirror's card says.

## What is not here

Sources that failed our licence audit are excluded from training and calibration fitting: MNLI, SST-5, HaluEval, Aegis 2.0,
UltraFeedback, and the GLUE and SuperGLUE tasks whose primary grants could not be verified. SNLI (CC BY-SA 4.0) cleared the
audit but is not in the release mixture. The data builder still contains adapters for some of them, marked `admitted=False`,
and a release build refuses them. One exception on the evaluation side: SST-5 is one of the Laya list's 59 zero-shot suites,
and the Laya-list median is one of the four reads behind the 1.7B selection. It is scored, not trained on or redistributed.

## Corrections

If you believe a grant is recorded incorrectly here, please open an issue. We will correct the record.

## Laya's published results

`docs/results/launch/laya_published/t4_colab_benchmark.json` and `app_benchmark_results.json` are verbatim copies of
`research/results/` in https://github.com/NandhaKishorM/laya (Apache License 2.0; git blobs `ddc400a4` and `5f259ff2`,
fetched 2026-09-23). They are kept so the Laya columns of [docs/28](docs/28-laya-list.md) can be checked against their
source. Copyright remains with their authors.

## Vendored code: Laya e-mail state helper

`scripts/_laya_email.py` is a modified copy of `laya/email.py` from https://github.com/NandhaKishorM/laya (Apache License 2.0),
fetched 2026-09-23, with one package-relative import removed so the file runs standalone. It is used only so that the e-mail suites
of the Laya benchmark list (`scripts/bench_laya_list.py`) are built exactly as Laya's own benchmark code builds them. Copyright remains
with its authors; the Apache-2.0 licence text is in that repository and applies to this file. The benchmark construction logic in
`scripts/bench_laya_list.py` was re-implemented from Laya's published scripts (`research/scripts/build_benchmark_nb.py`,
`research/eval/laya_eval.py`, `research/scripts/bench_apps.py`), also Apache-2.0; instruction strings, option-sampling rules and
metric definitions are reproduced deliberately so the suites are identical.
