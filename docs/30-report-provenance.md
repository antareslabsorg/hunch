# Where the technical report's numbers come from

The technical report, [*Hunch: Open-Weight Decision Scorers*](https://antareslabs.org/hunch/hunch-technical-report.pdf),
prints every release number from a macro that its build tools write from byte-for-byte copies of the files named below. Before
writing, the tools recompute from the per-question rows every quantity they can (temperatures; dev, sealed and held-out metrics;
bootstrap intervals and paired differences; Laya-list summaries; probe rates; contamination counts; the on-device and
shared-prefix gate statistics; the comparisons with Jev) and stop on any disagreement. The report's source and build tools are
not in this repository. Every path in the report, and every path here, is a path in this repository.

## Release results

| Report section | Results | Files | How the number is obtained |
|---|---|---|---|
| 9.1, Tables 7, 19 and 20, Figure 3, Appendix D | Sealed test read, per family, beside the untrained backbone and each family's constant predictor; questions whose state also occurs in training | `docs/results/launch/sealed.json`, `docs/results/launch/sealed/`, `docs/results/frozen-test-full/`, `docs/results/launch/sealed_contamination_ids.json`, `docs/29-sealed-test.md` | Recomputed from the per-question rows; the constant predictor from the sealed targets; overlap counted from the id list; the read's timing and budgets quoted from docs/29 |
| 9.1, Figure 4 | Dev reads, the release temperature, calibration before and after it | `docs/results/day6/w16-06-v3dgp10-1200-s55_dev.json`, `docs/results/day7/v1/g2-17-v4t_dev.json`, the two `_calibration.json` reads beside them, `docs/results/launch/derived.json` | Temperature refitted on the same grid; smooth ECE recomputed with the kernel of `hunch/metrics.py` |
| 9.2, Table 8, Figure 5 | Held-out families: both releases, the 1.7B's parent and replication seed, the 0.6B candidates | the `_heldout2000.json` reads under `docs/results/`, `docs/results/launch/heldout-majority.json`, `docs/results/launch/derived.json` | Family metrics recomputed from the rows; the constant predictor from the targets; pools without Circa recomputed |
| 9.3, Table 9, Figure 6 | typed-decisions: both releases, the parent, the open Decider models, the card's rows; Jev 1.13 measured through its API | the `td_` reads under `docs/results/`, `docs/results/day7/v1/rescore/`, `docs/results/launch/td_card.json`, `docs/results/launch/derived.json`, `docs/results/launch/jev/jev_typed_decisions.json` and its `.items.jsonl` | Case-bootstrap intervals and paired differences recomputed (`scripts/td_bootstrap.py`); Jev's agreement recomputed from its rows; accuracy only for Jev |
| 9.3, Table 10 | Jev 1.13 and the two open Decider models, zero-shot, on the calibration-split questions of the releases' own families | `docs/results/launch/jev/jev_calibration.json`, `docs/results/launch/decider_calibration_decider-2b.json`, `docs/results/launch/decider_calibration_decider-0.8b.json`, each with its `.items.jsonl`; the two `_calibration.json` reads; `docs/results/launch/contamination_ids.json` | Accuracy, Brier and the state-group bootstrap recomputed from the rows, with and without the questions whose state occurs in training; the families where each generalist is ahead checked against the rows; the Deciders' revisions checked against their typed-decisions reads |
| 9.3 | typed-decisions-v2-system-one | the `v2so_` reads under `docs/results/` | Read |
| 9.3, Table 11 | The Laya list | `docs/results/day7/laya/`, `docs/results/day7/v1/laya/`, `docs/results/launch/laya_published/`, `docs/28-laya-list.md` | Macros over languages computed; medians and counts recomputed |
| 9.4, Table 12 | Planted instructions and formatting probes | `docs/results/day6/robust/`, `docs/results/day7/v1/robust/`, `docs/26-injection-and-jaggedness.md` | Pooled rates recomputed; the 1.7B parent's probe files are not published |
| 9.5, Table 13 | Fairness by identity mention | `docs/results/day6/fairness.json`, `docs/results/day7/v1/fairness_g2-17-v4t.json`, `docs/27-fairness.md` | Read; counts are rate times group size |
| 9.6, Table 14, Figure 7 | On-device builds against fp32 | `hunch/formats/out/equivalence-0.6b.json`, `hunch/formats/out/equivalence-1.7b.json`, `FORMATS.md` | Every gate statistic and verdict recomputed from the counts, as `hunch/formats/equivalence_test.py` computes them |
| 9.7, Table 15 | Latency; the shared-prefix engine's gates on the GPU and on the MLX builds; Jev's round trip | `docs/results/day7/v1/latency/`, `docs/results/launch/fast/`, `docs/results/launch/jev/jev_typed_decisions.json` and its `.requests.jsonl`, `docs/results/launch/derived.json` (`fast_engine`) | Gate verdicts, floor multiples, two-proportion statistics and every speed ratio recomputed from the stored counts and medians; Jev's percentiles recomputed from the request times; a ratio is quoted only where its gate passed |
| 6, Tables 3 and 4 | Training data counts and licences; overlap between evaluation and training text; personal data | `REPRODUCE.md`, `THIRD_PARTY_NOTICES.md`, `docs/results/launch/contamination_ids.json`, `docs/21-contamination-report.md`, `docs/25-pii-scan.md`; the data builder's `manifest.json` and the teacher converter's `report.json`, which the commands in REPRODUCE.md regenerate and which are not shipped | Sums checked across the three training versions; overlap counted from the id lists; sentences quoted from the documents |
| 7.1, Table 5 | Training recipes and durations | `docs/results/run_configs/released/w16-06-v3dgp10-1200-s55_config.json`, `docs/results/day7/v1/g2-17-v4t_config.json` | Read and checked against the described recipe; the parent's configuration and the two training logs are not published |
| 7.2 | Selection rules and their outcomes | `MODEL_CARD.md`, `model_cards/hunch-0.6b-preview.md` | Rule sentences quoted from the cards |
| 10.3, Table 18, Figure 10 | Training length at 0.6B, five seeds against five | the release's own reads in `docs/results/day6/`; the other nine runs' reads are not published | Rank tails computed by exact enumeration from the per-run values |
| 10.1, 10.2, 10.4, 10.5, Tables 16 and 17, Figures 8 and 9 | Candidate descriptions, prior share, model size and weight averaging, the fitted research checkpoints | not published: the per-run reads and configurations of the research record | Values transcribed from the archived per-run files and audited against them; the fitted checkpoints' intervals computed from their rows |

## Constants the report states

Rule thresholds, implementation constants and quotations, not measurements, are stated with their source in the report itself:
the fairness threshold and toxic definitions (`docs/27-fairness.md`), the contamination scan's thresholds
(`docs/21-contamination-report.md`), the typed-decisions card's reading scale (quoted in `BENCHMARKS.md`), the smooth-ECE kernel
and grid (`hunch/metrics.py`), the temperature grid (`scripts/compare_results.py`), the on-device floor rule
(`hunch/formats/equivalence_test.py`), the shared-prefix gates (`scripts/launch_derived.py`), the probability-sum check and token
budgets (`hunch/infer.py`, `FORMATS.md`), and the training constants (the configurations above and `hunch/train.py`).

## Corrections

Claims made during development or in drafts of the launch documents that did not survive their own evidence, with where each is
recorded:

- The 1.7B's held-out gain was first credited to the teacher data. It is the format transfer of the synthetic indirect-answer
  family, which reuses Circa's instruction and labels, so Circa is not held out for either release. `MODEL_CARD.md`; report
  Sections 6.2 and 9.2.
- "A better model than its parent" was withdrawn from the 1.7B card: the release is worse in family and, without Circa, held out.
  `MODEL_CARD.md`; report Section 9.2.
- An argument that Hunch's probabilities transfer to typed-decisions, because its KL was lower than that of one commercial card
  row, was withdrawn; the relevant reference is the input-ignoring prior. `BENCHMARKS.md`; report Section 9.3.
- A Laya-list regression attributed to the teacher data at 0.6B is inside seed noise: the two candidate seeds' medians differ in
  sign. `docs/results/launch/derived.json` (`laya_candidates`).
- A training-length effect on Circa seen at three against three does not hold at five against five. Report Table 18.
- The first contamination scan compared evaluation states with the base mixture only, and with their field names. The rerun over
  both releases' training text, on string values, is `docs/21-contamination-report.md`; report Section 6.6.
- The on-device gate's fixed bars were replaced by the floor rule after they had failed every conversion of earlier checkpoints,
  and the replacement is disclosed as made after that was seen. `FORMATS.md`; report Section 9.6.
- The converters embedded one temperature for every size, and nothing wrote the configuration file the PyTorch loader reads it
  from; both were fixed before upload. The project log, which is not published.
- The sealed read's first attempt stopped in batching before scoring anything. `docs/29-sealed-test.md`; report Section 8.2 and
  Appendix D.
- The shared-prefix engine's first declared gate, fast bf16 against reference bf16, failed at both sizes; the gates it passed were
  declared after that result and before they were run. `BENCHMARKS.md`; report Section 9.7.
- From the development period: a descriptions comparison that confounded data version with recipe was withdrawn; a published
  temperature for an earlier 1.7B checkpoint that no archived fit reproduces was replaced by one global fit on the calibration
  split; an earlier prior-share study at a smaller read size was superseded, and a prior-share claim that confounded the prior with
  the synthetic family was withdrawn. The research record, which is not published; report Sections 8.3, 10.1 and 10.2.
