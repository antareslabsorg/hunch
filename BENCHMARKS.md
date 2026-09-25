# Public benchmarks

We scored both released checkpoints zero-shot on three public collections. Neither checkpoint trained on any split of the
first two. The third, the Laya list, includes a few suites whose training splits are in our mixture, and marks them. The short version:
on typed-decisions and on v2-system-one, both are below the open `Mapika/decider-2b`. On typed-decisions both are also below
the card's majority baseline, which ignores the input (the 1.7B within its interval), and the 0.6B is below the input-ignoring
prior as well. On the Laya list the picture is mixed: strong where the task was trained, including
in languages the readout never saw, and weak on several untrained tasks. On the 12 families Hunch was trained on, both sizes are
ahead of TypeSafe's Jev 1.13 and of both open Deciders, all answering zero-shot.

Every non-Hunch number here is either published on the suite's own card or measured by us with the same scoring code. The
measured ones are the open Decider models (Apache-2.0), run through their own interface at their shipped temperature, and
TypeSafe's Jev 1.13, queried through OpenRouter. The other commercial rows are the suite's published figures.

## typed-decisions

[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) at revision `c76749ec58bd`: four
workflows, 400 test cases, 2,000 decisions. It takes our state/question/candidate interface directly.

Its gold labels are the mean of three samples from a teacher endpoint "of roughly 4B-class capability" that the card does not
name, so a score measures agreement with that teacher, not correctness. The card's own scale: a majority baseline of 0.520, a
model fitted to the latent factors of each case at 0.704, and teacher self-agreement at 0.735. "Read 0.52 as the floor. Around
0.70 is strong. Around 0.75 is saturation."

Before trusting our harness, we scored the card's "Prior" row, which ignores the input, with our own metric code from the
training split alone. We get 0.4675 / 0.3474 / 0.1887 against the published 0.470 / 0.347 / 0.189
([`td_card.json`](docs/results/launch/td_card.json), written by `scripts/td_card_check.py`, which also reads every published row
below from the card itself).

| model | class | accuracy ↑ | KL ↓ | Brier ↓ | source |
|---|---|---:|---:|---:|---|
| meraGPT Decider 1 | general, zero-shot | 0.768 | 0.096 | 0.052 | card |
| TypeSafe Jev 1.13.0 | general, zero-shot | 0.727 | 1.442 | 0.148 | card |
| TypeSafe Jev 1.13 (`jev-1.13-20260917`) | general, zero-shot | 0.7375 | 1.5137 | 0.1482 | [measured](docs/results/launch/jev/jev_typed_decisions.json) through OpenRouter |
| ModernBERT-base (149M) | specialist, fitted per workflow | 0.646 | 0.223 | 0.119 | card |
| `Mapika/decider-2b` (open) | general, zero-shot | 0.5895 [0.5645, 0.6145] | 0.4109 | 0.2115 | [measured](docs/results/day7/v1/td_decider-2b.json) |
| MiniLM-L6 (22M) | specialist, fitted per workflow | 0.587 | 0.262 | 0.143 | card |
| **Hunch 1.7B** (`g2-17-v4t`) | general, zero-shot | **0.5105** [0.4845, 0.5365] | 0.3542 | 0.1951 | [measured](docs/results/day7/v1/td_g2-17-v4t.json) |
| `Mapika/decider-0.8b` (open) | general, zero-shot | 0.5105 [0.4870, 0.5345] | 0.6279 | 0.2888 | [measured](docs/results/day7/v1/td_decider-0.8b.json) |
| majority baseline, ignores the input | reference | 0.520 | | | card |
| prior, ignores the input | reference | 0.470 | 0.347 | 0.189 | card |
| **Hunch 0.6B** (`w16-06-v3dgp10-1200-s55`) | general, zero-shot | **0.3795** [0.3550, 0.4050] | 0.5889 | 0.2906 | [measured](docs/results/day6/td_zeroshot_w16-06-v3dgp10-1200-s55.json) |
| Hunch 1.7B parent (`h200-17-fullk-v3d`, the release's first stage) | general, zero-shot | 0.3665 | 0.8098 | 0.3581 | [measured](docs/results/day6/td_zeroshot_h200-17-fullk-v3d.json) |

Brackets are 95 % case-bootstrap intervals: the 400 test cases resampled with their five decisions each, 10,000 draws
(`scripts/td_bootstrap.py`, [`derived.json`](docs/results/launch/derived.json)).

![typed-decisions accuracy, zero-shot, with 95 % intervals: both releases and the two open Decider models](docs/figures/typed-decisions.svg)

**The 1.7B.** Its gain over its parent (0.3665 → 0.5105) comes with the open Decider's teacher data (see the
[card](MODEL_CARD.md)). That data contains no typed-decisions split. The contamination scan
([docs/21](docs/21-contamination-report.md)) finds no typed-decisions or v2-system-one test state in either release's training
text, by exact match, 13-gram Jaccard or MinHash. It ties the open 0.8B on accuracy, with better KL and Brier. It trails the open 2B by
7.9 points [5.3, 10.5] on accuracy, paired on the same 2,000 decisions, while its KL and Brier are lower than
the 2B's, with paired intervals that exclude zero. On KL it is level with the input-ignoring prior (0.3542
against 0.347), and the prior's Brier is lower still.

**The 0.6B** is below the prior on every column. On ordinal `score` questions it reads 0.280, close to the 0.244 of guessing (700 of the suite's
800 score questions have four levels and 100 have five).

## typed-decisions-v2-system-one

[`pngwn/typed-decisions-v2-system-one`](https://huggingface.co/datasets/pngwn/typed-decisions-v2-system-one): 5,214 test rows,
every one a `choice` with 2 to 10 options, T = 1. **Read it with more suspicion than the first suite.** Its card is empty: it
publishes no reference results and does not say how the gold labels were produced. The gold is a hard index, so only
accuracy, Brier and NLL are defined.

| model | accuracy ↑ | Brier ↓ | NLL ↓ | file |
|---|---:|---:|---:|---|
| `Mapika/decider-2b` (open; accuracy only) | 0.5297 | | | [`v2so_decider-2b.json`](docs/results/day7/v1/v2so_decider-2b.json) |
| `Mapika/decider-0.8b` (open; accuracy only) | 0.5021 | | | [`v2so_decider-0.8b.json`](docs/results/day7/v1/v2so_decider-0.8b.json) |
| **Hunch 1.7B** (`g2-17-v4t`) | **0.4845** | 0.6146 | 1.1836 | [`v2so_g2-17-v4t.json`](docs/results/day7/v1/v2so_g2-17-v4t.json) |
| **Hunch 0.6B** (`w16-06-v3dgp10-1200-s55`) | **0.4647** | 0.6392 | 1.2457 | [`v2so_w16-06-v3dgp10-1200-s55.json`](docs/results/day6/v2so_w16-06-v3dgp10-1200-s55.json) |
| Hunch 1.7B parent (`h200-17-fullk-v3d`) | 0.4409 | 0.6855 | 1.3171 | [`v2so_h200-17-fullk-v3d.json`](docs/results/day6/v2so_h200-17-fullk-v3d.json) |
| majority class, ignores the input | 0.3506 | | | same files |

Both release checkpoints clear the majority class and trail both open Deciders. The Decider rows are accuracy only because
their interface returns per-option probabilities that we did not re-score for Brier or NLL.

## The Laya list

Laya published its benchmark code and raw results, so its 59 suites were rebuilt row for row and scored zero-shot with both
release checkpoints, next to Laya's own published numbers: [docs/28](docs/28-laya-list.md). Some suites' training splits are in
our mixture, and the page marks them.

- **Where the task was trained, it carries into languages the readout never saw.** MASSIVE intent over 14 languages is one
  example: the task was trained in English only, and the backbone is multilingual.
- **Where the task was never trained, it mostly loses.** English XNLI reads 0.613 (0.6B) and 0.747 (1.7B) against Laya's 0.860.
  On the Boolean e-mail spam and phishing suites it reads 0.482 / 0.517 and 0.620 / 0.650, where Laya, which trained on both,
  reads 0.993 and 0.980.
- **Option order cannot move it.** Permuting the options of 200 cases flips 0.0 % of answers on all three suites Laya tests,
  for both checkpoints. That holds by construction: each candidate is scored as its own path, and the order is not part of any
  path.

## On Hunch's own task families, next to Jev and the open Deciders

The suites above are out of family for Hunch. To see the other side, we sent the 3,523 calibration-split questions that both
release reads scored, from the 12 families Hunch was trained on, to TypeSafe's Jev 1.13 and to the two open Decider models,
each through its own interface. That split only fitted Hunch's temperature,
which cannot move an answer, and no checkpoint was chosen on it. Each question went as one request carrying what Hunch reads: the
state, the instruction, the candidates with their descriptions, a boolean's two definitions, a score's levels. Hunch was trained
on these families and the other three answer zero-shot, so this compares specialists with generalists on the specialists' own
tasks. The Deciders' public data registry has none of these task formats, though some of their public sources (Banking77,
CLINC150, MASSIVE, BoolQ and others) may be in their training data; we did not test that.

| model | class | accuracy ↑ | Brier ↓ | NLL ↓ |
|---|---|---:|---:|---:|
| **Hunch 1.7B** (`g2-17-v4t`) | trained on these families | **0.8731** | 0.1474 | 0.3866 |
| **Hunch 0.6B** (`w16-06-v3dgp10-1200-s55`) | trained on these families | **0.8535** | 0.1681 | 0.4338 |
| TypeSafe Jev 1.13 (`jev-1.13-20260917`) | general, zero-shot | 0.8010 | 0.2734 | 1.2894 |
| `Mapika/decider-2b` (open) | general, zero-shot | 0.7346 | 0.3087 | 0.7151 |
| `Mapika/decider-0.8b` (open) | general, zero-shot | 0.6787 | 0.3690 | 0.8339 |

Paired on the same questions, Hunch 1.7B leads Jev by 7.2 points [5.7, 8.8] and Hunch 0.6B by 5.3 [3.7, 6.9] (95 % intervals,
state groups resampled, 10,000 draws). Without the 204 questions whose state occurs exactly in the training text
([docs/21](docs/21-contamination-report.md)), the leads are 7.4 [5.8, 9.0] and 5.4 [3.7, 7.1]. Jev is ahead of both sizes on
two families, yes/no reading questions (BoolQ: 0.953 against 0.853 for the 1.7B) and urn draws (0.970 against 0.933), and ahead
of the 0.6B on fact verification (VitaminC: 0.833 against 0.803). No request failed; the run cost $0.15. Per-family rows:
[`jev_calibration.json`](docs/results/launch/jev/jev_calibration.json), per question
[`jev_calibration.items.jsonl`](docs/results/launch/jev/jev_calibration.items.jsonl), written by `scripts/bench_jev_infamily.py`.

The open Deciders lead Hunch on both typed-decisions suites; here Hunch 1.7B leads the 2B by 13.9 points [10.9, 17.0] and the
0.8B by 19.4 [16.2, 23.0], and Hunch 0.6B leads them by 11.9 [8.9, 15.2] and 17.5 [14.1, 21.2]. Without the 204 overlap questions
the four leads are 13.6 [10.5, 17.0], 19.2 [15.7, 23.0], 11.7 [8.5, 15.1] and 17.3 [13.7, 21.2]. The 2B is ahead of Hunch 1.7B on
two families, helpfulness ratings (HelpSteer2: 0.487 against 0.397) and BoolQ (0.863 against 0.853); the 0.8B on none. Both
answered every question, on one RTX 5090, at the revisions of their typed-decisions reads. Files:
[`decider-2b`](docs/results/launch/decider_calibration_decider-2b.json) ([per question](docs/results/launch/decider_calibration_decider-2b.items.jsonl)),
[`decider-0.8b`](docs/results/launch/decider_calibration_decider-0.8b.json) ([per question](docs/results/launch/decider_calibration_decider-0.8b.items.jsonl)),
written by `scripts/bench_decider_infamily.py`.

## Latency

This section times one request at a time under the conditions stated below. It is not a throughput figure. Its one comparison
with another model, TypeSafe's Jev at the end, sets a local run beside a network API and says so.

**Conditions**
- Hardware and software: one NVIDIA GeForce RTX 5090, bf16, token budget 16,384, PyTorch 2.11.0+cu128.
- Calls: `Hunch.decide` receives one request at a time.
- Requests: the 400 typed-decisions test states with their questions. Gold labels are not read.
- Timing: the first 20 requests are warm-up. The remaining 380 are timed as wall-clock, including tokenisation.

| checkpoint | per request, median | per request, 90th percentile | per decision, median | per decision, 90th percentile |
|---|---:|---:|---:|---:|
| Hunch 0.6B | 78.1 ms | 107.4 ms | 15.6 ms | 21.5 ms |
| Hunch 1.7B | 140.6 ms | 188.3 ms | 28.1 ms | 37.7 ms |

"Per decision" is a request's time divided by its number of questions. A request's questions are scored together in one batched
pass, so this figure is an average share of a request, not the time a single-question request would take. Files:
[0.6B](docs/results/day7/v1/latency/latency_w16-06-v3dgp10-1200-s55.json), [1.7B](docs/results/day7/v1/latency/latency_g2-17-v4t.json),
written by `scripts/bench_latency.py`.

**On a laptop.** The same protocol with the MLX f16 builds, run on one Apple M4 Pro with MLX 0.32.2. Activations were fp32,
the runtime's default and the setting the equivalence gate used. The laptop was in normal use: its 1-minute load average was
2.16 and 2.57 at the start of the two reference runs, and 4.19 and 3.81 at the start of the two fast ones. The fast engine on the
same builds (`hunch_mlx.load(path, fast=True)`) changes no answer against the MLX reference read on the 6,000 held-out questions,
at either size (largest TV 1.49e-05 at 0.6B, 1.20e-05 at 1.7B), the gate declared before it was timed.

| checkpoint | engine | per request, median | per request, 90th percentile | per decision, median | per decision, 90th percentile |
|---|---|---:|---:|---:|---:|
| Hunch 0.6B, MLX f16 | reference | 950.8 ms | 1309.8 ms | 190.2 ms | 262.0 ms |
| Hunch 0.6B, MLX f16 | fast | 156.2 ms | 185.6 ms | 31.2 ms | 37.1 ms |
| Hunch 1.7B, MLX f16 | reference | 2746.4 ms | 3729.0 ms | 549.3 ms | 745.8 ms |
| Hunch 1.7B, MLX f16 | fast | 400.7 ms | 459.1 ms | 80.1 ms | 91.8 ms |

At the median the fast engine is 6.1× (0.6B) and 6.9× (1.7B) faster, although its runs started on a busier laptop. The reference
rows were timed on 24 September (Python 3.12.7), the fast rows on 25 September in the fresh-install environment (Python 3.13.9),
both on MLX 0.32.2. Files: reference [0.6B](docs/results/day7/v1/latency/latency_mlx_w16-06-v3dgp10-1200-s55.json),
[1.7B](docs/results/day7/v1/latency/latency_mlx_g2-17-v4t.json) (`scripts/bench_latency.py --mlx`); fast
[0.6B](docs/results/launch/fast/latency_mlx_fast_w16-06-v3dgp10-1200-s55.json), [1.7B](docs/results/launch/fast/latency_mlx_fast_g2-17-v4t.json)
(`--mlx --fast`); its gate [0.6B](docs/results/launch/fast/mlx_equiv_w16-06-v3dgp10-1200-s55.json),
[1.7B](docs/results/launch/fast/mlx_equiv_g2-17-v4t.json) (`scripts/check_fast_mlx.py`).

**A faster engine.** `hunch/fast.py` (`FastHunch`, the same arguments and output as `Hunch`) computes the same answers with less
work. The reference scores every candidate as its own path, so a request with one state and twenty candidates reads the state twenty
times. The paths share their prefixes token for token, so the fast engine packs a state's paths into one sequence, the state once,
each question once, each candidate once, under an attention mask that gives every token exactly the context and the position it
has in its own path. It was judged against the fp32 reference on the 6,000 held-out questions and the 2,000 typed-decisions
decisions:

- **In fp32 it is exact.** At both sizes it changes no answer on either set. Its largest total-variation distance from the
  reference on the held-out questions is 1.67e-05 (0.6B) and 6.11e-06 (1.7B).
- **In bf16 it is within the floor at 0.6B, not at 1.7B.** At 0.6B it changes 33 held-out answers, not significantly more than bf16
  alone (28; z = 0.64), and its largest TV is 0.88 × the floor's. At 1.7B its largest TV is 1.25 × the floor's, which fails the
  on-device builds' rule ([FORMATS.md](FORMATS.md)), so run the 1.7B in fp32.

A first gate, fast bf16 against reference bf16, failed at both sizes (33 and 35 answers changed, against 28). It stacks two bf16
roundings against a floor that measures one. The two gates above were declared after that result, and before they were run.

Latency, same protocol and card, the reference and the fast engine timed back to back:

| checkpoint | engine | per request, median | per request, 90th percentile | faster than the reference |
|---|---|---:|---:|---:|
| Hunch 0.6B | reference, bf16 | 75.6 ms | 105.0 ms | |
| Hunch 0.6B | fast, fp32 (exact) | 36.7 ms | 39.4 ms | 2.1× |
| Hunch 0.6B | fast, bf16 | 21.0 ms | 22.3 ms | 3.6× |
| Hunch 1.7B | reference, bf16 | 139.8 ms | 184.0 ms | |
| Hunch 1.7B | fast, fp32 (exact) | 66.6 ms | 71.0 ms | 2.1× |

The 1.7B's fast bf16 median (37.6 ms) is in no comparison, because that configuration failed its gate. The two reference medians are
a second measurement beside the table above. Files: fp32 exactness
[0.6B](docs/results/launch/fast/equivA_w16-06-v3dgp10-1200-s55.json), [1.7B](docs/results/launch/fast/equivA_g2-17-v4t.json);
bf16 against fp32 [0.6B](docs/results/launch/fast/equivB_w16-06-v3dgp10-1200-s55.json), [1.7B](docs/results/launch/fast/equivB_g2-17-v4t.json);
the first gate [0.6B](docs/results/launch/fast/equivalence_w16-06-v3dgp10-1200-s55.json), [1.7B](docs/results/launch/fast/equivalence_g2-17-v4t.json);
latency of the reference [0.6B](docs/results/launch/fast/latency_ref_w16-06-v3dgp10-1200-s55.json), [1.7B](docs/results/launch/fast/latency_ref_g2-17-v4t.json),
of fast fp32 [0.6B](docs/results/launch/fast/latency_fast_fp32_w16-06-v3dgp10-1200-s55.json), [1.7B](docs/results/launch/fast/latency_fast_fp32_g2-17-v4t.json)
and of fast bf16 [0.6B](docs/results/launch/fast/latency_fast_w16-06-v3dgp10-1200-s55.json), [1.7B](docs/results/launch/fast/latency_fast_g2-17-v4t.json).
They are written by `scripts/check_fast.py` and `scripts/bench_latency.py --fast`; the gates are applied in
[`derived.json`](docs/results/launch/derived.json) (`fast_engine`).

**Jev, through its API.** TypeSafe's Jev 1.13 is served over the network, so its latency includes the network. We timed it on the
same 400 requests, one at a time, through OpenRouter (`typesafe/jev-1.13`, served as `jev-1.13-20260917`), as the round trip from
our laptop over the internet: 380 timed after 20 warm-up, a median of 369.1 ms and a 90th percentile of 458.5 ms. TypeSafe's launch post
states 70-500 ms end to end. Hunch runs locally, with no network in the way. With the fast engine on one RTX 5090, Hunch 0.6B's median is
36.7 ms in fp32, 10.1× lower than Jev's round trip, and 21.0 ms in bf16, 17.6× lower; Hunch 1.7B's is 66.6 ms in fp32, 5.5×
lower. These ratios set a local GPU beside a network API measured from our laptop, and we only state them that way. On that
laptop, with the fast engine, Hunch 0.6B's median (156.2 ms) is 2.4× lower than Jev's round trip from the same machine, and Hunch
1.7B's (400.7 ms) is higher. Jev's agreement on the same decisions is higher (the table above). File:
[`jev_typed_decisions.json`](docs/results/launch/jev/jev_typed_decisions.json), written by `scripts/bench_jev.py`.

## What we have not run, and why

Eight public datasets carry this category's name. We ran two of them, plus the Laya list. Of the other six:

- `pngwn/typed-decisions-causal-experiment` and `emretheus/jev-rag-benchmark` are results dumps, not suites with a test split.
- `Luni/laya-jev-benchmark` lists no files.
- `com-kotobalabs/typed-decisions-repo-governance` has no gold labels, so it cannot be scored.
- `com-kotobalabs/typed-decisions-code-holes` cannot be scored by this model class. Its states are whole code files with
  hundreds of options each, and each candidate is scored as its own path, so a single row costs millions of tokens; many
  states exceed the scorer's 32,768-token cap. That is a limit of the architecture, stated as one.
- `pngwn/typed-decisions-v2` is the one that looks runnable. It has no adapter yet.

Reproduction commands are in [REPRODUCE](REPRODUCE.md).
