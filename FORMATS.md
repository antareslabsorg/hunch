# On-device formats

Both releases are converted to GGUF and MLX. Converting is the easy part. The real question is whether a converted file answers
the same way as the original. We measured that on all 6,000 held-out questions, against an fp32 run of the same checkpoint: on
CPU for the 0.6B, on CUDA for the 1.7B.

## What "equivalent" means

Every conversion is compared with one fp32 run of the same checkpoint, on two statistics:

- **max TV**: the largest total-variation distance between the converted file's answer distribution and the reference's,
  over all 6,000 questions. It is a worst case, not an average.
- **disagreement rate**: how often the converted file's top answer differs from the reference's.

The bar is a measurement, not a preference. This project evaluates in bf16, and on each checkpoint bf16 alone already differs
from fp32. That difference is the floor. **A conversion passes if its max TV is within the floor and its disagreement rate is
not significantly above the floor's** (a one-sided two-proportion test at 95 %; `floor_gate` in `hunch/formats/equivalence_test.py`).
"Pass" means significantly below the floor rate. "At floor" means indistinguishable from it. "No" means above it on either
statistic.

**This is not the first gate.** The page first used fixed bars: for f16, max TV at most 1e-3 and no top answer changed. On
2026-09-23, run on earlier checkpoints, those bars failed all twelve conversions, including one that changed one answer in
6,000. Nothing could meet them against the bf16 read, not even fp32 PyTorch on the same checkpoint. After seeing that, we
replaced them with the floor rule, which is built from a quantity the report already computed. At first it allowed a 5 % margin
on the rate; the one-sided test above replaced that margin before any released checkpoint was converted. All four published f16
files fail the old bars (max TV 5.69e-03 and 4.88e-03 at 0.6B, 3.13e-03 and 1.45e-02 at 1.7B), and the reports still print
the old verdicts next to the new ones.

## Hunch 0.6B

On this checkpoint bf16 alone differs from fp32 by up to 7.68e-02 in max TV and changes 28 of 6,000 answers (0.467 %). The largest
differences are on three Circa questions, whose eight answers sit close together. Both builds were read on Apple silicon (Metal).

| format | file size | max TV | × floor | answers changed (of 6,000) | × floor | z | gate | published |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| **MLX f16** | 1,192 MB | 5.69e-03 | 0.07 | 0 (0.000 %) | 0.00 | -5.30 | pass | yes |
| **GGUF f16** | 1,198 MB | 4.88e-03 | 0.06 | 1 (0.017 %) | 0.04 | -5.02 | pass | yes |
| GGUF Q8_0 | 639 MB | 1.08e-01 | 1.40 | 44 (0.733 %) | 1.57 | +1.89 | no | no |
| MLX 8-bit | 633 MB | 1.29e-01 | 1.68 | 49 (0.817 %) | 1.75 | +2.40 | no | no |
| GGUF Q4_K_M | 397 MB | 8.94e-01 | 11.64 | 522 (8.700 %) | 18.64 | +21.56 | no | no |
| MLX 4-bit | 335 MB | 8.26e-01 | 10.76 | 787 (13.117 %) | 28.11 | +27.54 | no | no |

The quantised reads used the same batch settings as the f16 reads. The two 8-bit builds come closest, and even they are outside the
floor on both statistics (see the two × floor columns). The 4-bit builds change 8.700 % and 13.117 % of answers.

## Hunch 1.7B

On this checkpoint bf16 alone differs from fp32 by up to 3.08e-02 in max TV and changes 28 of 6,000 answers. Only f16 was converted,
because every quantised build of the 0.6B failed the gate. The MLX build was read on Apple silicon with fp32 activations. The GGUF
build was read on llama.cpp's CUDA backend, on one RTX 5090, and it is gated on that read alone, as fixed before either read.

| format | file size | max TV | × floor | answers changed (of 6,000) | × floor | z | gate | published |
|---|---:|---:|---:|---:|---:|---:|:--:|:--:|
| **MLX f16** (Metal) | 3,441 MB | 3.13e-03 | 0.10 | 0 (0.000 %) | 0.00 | -5.30 | pass | yes |
| **GGUF f16** (CUDA) | 3,447 MB | 1.45e-02 | 0.47 | 10 (0.167 %) | 0.36 | -2.92 | pass | yes |

The GGUF read changes more answers than the MLX read, 10 against 0. It is still significantly below the floor. llama.cpp's
CUDA path computes f16 matrix products differently from Metal, and an earlier 1.7B checkpoint showed the same gap between the
two backends.

## Which file to use

**On Apple silicon, use MLX f16. Elsewhere, use GGUF f16.** At both sizes both files pass: MLX f16 changes no answer, and GGUF f16
changes 1 answer in 6,000 at 0.6B (on Metal) and 10 at 1.7B (on CUDA). So choose by runtime, not by fidelity. On Apple silicon,
`hunch_mlx.load(path, fast=True)` runs the MLX f16 build with shared-prefix execution: on the 6,000 held-out questions it changes
no answer against the MLX reference at either size, and on the laptop it is 6.1× (0.6B) and 6.9× (1.7B) faster ([BENCHMARKS](BENCHMARKS.md#latency)). Do not use quantised
builds for scoring. The 0.6B table shows how far they move the answers, and none is published.

Every artifact embeds the release temperature (T = 0.8706, fitted on the calibration split), and
`hunch.formats.hunch_gguf.load` and `hunch.formats.hunch_mlx.load` apply it by default. For out-of-family use, pass
`temperature=1.0`, the setting every benchmark here uses (for the PyTorch weights, `model.T = 1.0`).

Both loaders default to `token_budget=16384` (for GGUF it is also the context size), half the PyTorch loader's 32,768. A
question needs its candidate count times its longest candidate path in padded tokens, and one above the budget makes `decide`
raise; it is never truncated. Among the 3,467 questions of the sealed test read, none needs more than 16,384 ([docs/29](docs/29-sealed-test.md)); for
longer candidate lists, pass a larger budget.

## What this does not cover

- **Backends.** The 0.6B builds were read on Metal only. The 1.7B GGUF was read on CUDA only, and the 1.7B MLX build on Metal.
  llama.cpp's other backends use different kernels, and they were not measured for these checkpoints.
- **Quantised 1.7B builds.** None was converted or measured, and none is published.

The full reports, with per-family calibration and the files' sha256, are
[`hunch/formats/out/equivalence-0.6b.json`](hunch/formats/out/equivalence-0.6b.json) (readable version:
[`equivalence-0.6b.md`](hunch/formats/out/equivalence-0.6b.md)) and
[`hunch/formats/out/equivalence-1.7b.json`](hunch/formats/out/equivalence-1.7b.json) (readable version:
[`equivalence-1.7b.md`](hunch/formats/out/equivalence-1.7b.md)). The commands are in [REPRODUCE](REPRODUCE.md).
