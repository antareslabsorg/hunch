# Equivalence report: hunch-1.7b-preview

Reference archive `docs/results/day7/v1/g2-17-v4t_heldout2000.json`: 6000 held-out questions, bf16 autocast, T = 1. fp32 anchor: `pytorch-fp32-cuda`.

**The pass rule** (FORMATS.md): an artifact passes if its max TV against the fp32 anchor is within the floor (bf16 against fp32 on the same checkpoint) and its disagreement rate is not significantly above the floor's (two-proportion z, one-sided 95 %). "pass" = significantly below the floor rate, "at floor" = indistinguishable from it, "no" = above it.

## The floor gate against `pytorch-fp32-cuda`

Floor: max TV 3.08e-02, 28 of 6000 answers differ.

| artifact | n | max TV | × floor | answers differing | × floor | z | verdict |
|---|---|---|---|---|---|---|---|
| gguf-f16+cuda | 6000 | 1.45e-02 | 0.47 | 10 (0.167 %) | 0.36 | -2.92 | **pass** |
| mlx-f16 | 6000 | 3.13e-03 | 0.10 | 0 (0.000 %) | 0.00 | -5.30 | **pass** |

The tables below also show the original fixed bars (max TV f16 ≤ 1e-3, Q8/8-bit ≤ 5e-3, Q4 ≤ 2e-2; argmax 100 % for f16/Q8, ≥ 99.5 % for Q4; smooth ECE at T = 1 and T = 0.8706 within 0.005; pooled NLL within 0.005). They are tighter than bf16 itself and are not the pass rule.

## Noise floor: the reference against itself

| artifact | n | max TV | mean TV | argmax agree (differ) | max abs ΔsmECE | pooled ΔNLL (T=1) | non-finite |
|---|---|---|---|---|---|---|---|
| pytorch-fp32-cuda vs archive | 6000 | 3.08e-02 | 2.92e-03 | 99.53 % (28) | 0.0022 | +0.0006 | 0 |

`pytorch-fp32-cuda`: max |readout(hidden) − DecisionScorer.forward| = 1.67e-06 (this package's fp32 readout against the reference's own).

The rejected GGUF design (readout norm folded into `output_norm`, dropping `backbone.norm`), scored from the reference's pre-norm hidden state on 201 questions: max TV 0.081, mean TV 0.034, argmax agreement 97.5 %.

## Artifacts against the archived bf16 reference (all scored questions)

| artifact | n | max TV | mean TV | argmax agree (differ) | max abs ΔsmECE | pooled ΔNLL (T=1) | non-finite | fixed TV bar | fixed argmax bar | smECE ≤ 0.005 | NLL ≤ 0.005 | all fixed bars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gguf-f16+cuda | 6000 | 3.27e-02 | 3.20e-03 | 99.50 % (30) | 0.0026 | +0.0005 | 0 | **FAIL** | **FAIL** | pass | pass | no |
| mlx-f16 | 6000 | 3.04e-02 | 2.93e-03 | 99.53 % (28) | 0.0021 | +0.0006 | 0 | **FAIL** | **FAIL** | pass | pass | no |

## Artifacts against the PyTorch fp32 reference (`pytorch-fp32-cuda`, common questions)

| artifact | n | max TV | mean TV | argmax agree (differ) | max abs ΔsmECE | pooled ΔNLL (T=1) | non-finite | fixed TV bar | fixed argmax bar | smECE ≤ 0.005 | NLL ≤ 0.005 | all fixed bars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gguf-f16+cuda | 6000 | 1.45e-02 | 1.31e-03 | 99.83 % (10) | 0.0011 | -0.0000 | 0 | **FAIL** | **FAIL** | pass | pass | no |
| mlx-f16 | 6000 | 3.13e-03 | 2.27e-04 | 100.00 % (0) | 0.0001 | +0.0000 | 0 | **FAIL** | pass | pass | pass | no |

## Calibration: smooth ECE per family, T = 1 / T = 0.8706

Each cell: the run's smooth ECE on the questions it scored, with (Δ) against the archive on the same questions. smECE is sample-size dependent, so only the deltas are comparable across rows with different n.

| run (n) | emotion.goemotions | legal.casehold | pragmatics.circa | pooled |
|---|---|---|---|---|
| archive, bf16 (6000) | 0.0347 / 0.0448 | 0.1185 / 0.0838 | 0.0667 / 0.1151 | 0.0460 / 0.0463 |
| pytorch-fp32-cuda (6000) | 0.0347 (+0.0001) / 0.0449 (+0.0002) | 0.1163 (-0.0022) / 0.0823 (-0.0015) | 0.0682 (+0.0016) / 0.1166 (+0.0015) | 0.0449 (-0.0011) / 0.0454 (-0.0009) |
| gguf-f16+cuda (6000) | 0.0348 (+0.0001) / 0.0450 (+0.0002) | 0.1170 (-0.0015) / 0.0824 (-0.0013) | 0.0691 (+0.0024) / 0.1177 (+0.0026) | 0.0447 (-0.0013) / 0.0453 (-0.0011) |
| mlx-f16 (6000) | 0.0347 (+0.0000) / 0.0448 (+0.0001) | 0.1164 (-0.0021) / 0.0824 (-0.0014) | 0.0682 (+0.0015) / 0.1166 (+0.0015) | 0.0449 (-0.0011) / 0.0454 (-0.0009) |

## Localisation on the 201-question subset, against `pytorch-fp32-cuda`

| artifact | paths | max rel ‖Δh‖ (post-norm) | mean rel ‖Δh‖ | mean rel ‖Δh‖ vs PRE-norm | max abs Δz | mean abs Δz | max TV vs fp32 | regroup TV | tokenizer segments mismatched |
|---|---|---|---|---|---|---|---|---|---|
| gguf-f16+cuda | 1005 | 4.39e-03 | 1.30e-03 | 0.99 | 6.60e-02 | 3.80e-03 | 1.45e-02 | 1.2e-07 | 0 / 1608 |
| mlx-f16 | 1005 | 5.41e-04 | 2.61e-04 | 0.99 | 5.77e-03 | 7.21e-04 | 3.13e-03 | 0.0e+00 | n/a |

## Worst questions per artifact (against the archive)

- `gguf-f16+cuda` `google-research-datasets/circa:1103:meaning` TV 0.0327: artifact [0.3314, 0.0278, 0.5496, 0.0186, 0.018, 0.0117, 0.0322, 0.0107] vs reference [0.3019, 0.027, 0.5823, 0.0184, 0.0173, 0.0113, 0.0313, 0.0104]
- `gguf-f16+cuda` `google-research-datasets/circa:298:meaning` TV 0.0313: artifact [0.7343, 0.0772, 0.0522, 0.0259, 0.0275, 0.0197, 0.0491, 0.0139] vs reference [0.703, 0.0939, 0.0565, 0.0277, 0.0295, 0.0211, 0.0533, 0.0151]
- `mlx-f16` `google-research-datasets/circa:1417:meaning` TV 0.0304: artifact [0.2584, 0.1802, 0.3976, 0.0137, 0.0928, 0.0137, 0.0337, 0.01] vs reference [0.2294, 0.1942, 0.4087, 0.0132, 0.0981, 0.0131, 0.0334, 0.0098]
- `mlx-f16` `google-research-datasets/circa:1103:meaning` TV 0.0301: artifact [0.3302, 0.0273, 0.5522, 0.0185, 0.0178, 0.0116, 0.0318, 0.0106] vs reference [0.3019, 0.027, 0.5823, 0.0184, 0.0173, 0.0113, 0.0313, 0.0104]

## Artifacts (sha256 recorded at conversion)

| file | format | level | bytes | sha256 |
|---|---|---|---|---|
| `hunch-1.7b-preview-mlx-f16/model.safetensors` | mlx | f16, float16 | 3,441,199,823 | `affde731fb5083d2cfdf515cfaa0e4881468ce092b357e0d801a322f89ac687f` |
| `hunch-1.7b-preview.f16.gguf` | gguf | f16 | 3,447,362,016 | `a2b4011fc0f7ad6ccc6113abdc486be38b838e4e7996b4de2ff34233782c6f0a` |

## Run settings

| run | settings | paths | tokens | seconds |
|---|---|---|---|---|
| gguf-f16+cuda | `{"flash_attn": false, "kv_unified": true, "llama_cpp_python": "0.3.35", "n_ctx": 16384, "n_gpu_layers": 999, "n_ubatch": 512, "type_k": 1, "type_v": 1}` | 30000 | 5109912 | 617.3 |
| mlx-f16 | `{"activations": "float32", "dtype": "float16", "mlx": "0.32.2", "quantization": null}` | 30000 | 5109912 | 2549.2 |
| pytorch-fp32-cuda | `{"device": "cuda", "dtype": "fp32", "torch": "2.11.0+cu128"}` | 30000 | 5109912 | 330.2 |
