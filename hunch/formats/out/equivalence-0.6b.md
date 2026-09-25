# Equivalence report: hunch-0.6b-preview

Reference archive `docs/results/day6/w16-06-v3dgp10-1200-s55_heldout2000.json`: 6000 held-out questions, bf16 autocast, T = 1. fp32 anchor: `pytorch-fp32-cpu`.

**The pass rule** (FORMATS.md): an artifact passes if its max TV against the fp32 anchor is within the floor (bf16 against fp32 on the same checkpoint) and its disagreement rate is not significantly above the floor's (two-proportion z, one-sided 95 %). "pass" = significantly below the floor rate, "at floor" = indistinguishable from it, "no" = above it.

## The floor gate against `pytorch-fp32-cpu`

Floor: max TV 7.68e-02, 28 of 6000 answers differ.

| artifact | n | max TV | × floor | answers differing | × floor | z | verdict |
|---|---|---|---|---|---|---|---|
| gguf-Q4_K_M | 6000 | 8.94e-01 | 11.64 | 522 (8.700 %) | 18.64 | +21.56 | **no** |
| gguf-Q8_0 | 6000 | 1.08e-01 | 1.40 | 44 (0.733 %) | 1.57 | +1.89 | **no** |
| gguf-f16 | 6000 | 4.88e-03 | 0.06 | 1 (0.017 %) | 0.04 | -5.02 | **pass** |
| mlx-4bit | 6000 | 8.26e-01 | 10.76 | 787 (13.117 %) | 28.11 | +27.54 | **no** |
| mlx-8bit | 6000 | 1.29e-01 | 1.68 | 49 (0.817 %) | 1.75 | +2.40 | **no** |
| mlx-f16 | 6000 | 5.69e-03 | 0.07 | 0 (0.000 %) | 0.00 | -5.30 | **pass** |

The tables below also show the original fixed bars (max TV f16 ≤ 1e-3, Q8/8-bit ≤ 5e-3, Q4 ≤ 2e-2; argmax 100 % for f16/Q8, ≥ 99.5 % for Q4; smooth ECE at T = 1 and T = 0.8706 within 0.005; pooled NLL within 0.005). They are tighter than bf16 itself and are not the pass rule.

## Noise floor: the reference against itself

| artifact | n | max TV | mean TV | argmax agree (differ) | max abs ΔsmECE | pooled ΔNLL (T=1) | non-finite |
|---|---|---|---|---|---|---|---|
| pytorch-fp32-cpu vs archive | 6000 | 7.68e-02 | 3.34e-03 | 99.53 % (28) | 0.0004 | +0.0002 | 0 |

`pytorch-fp32-cpu`: max |readout(hidden) − DecisionScorer.forward| = 0.00e+00 (this package's fp32 readout against the reference's own).

The rejected GGUF design (readout norm folded into `output_norm`, dropping `backbone.norm`), scored from the reference's pre-norm hidden state on 201 questions: max TV 0.112, mean TV 0.026, argmax agreement 99.5 %.

## Artifacts against the archived bf16 reference (all scored questions)

| artifact | n | max TV | mean TV | argmax agree (differ) | max abs ΔsmECE | pooled ΔNLL (T=1) | non-finite | fixed TV bar | fixed argmax bar | smECE ≤ 0.005 | NLL ≤ 0.005 | all fixed bars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gguf-Q4_K_M | 6000 | 8.98e-01 | 5.68e-02 | 91.33 % (520) | 0.0120 | -0.0036 | 0 | **FAIL** | **FAIL** | **FAIL** | pass | no |
| gguf-Q8_0 | 6000 | 9.72e-02 | 6.06e-03 | 99.10 % (54) | 0.0020 | -0.0003 | 0 | **FAIL** | **FAIL** | pass | pass | no |
| gguf-f16 | 6000 | 7.50e-02 | 3.35e-03 | 99.52 % (29) | 0.0005 | +0.0002 | 0 | **FAIL** | **FAIL** | pass | pass | no |
| mlx-4bit | 6000 | 8.24e-01 | 8.56e-02 | 86.90 % (786) | 0.0212 | +0.0330 | 0 | **FAIL** | **FAIL** | **FAIL** | **FAIL** | no |
| mlx-8bit | 6000 | 1.59e-01 | 6.34e-03 | 99.05 % (57) | 0.0010 | +0.0014 | 0 | **FAIL** | **FAIL** | pass | pass | no |
| mlx-f16 | 6000 | 7.77e-02 | 3.36e-03 | 99.53 % (28) | 0.0005 | +0.0002 | 0 | **FAIL** | **FAIL** | pass | pass | no |

## Artifacts against the PyTorch fp32 reference (`pytorch-fp32-cpu`, common questions)

| artifact | n | max TV | mean TV | argmax agree (differ) | max abs ΔsmECE | pooled ΔNLL (T=1) | non-finite | fixed TV bar | fixed argmax bar | smECE ≤ 0.005 | NLL ≤ 0.005 | all fixed bars |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gguf-Q4_K_M | 6000 | 8.94e-01 | 5.69e-02 | 91.30 % (522) | 0.0116 | -0.0038 | 0 | **FAIL** | **FAIL** | **FAIL** | pass | no |
| gguf-Q8_0 | 6000 | 1.08e-01 | 5.19e-03 | 99.27 % (44) | 0.0020 | -0.0005 | 0 | **FAIL** | **FAIL** | pass | pass | no |
| gguf-f16 | 6000 | 4.88e-03 | 3.29e-04 | 99.98 % (1) | 0.0005 | -0.0000 | 0 | **FAIL** | **FAIL** | pass | pass | no |
| mlx-4bit | 6000 | 8.26e-01 | 8.59e-02 | 86.88 % (787) | 0.0212 | +0.0328 | 0 | **FAIL** | **FAIL** | **FAIL** | **FAIL** | no |
| mlx-8bit | 6000 | 1.29e-01 | 5.52e-03 | 99.18 % (49) | 0.0010 | +0.0011 | 0 | **FAIL** | **FAIL** | pass | pass | no |
| mlx-f16 | 6000 | 5.69e-03 | 2.78e-04 | 100.00 % (0) | 0.0001 | +0.0000 | 0 | **FAIL** | pass | pass | pass | no |

## Calibration: smooth ECE per family, T = 1 / T = 0.8706

Each cell: the run's smooth ECE on the questions it scored, with (Δ) against the archive on the same questions. smECE is sample-size dependent, so only the deltas are comparable across rows with different n.

| run (n) | emotion.goemotions | legal.casehold | pragmatics.circa | pooled |
|---|---|---|---|---|
| archive, bf16 (6000) | 0.2061 / 0.1981 | 0.0626 / 0.0272 | 0.1774 / 0.2221 | 0.1054 / 0.1301 |
| pytorch-fp32-cpu (6000) | 0.2066 (+0.0004) / 0.1986 (+0.0004) | 0.0626 (+0.0000) / 0.0272 (-0.0000) | 0.1775 (+0.0001) / 0.2221 (+0.0000) | 0.1051 (-0.0002) / 0.1299 (-0.0002) |
| gguf-Q4_K_M (6000) | 0.2170 (+0.0108) / 0.2101 (+0.0120) | 0.0661 (+0.0035) / 0.0319 (+0.0048) | 0.1776 (+0.0002) / 0.2233 (+0.0012) | 0.1093 (+0.0039) / 0.1343 (+0.0043) |
| gguf-Q8_0 (6000) | 0.2066 (+0.0005) / 0.1986 (+0.0005) | 0.0646 (+0.0020) / 0.0291 (+0.0019) | 0.1784 (+0.0010) / 0.2232 (+0.0011) | 0.1050 (-0.0003) / 0.1298 (-0.0003) |
| gguf-f16 (6000) | 0.2065 (+0.0004) / 0.1985 (+0.0004) | 0.0632 (+0.0005) / 0.0277 (+0.0005) | 0.1775 (+0.0001) / 0.2222 (+0.0001) | 0.1050 (-0.0004) / 0.1297 (-0.0004) |
| mlx-4bit (6000) | 0.2049 (-0.0012) / 0.1968 (-0.0014) | 0.0817 (+0.0191) / 0.0484 (+0.0212) | 0.1713 (-0.0061) / 0.2185 (-0.0036) | 0.0980 (-0.0073) / 0.1226 (-0.0075) |
| mlx-8bit (6000) | 0.2057 (-0.0005) / 0.1976 (-0.0005) | 0.0628 (+0.0001) / 0.0273 (+0.0001) | 0.1784 (+0.0010) / 0.2228 (+0.0007) | 0.1057 (+0.0003) / 0.1303 (+0.0002) |
| mlx-f16 (6000) | 0.2066 (+0.0005) / 0.1987 (+0.0005) | 0.0626 (-0.0000) / 0.0271 (-0.0000) | 0.1775 (+0.0001) / 0.2222 (+0.0001) | 0.1052 (-0.0002) / 0.1299 (-0.0002) |

## Localisation on the 201-question subset, against `pytorch-fp32-cpu`

| artifact | paths | max rel ‖Δh‖ (post-norm) | mean rel ‖Δh‖ | mean rel ‖Δh‖ vs PRE-norm | max abs Δz | mean abs Δz | max TV vs fp32 | regroup TV | tokenizer segments mismatched |
|---|---|---|---|---|---|---|---|---|---|
| gguf-Q4_K_M | 1005 | 2.45e-01 | 7.11e-02 | 0.94 | 1.83e+00 | 1.66e-01 | 8.94e-01 | 0.0e+00 | 0 / 1608 |
| gguf-Q8_0 | 1005 | 2.03e-02 | 7.63e-03 | 0.94 | 1.62e-01 | 1.66e-02 | 1.08e-01 | 0.0e+00 | 0 / 1608 |
| gguf-f16 | 1005 | 1.60e-03 | 3.43e-04 | 0.94 | 9.04e-03 | 8.57e-04 | 4.88e-03 | 0.0e+00 | 0 / 1608 |
| mlx-4bit | 1005 | 3.03e-01 | 1.21e-01 | 0.94 | 2.31e+00 | 2.29e-01 | 8.26e-01 | 0.0e+00 | n/a |
| mlx-8bit | 1005 | 2.70e-02 | 7.37e-03 | 0.94 | 2.34e-01 | 1.43e-02 | 1.29e-01 | 0.0e+00 | n/a |
| mlx-f16 | 1005 | 1.24e-03 | 3.70e-04 | 0.94 | 9.63e-03 | 9.18e-04 | 5.69e-03 | 0.0e+00 | n/a |

## Worst questions per artifact (against the archive)

- `gguf-Q4_K_M` `google-research-datasets/circa:1687:meaning` TV 0.8980: artifact [0.9172, 0.0342, 0.0109, 0.0056, 0.007, 0.0074, 0.0102, 0.0074] vs reference [0.0196, 0.9185, 0.0105, 0.0073, 0.012, 0.0085, 0.0136, 0.01]
- `gguf-Q4_K_M` `google-research-datasets/circa:51:meaning` TV 0.8120: artifact [0.8458, 0.1078, 0.0087, 0.0066, 0.0078, 0.0072, 0.0095, 0.0066] vs reference [0.0338, 0.8685, 0.0168, 0.0126, 0.0201, 0.0139, 0.0202, 0.0141]
- `gguf-Q8_0` `google-research-datasets/circa:1132:meaning` TV 0.0972: artifact [0.0879, 0.1529, 0.1207, 0.0795, 0.1852, 0.0757, 0.2232, 0.0749] vs reference [0.1029, 0.1037, 0.1501, 0.0828, 0.1373, 0.082, 0.2622, 0.0792]
- `gguf-Q8_0` `google-research-datasets/circa:1568:meaning` TV 0.0952: artifact [0.0706, 0.1478, 0.1059, 0.0527, 0.443, 0.0478, 0.0766, 0.0556] vs reference [0.0874, 0.1517, 0.1442, 0.0603, 0.3477, 0.0556, 0.0893, 0.0637]
- `gguf-f16` `google-research-datasets/circa:1813:meaning` TV 0.0750: artifact [0.0446, 0.4962, 0.0435, 0.0307, 0.2561, 0.0342, 0.0546, 0.0401] vs reference [0.0348, 0.5711, 0.0336, 0.0239, 0.236, 0.0266, 0.0426, 0.0313]
- `gguf-f16` `google-research-datasets/circa:1490:meaning` TV 0.0731: artifact [0.0659, 0.1164, 0.1925, 0.0461, 0.2869, 0.0451, 0.1545, 0.0926] vs reference [0.0683, 0.1002, 0.2597, 0.047, 0.2312, 0.0462, 0.1533, 0.094]
- `mlx-4bit` `google-research-datasets/circa:74:meaning` TV 0.8241: artifact [0.7407, 0.0595, 0.0374, 0.0173, 0.0286, 0.0257, 0.0533, 0.0374] vs reference [0.0353, 0.8836, 0.0152, 0.0076, 0.0127, 0.0089, 0.0218, 0.0149]
- `mlx-4bit` `google-research-datasets/circa:370:meaning` TV 0.8097: artifact [0.0157, 0.043, 0.0174, 0.0093, 0.0161, 0.0112, 0.8708, 0.0166] vs reference [0.0267, 0.7849, 0.0263, 0.016, 0.0397, 0.0181, 0.0611, 0.0271]
- `mlx-8bit` `coastalcph/lex_glue:validation:1002:holding` TV 0.1586: artifact [0.2701, 0.213, 0.0683, 0.3988, 0.0498, 0.0, 0.0, 0.0] vs reference [0.2057, 0.1529, 0.0478, 0.5573, 0.0363, 0.0, 0.0, 0.0]
- `mlx-8bit` `google-research-datasets/circa:1490:meaning` TV 0.1272: artifact [0.0624, 0.1281, 0.1534, 0.044, 0.3305, 0.0433, 0.1496, 0.0887] vs reference [0.0683, 0.1002, 0.2597, 0.047, 0.2312, 0.0462, 0.1533, 0.094]
- `mlx-f16` `google-research-datasets/circa:1490:meaning` TV 0.0777: artifact [0.0657, 0.1183, 0.1886, 0.046, 0.2896, 0.045, 0.1545, 0.0922] vs reference [0.0683, 0.1002, 0.2597, 0.047, 0.2312, 0.0462, 0.1533, 0.094]
- `mlx-f16` `google-research-datasets/circa:1813:meaning` TV 0.0771: artifact [0.0446, 0.4941, 0.0435, 0.0307, 0.2585, 0.0341, 0.0546, 0.0401] vs reference [0.0348, 0.5711, 0.0336, 0.0239, 0.236, 0.0266, 0.0426, 0.0313]

## Artifacts (sha256 recorded at conversion)

| file | format | level | bytes | sha256 |
|---|---|---|---|---|
| `hunch-0.6b-preview-mlx-4bit/model.safetensors` | mlx | 4bit (affine, group 64, 4-bit) | 335,454,107 | `0aa031828c5927a5c24acc861d586f241fca2eacfd4634cc40072fb917b1f514` |
| `hunch-0.6b-preview-mlx-8bit/model.safetensors` | mlx | 8bit (affine, group 64, 8-bit) | 633,446,639 | `fa83efde68b4333d70898a5b37933191348f05b3851949eb915a04ba321c8f90` |
| `hunch-0.6b-preview-mlx-f16/model.safetensors` | mlx | f16, float16 | 1,192,141,195 | `c1cd21e8436ea2b5c25590627dc5f4b607b54b4029038d289e9109d2e238dda7` |
| `hunch-0.6b-preview.Q4_K_M.gguf` | gguf | Q4_K_M | 396,709,344 | `58743600e9e07d3b29c25ba7e9b080a5e56de09c0aeef53f526f614b474bce10` |
| `hunch-0.6b-preview.Q8_0.gguf` | gguf | Q8_0 | 639,451,616 | `3c8492ecfdb425e8f16f2c6dc56029251f4fa379dfc98040fe7c4ad62195358a` |
| `hunch-0.6b-preview.f16.gguf` | gguf | f16 | 1,198,186,976 | `45dc23bd2e6565fb309a85b4790da761f9d878455c90297a9e3ebe50e4c9e444` |

## Run settings

| run | settings | paths | tokens | seconds |
|---|---|---|---|---|
| gguf-Q4_K_M | `{"flash_attn": false, "kv_unified": true, "llama_cpp_python": "0.3.35", "n_ctx": 16384, "n_gpu_layers": 999, "n_ubatch": 512, "type_k": 1, "type_v": 1}` | 30000 | 5109912 | 6969.6 |
| gguf-Q8_0 | `{"flash_attn": false, "kv_unified": true, "llama_cpp_python": "0.3.35", "n_ctx": 16384, "n_gpu_layers": 999, "n_ubatch": 512, "type_k": 1, "type_v": 1}` | 30000 | 5109912 | 8457.0 |
| gguf-f16 | `{"flash_attn": false, "kv_unified": true, "llama_cpp_python": "0.3.35", "n_ctx": 16384, "n_gpu_layers": 999, "n_ubatch": 512, "type_k": 1, "type_v": 1}` | 30000 | 5109912 | 4366.5 |
| mlx-4bit | `{"activations": "float32", "dtype": "float16", "mlx": "0.32.2", "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}` | 30000 | 5109912 | 1556.5 |
| mlx-8bit | `{"activations": "float32", "dtype": "float16", "mlx": "0.32.2", "quantization": {"bits": 8, "group_size": 64, "mode": "affine"}}` | 30000 | 5109912 | 1413.8 |
| mlx-f16 | `{"activations": "float32", "dtype": "float16", "mlx": "0.32.2", "quantization": null}` | 30000 | 5109912 | 887.5 |
| pytorch-fp32-cpu | `{"device": "cpu", "dtype": "fp32", "torch": "2.14.0"}` | 30000 | 5109912 | 4849.2 |
