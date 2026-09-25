# The Laya list

Every public suite Laya reports on, rebuilt row for row from Laya's published benchmark code (seed 13; first 300 test rows per
language; 20 options = gold + 19 sampled; their instruction strings and label rendering) and scored zero-shot with the release
checkpoints — no adaptation, no fine-tuning. Declared in the project log (not published) on 2026-09-23 17:16, before any row existed.
Laya's columns are read from their published results files, kept verbatim in `docs/results/launch/laya_published/`.
Ours at T = 1; accuracy does not depend on the temperature. Laya's as shipped with their fitted temperatures ("laya", "laya-ml" =
`laya-multilingual`). **Rows marked ⚠ are suites whose training split is in Hunch's training mixture** (Banking77, MASSIVE-en, BoolQ;
MASSIVE scenario shares utterances with the intent task): those read as retention, not generalisation, and are not comparable to
Laya's zero-shot figure on the same row. **⚠task marks the MASSIVE rows outside English: the same 60-intent task the mixture trained
on in English, in a language the readout never saw — cross-lingual transfer of a trained task, not zero-shot task transfer;
Laya's checkpoints did not train on MASSIVE at all.** Suites Laya trained on carry ⚑ from their own notes. Raw outputs: `docs/results/day7/laya/laya_list_w16-06-v3dgp10-1200-s55.json`, `docs/results/day7/v1/laya/laya_list_g2-17-v4t.json`.

- `w16-06-v3dgp10-1200-s55` (0.6B): suites scored 59 of 59; failed to build: none
- `g2-17-v4t` (1.7B): suites scored 59 of 59; failed to build: none

## MASSIVE intent, 14 languages, 20 options (random = 0.050)

| suite | n | k | Hunch 0.6B acc · ECE | Hunch 1.7B acc · ECE | laya | laya-ml | flags |
|---|---:|---:|---:|---:|---:|---:|---|
| massive_intent.en | 300 | 20 | **0.897** · 0.142 | **0.927** · 0.126 | 0.783 | 0.657 | ⚠ |
| massive_intent.de | 300 | 20 | **0.847** · 0.169 | **0.883** · 0.149 | 0.420 | 0.467 | ⚠task |
| massive_intent.fr | 300 | 20 | **0.847** · 0.159 | **0.903** · 0.156 | 0.487 | 0.553 | ⚠task |
| massive_intent.es | 300 | 20 | **0.797** · 0.138 | **0.893** · 0.152 | 0.480 | 0.503 | ⚠task |
| massive_intent.pt | 300 | 20 | **0.800** · 0.140 | **0.863** · 0.137 | 0.457 | 0.493 | ⚠task |
| massive_intent.ru | 300 | 20 | **0.853** · 0.151 | **0.890** · 0.142 | 0.293 | 0.500 | ⚠task |
| massive_intent.tr | 300 | 20 | **0.713** · 0.116 | **0.807** · 0.146 | 0.160 | 0.437 | ⚠task |
| massive_intent.ar | 300 | 20 | **0.650** · 0.107 | **0.773** · 0.177 | 0.127 | 0.383 | ⚠task |
| massive_intent.hi | 300 | 20 | **0.667** · 0.123 | **0.833** · 0.149 | 0.100 | 0.387 | ⚠task |
| massive_intent.ta | 300 | 20 | **0.347** · 0.110 | **0.737** · 0.139 | 0.113 | 0.250 | ⚠task |
| massive_intent.zh-CN | 300 | 20 | **0.920** · 0.153 | **0.923** · 0.139 | 0.593 | 0.607 | ⚠task |
| massive_intent.ja | 300 | 20 | **0.897** · 0.166 | **0.923** · 0.172 | 0.547 | 0.587 | ⚠task |
| massive_intent.ko | 300 | 20 | **0.823** · 0.147 | **0.903** · 0.185 | 0.103 | 0.490 | ⚠task |
| massive_intent.sw | 300 | 20 | **0.217** · 0.115 | **0.343** · 0.110 | 0.103 | 0.210 | ⚠task |
| **macro over 14 languages** | | | Hunch 0.6B raw **0.734** | | | | |
| **macro over 14 languages** | | | Hunch 1.7B raw **0.829** | | | | |
| **Laya macro** | | | | | **0.340** | **0.466** | |

## MASSIVE scenario, 14 languages, 20 options

| suite | n | k | Hunch 0.6B acc · ECE | Hunch 1.7B acc · ECE | laya | laya-ml | flags |
|---|---:|---:|---:|---:|---:|---:|---|
| massive_scenario.en | 300 | 20 | **0.630** · 0.139 | **0.643** · 0.153 | 0.603 | 0.560 | ⚠ |
| massive_scenario.de | 300 | 20 | **0.620** · 0.146 | **0.633** · 0.140 | 0.297 | 0.477 | ⚠task |
| massive_scenario.fr | 300 | 20 | **0.600** · 0.166 | **0.613** · 0.097 | 0.460 | 0.453 | ⚠task |
| massive_scenario.es | 300 | 20 | **0.543** · 0.098 | **0.590** · 0.125 | 0.383 | 0.447 | ⚠task |
| massive_scenario.pt | 300 | 20 | **0.533** · 0.117 | **0.603** · 0.094 | 0.420 | 0.463 | ⚠task |
| massive_scenario.ru | 300 | 20 | **0.610** · 0.153 | **0.653** · 0.154 | 0.340 | 0.473 | ⚠task |
| massive_scenario.tr | 300 | 20 | **0.500** · 0.076 | **0.587** · 0.101 | 0.227 | 0.483 | ⚠task |
| massive_scenario.ar | 300 | 20 | **0.517** · 0.082 | **0.583** · 0.105 | 0.140 | 0.397 | ⚠task |
| massive_scenario.hi | 300 | 20 | **0.520** · 0.131 | **0.610** · 0.090 | 0.097 | 0.447 | ⚠task |
| massive_scenario.ta | 300 | 20 | **0.180** · 0.144 | **0.510** · 0.105 | 0.103 | 0.337 | ⚠task |
| massive_scenario.zh-CN | 300 | 20 | **0.670** · 0.161 | **0.633** · 0.122 | 0.487 | 0.497 | ⚠task |
| massive_scenario.ja | 300 | 20 | **0.627** · 0.144 | **0.633** · 0.139 | 0.467 | 0.507 | ⚠task |
| massive_scenario.ko | 300 | 20 | **0.587** · 0.114 | **0.623** · 0.130 | 0.127 | 0.423 | ⚠task |
| massive_scenario.sw | 300 | 20 | **0.097** · 0.140 | **0.190** · 0.041 | 0.100 | 0.300 | ⚠task |
| **macro over 14 languages** | | | Hunch 0.6B raw **0.517** | | | | |
| **macro over 14 languages** | | | Hunch 1.7B raw **0.579** | | | | |
| **Laya macro** | | | | | **0.304** | **0.447** | |

## XNLI, 15 languages, 3 options

| suite | n | k | Hunch 0.6B acc · ECE | Hunch 1.7B acc · ECE | laya | laya-ml | flags |
|---|---:|---:|---:|---:|---:|---:|---|
| xnli.en | 300 | 3 | **0.613** · 0.060 | **0.747** · 0.066 | 0.860 | 0.843 |  |
| xnli.de | 300 | 3 | **0.577** · 0.049 | **0.700** · 0.079 | 0.640 | 0.793 |  |
| xnli.fr | 300 | 3 | **0.583** · 0.063 | **0.673** · 0.079 | 0.680 | 0.767 |  |
| xnli.es | 300 | 3 | **0.563** · 0.047 | **0.683** · 0.095 | 0.683 | 0.800 |  |
| xnli.ru | 300 | 3 | **0.550** · 0.044 | **0.623** · 0.063 | 0.610 | 0.740 |  |
| xnli.tr | 300 | 3 | **0.523** · 0.036 | **0.627** · 0.055 | 0.383 | 0.740 |  |
| xnli.ar | 300 | 3 | **0.517** · 0.067 | **0.633** · 0.037 | 0.440 | 0.727 |  |
| xnli.hi | 300 | 3 | **0.477** · 0.063 | **0.600** · 0.075 | 0.400 | 0.660 |  |
| xnli.ur | 300 | 3 | **0.487** · 0.077 | **0.587** · 0.067 | 0.373 | 0.683 |  |
| xnli.vi | 300 | 3 | **0.563** · 0.059 | **0.663** · 0.069 | 0.510 | 0.723 |  |
| xnli.th | 300 | 3 | **0.550** · 0.064 | **0.620** · 0.054 | 0.417 | 0.697 |  |
| xnli.el | 300 | 3 | **0.540** · 0.099 | **0.620** · 0.042 | 0.517 | 0.727 |  |
| xnli.bg | 300 | 3 | **0.537** · 0.077 | **0.663** · 0.078 | 0.543 | 0.793 |  |
| xnli.zh | 300 | 3 | **0.560** · 0.055 | **0.653** · 0.067 | 0.673 | 0.757 |  |
| xnli.sw | 300 | 3 | **0.323** · 0.202 | **0.390** · 0.088 | 0.423 | 0.627 |  |
| **macro over 15 languages** | | | Hunch 0.6B raw **0.531** | | | | |
| **macro over 15 languages** | | | Hunch 1.7B raw **0.632** | | | | |
| **Laya macro** | | | | | **0.544** | **0.738** | |

## English suites (Laya's Colab set)

| suite | n | k | Hunch 0.6B acc · ECE | Hunch 1.7B acc · ECE | laya | laya-ml | flags |
|---|---:|---:|---:|---:|---:|---:|---|
| en.sst5 | 600 |  | **0.343** · 0.067 | **0.400** · 0.114 | 0.372 | 0.282 |  |
| en.emotion | 600 |  | **0.522** · 0.063 | **0.492** · 0.078 | 0.573 | 0.513 |  |
| en.prompt_injections | 116 |  | **0.560** · 0.092 | **0.578** · 0.233 | 0.698 | 0.578 |  |
| en.banking77_full | 500 |  | **0.840** · 0.190 | **0.894** · 0.176 | — | — | ⚠ |
| en.ag_news | 600 |  | **0.782** · 0.096 | **0.838** · 0.058 | 0.947 | 0.937 | ⚑ |
| en.boolq | 600 |  | **0.803** · 0.024 | **0.842** · 0.040 | 0.830 | 0.787 | ⚠ ⚑ |

## Application themes (Laya's `bench_apps.py`, 400 cases each)

| suite | n | k | Hunch 0.6B acc · ECE | Hunch 1.7B acc · ECE | laya | laya-ml | Jev (Laya quotes, n = 100) | flags |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| jev.ag_news | 400 |  | **0.812** · 0.105 | **0.855** · 0.067 | 0.950 | 0.930 | 0.910 | ⚑ |
| jev.emotion | 400 |  | **0.515** · 0.070 | **0.500** · 0.077 | 0.595 | 0.530 | 0.480 |  |
| jev.banking77_full | 400 |  | **0.853** · 0.198 | **0.895** · 0.179 | 0.425 | 0.425 | 0.870 | ⚠ |
| app.support_triage | 400 |  | **0.395** · 0.108 | **0.407** · 0.148 | 0.502 | 0.522 | — | ⚑ |
| app.email_spam | 400 |  | **0.482** · 0.252 | **0.517** · 0.358 | 0.993 | 0.993 | — | ⚑ |
| app.phishing | 400 |  | **0.620** · 0.307 | **0.650** · 0.183 | 0.980 | 0.993 | — | ⚑ |
| app.guardrails_jailbreak | 400 |  | **0.757** · 0.134 | **0.795** · 0.057 | 0.708 | 0.755 | — |  |
| app.moderation_toxicity | 400 |  | **0.605** · 0.123 | **0.618** · 0.229 | 0.530 | 0.525 | — |  |
| app.rag_relevance | 400 |  | **0.507** · 0.140 | **0.555** · 0.176 | 0.625 | 0.657 | — | ⚑ |
| app.model_routing_domain | 399 |  | **0.511** · 0.164 | **0.714** · 0.183 | 0.639 | 0.123 | — |  |

## Option-order robustness (first 200 cases, options permuted with seed 99; flip rate of the chosen option)

Hunch's rate is zero by construction, not by luck: each candidate is scored as its own path (state + candidate) and the
order in which candidates arrive is not part of any path, so the scores cannot depend on it. Laya's readers see all
options in one sequence, which is where their non-zero flip rates come from.

| suite | Hunch 0.6B | Hunch 1.7B | laya | laya-ml |
|---|---:|---:|---:|---:|
| massive_intent.en | 0.000 | 0.000 | 0.150 | 0.230 |
| en.emotion | 0.000 | 0.000 | 0.040 | 0.090 |
| xnli.en | 0.000 | 0.000 | 0.000 | 0.015 |

## Reading

Laya's rows are what their `Router`/checkpoints produce on the same constructed cases; ours are a decision scorer that saw none of
these suites except the ⚠ rows, and saw the *task* of the ⚠task rows in English. Where a ⚠ row reads high, that is the training
split showing through; where a ⚠task row reads high, that is a trained task carried into an unseen language by the backbone — real,
and not the same thing as Laya's zero-shot number beside it. Both are said on the row.
Where a zero-shot row reads low in a language the readout never saw, that is the measurement. Metrics beyond accuracy (macro-F1,
ECE over 15 bins, Brier, NLL, score MAE) are in the JSON files for every suite.
