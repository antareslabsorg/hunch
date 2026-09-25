# The sealed in-family test read

This read was declared in advance, on 2026-09-24 at 11:10. The in-family `test` split was read once per release checkpoint,
after 2026-09-25 00:00 CEST, with `hunch.evaluate --data data/sprint_v3d --split test --max-per-family 300 --amp bf16` at T = 1
(the commands are in [REPRODUCE](../REPRODUCE.md), section 3). Both checkpoints were fixed before the read, and nothing is
selected on it. On the box, before the read, their weight files matched the SHA-256 the on-device builds are converted from
(`hunch/formats/common.py`): 0.6B `75d678f36ce35cfcb8fc85f96a719bad77638af5b21b3ebabeea83e360ccf2a6`, 1.7B
`e2fa7f0ab23e3f1d1eadf0e630ba2b453e41c4bd13de5febca45cb4d9a572d66`. It was read at 2026-09-25 00:04 (0.6B) and 2026-09-25 00:06 (1.7B) CEST, on one RTX 5090.

**A first attempt, at 00:00:54, stopped before scoring anything.** The card was shared, and with less than 16 GB free the
script chose its fallback batching of 8,192 tokens. The evaluator refuses a question larger than the budget, and 311 of these
questions need more than 8,192 padded tokens (the largest 15,855; none more than 16,384). So both reads stopped while batching,
at the first such question (12,684 tokens). They wrote nothing and produced no number. The read was declared again, with the
fallback raised from 8,192 to 16,384 tokens and PyTorch's allocator set to expandable segments against fragmentation on the
shared card, and re-run from 00:04. The 0.6B read used the 16,384-token fallback. By the time of
the 1.7B read the card was free, so it used the full 32,768. The budget sets how questions are packed into batches; the
questions and the checkpoints are the same.

Taking 300 per family picks 3,467 questions in 12 families. They are the same questions the frozen-backbone baseline read on
2026-09-20. That baseline is the untrained Qwen3 instruct model of the same size, in fp32 (`hunch/baselines/frozen.py`). When a
question has at most 26 options, they are listed as letters and the letters' next-token probabilities are its answer; with
more options, each option is scored as a continuation (mean token log-probability, softmax over options). That baseline is the
only thing that had opened this split. No trained checkpoint had been scored on it. Its reads are
[`frozen_test_Qwen3-0.6B_v3d.json`](results/frozen-test-full/frozen_test_Qwen3-0.6B_v3d.json) and
[`frozen_test_Qwen3-1.7B_v3d.json`](results/frozen-test-full/frozen_test_Qwen3-1.7B_v3d.json).

Accuracy does not depend on the temperature. NLL and smooth ECE are given at T = 1 and at the release temperature, which was
fitted on the calibration split. The numbers are in [`sealed.json`](results/launch/sealed.json), written by
`scripts/sealed_report.py` from the two reads in [`results/launch/sealed/`](results/launch/sealed/).

**205 of the 3,467 questions have a state that also occurs in the releases' training text**, compared exactly as docs/21
compares dev, calibration and held-out, against the same text: `sprint_v3d`, `sprint_v3dgp10` and the teacher data, which the
1.7B trained on with its parent. The 0.6B trained on `sprint_v3dgp10` alone, which holds 204 of them. The ids are in
[`sealed_contamination_ids.json`](results/launch/sealed_contamination_ids.json), from `scripts/sealed_contamination.py`.
Without the 205, accuracy is 0.8455 for the 0.6B and 0.8679 for the 1.7B ([`derived.json`](results/launch/derived.json)).

## Hunch 0.6B (`w16-06-v3dgp10-1200-s55`)

| family | n | accuracy | frozen accuracy | NLL, T = 1 | NLL, release T | smooth ECE, T = 1 | smooth ECE, release T |
|---|---:|---:|---:|---:|---:|---:|---:|
| **pooled** | 3467 | 0.8543 | 0.3303 | 0.4415 | 0.4373 | 0.0162 | 0.0167 |
| mechanism.urn | 300 | 0.9300 | 0.2000 | 0.8536 | 0.8542 | 0.0070 | 0.0079 |
| moderation.civil_comments | 300 | 0.9867 | 0.3867 | 0.0764 | 0.0790 | 0.0072 | 0.0120 |
| moderation.score.civil_toxicity_level | 300 | 0.8067 | 0.1300 | 0.4337 | 0.4348 | 0.0404 | 0.0362 |
| quality.score.helpsteer2 | 300 | 0.4467 | 0.2733 | 1.2684 | 1.2730 | 0.0231 | 0.0388 |
| rules.refund_policy | 167 | 0.9820 | 0.3353 | 0.0686 | 0.0534 | 0.0424 | 0.0274 |
| support.routing.bitext | 300 | 1.0000 | 0.4800 | 0.0591 | 0.0281 | 0.0567 | 0.0274 |
| support.routing.intent.banking77 | 300 | 0.9133 | 0.1333 | 0.3686 | 0.3588 | 0.0532 | 0.0423 |
| support.routing.intent.clinc150 | 300 | 0.9600 | 0.2400 | 0.1636 | 0.1502 | 0.0419 | 0.0237 |
| support.routing.intent.massive | 300 | 0.9100 | 0.0967 | 0.4128 | 0.3926 | 0.0612 | 0.0302 |
| verification.boolq | 300 | 0.8167 | 0.6733 | 0.4161 | 0.4285 | 0.0409 | 0.0613 |
| verification.fact.vitaminc | 300 | 0.6200 | 0.4500 | 0.8223 | 0.8386 | 0.0756 | 0.1050 |
| verification.paraphrase.paws | 300 | 0.9367 | 0.5667 | 0.1889 | 0.1861 | 0.0357 | 0.0342 |

Release temperature T = 0.8706, from [`derived.json`](results/launch/derived.json).

## Hunch 1.7B (`g2-17-v4t`)

| family | n | accuracy | frozen accuracy | NLL, T = 1 | NLL, release T | smooth ECE, T = 1 | smooth ECE, release T |
|---|---:|---:|---:|---:|---:|---:|---:|
| **pooled** | 3467 | 0.8751 | 0.3337 | 0.3987 | 0.3922 | 0.0253 | 0.0136 |
| mechanism.urn | 300 | 0.9367 | 0.5733 | 0.8544 | 0.8567 | 0.0137 | 0.0192 |
| moderation.civil_comments | 300 | 0.9933 | 0.5900 | 0.0731 | 0.0747 | 0.0046 | 0.0084 |
| moderation.score.civil_toxicity_level | 300 | 0.8167 | 0.1300 | 0.4402 | 0.4441 | 0.0274 | 0.0302 |
| quality.score.helpsteer2 | 300 | 0.4300 | 0.3133 | 1.2510 | 1.2663 | 0.0411 | 0.0665 |
| rules.refund_policy | 167 | 1.0000 | 0.2635 | 0.0434 | 0.0290 | 0.0390 | 0.0259 |
| support.routing.bitext | 300 | 1.0000 | 0.2467 | 0.0311 | 0.0143 | 0.0297 | 0.0138 |
| support.routing.intent.banking77 | 300 | 0.9067 | 0.0267 | 0.3313 | 0.3113 | 0.0567 | 0.0339 |
| support.routing.intent.clinc150 | 300 | 0.9767 | 0.0367 | 0.1362 | 0.1090 | 0.0536 | 0.0263 |
| support.routing.intent.massive | 300 | 0.9433 | 0.0133 | 0.2971 | 0.2732 | 0.0717 | 0.0375 |
| verification.boolq | 300 | 0.8667 | 0.7567 | 0.3191 | 0.3244 | 0.0295 | 0.0347 |
| verification.fact.vitaminc | 300 | 0.7300 | 0.4500 | 0.7086 | 0.7048 | 0.0517 | 0.0504 |
| verification.paraphrase.paws | 300 | 0.9567 | 0.5733 | 0.1418 | 0.1380 | 0.0259 | 0.0145 |

Release temperature T = 0.8706, from [`derived.json`](results/launch/derived.json).

