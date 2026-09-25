# Reproduce

Everything here runs from the repository root. Training and the GPU evaluations need one NVIDIA GPU with bf16 (Ampere or
newer). A 24 GB card is enough for the 0.6B. The 1.7B's second stage fits a 32 GB card with the settings below; its first
stage used one 141 GB GPU. The MLX conversions and reads need Apple silicon. Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the build for your platform
pip install -e ".[train]"
pip install -r hunch/formats/requirements.txt                          # only for the GGUF/MLX conversions
```

The weights themselves are on Hugging Face (see the [README](README.md)). Everything below rebuilds them and reruns the
evaluations the cards report. The in-family `test` split was sealed until 2026-09-25 00:00 CEST. Before then only the
frozen-backbone baseline opened it, scoring the untrained Qwen3 models on 2026-09-20. After it, each release checkpoint was read
on it once (docs/29). The commands in section 3 that open it repeat those reads.

## 1. Data

All sources are public Hugging Face datasets with license class A or B. The builder downloads them and writes
`train/dev/calibration/test/heldout.jsonl` plus a `manifest.json` with per-source license classes, counts and SHA-256 hashes.
The held-out families (Circa, CaseHOLD, GoEmotions) are never trained on.

```bash
# the release-admitted mixture with authored candidate descriptions (docs/research/sources)
python -m hunch.data.build --out data/sprint_v3d --max-per-source 20000 --allow-license-classes A,B --generators 5000 \
  --release-only --authored docs/research/sources --no-paraphrase

# + a rule-generated indirect-answer family that uses Circa's instruction and answer labels (no Circa items, no model)
python -m hunch.data.indirect_gen prepare --base data/sprint_v3d --out data/sprint_v3dg --n 30000 --seed 7

# + 10 % prior-augmented rows (an uninformative state trained toward the family's base rate)
python -m hunch.data.prior_aug --base data/sprint_v3dg --out data/sprint_v3dgp10 --frac 0.10

# the 1.7B release's second stage adds the open Decider's teacher data (Apache-2.0), converted to this schema
git clone https://github.com/Mapika/decider .cache/decider && git -C .cache/decider checkout b44b4c9880a67291206499b86aac89004850134a
git -C .cache/decider rev-parse HEAD > .cache/decider/teacher_data/COMMIT
python scripts/build_v1_data.py teacher --src .cache/decider/teacher_data --out data/teacher_v1
mkdir -p data/v4t && cp data/sprint_v3dgp10/heldout.jsonl data/v4t/
cat data/sprint_v3dgp10/train.jsonl data/teacher_v1/train.jsonl > data/v4t/train.jsonl
cat data/sprint_v3dgp10/dev.jsonl data/teacher_v1/dev.jsonl > data/v4t/dev.jsonl
```

The builds are deterministic across machines, given the same upstream data. The adapters read each source's current revision
on the Hub and do not pin one, so a source that has changed since 2026-09-20, when these were built, changes your build.
These `train.jsonl` sha256 prefixes were confirmed on independent hosts:
`sprint_v3d` `a64fbb424f8e7774` (244,624 rows), `sprint_v3dgp10` `0d0eb01a55f62b07` (271,697 rows), `teacher_v1`
`3386cecc19f3c835` (40,612 rows), `v4t` `fec01bc2bf74c4de` (312,309 rows). The `dev`, `calibration` and `heldout` files of
the `sprint_*` builds are byte-identical across data versions. The teacher conversion drops questions the teacher answered
two different ways and removes any state that matches a typed-decisions test state or one of our held-out states; its
`report.json` counts both.

**Note on Circa.** Because `sprint_v3dg*` includes the indirect-answer family, Circa's *interface* is trained for any model
built on it, though no Circa item is. Circa scores of such models measure interface transfer, not held-out transfer. The
cards say which checkpoints this applies to.

## 2. Train

The 0.6B release (`w16-06-v3dgp10-1200-s55`, config archived at
[`docs/results/run_configs/released/w16-06-v3dgp10-1200-s55_config.json`](docs/results/run_configs/released/w16-06-v3dgp10-1200-s55_config.json)):

```bash
python -m hunch.train --data data/sprint_v3dgp10 --out runs/hunch-0.6b --model Qwen/Qwen3-0.6B \
  --amp bf16 --optimizer adamw8bit --max-steps 1200 --questions-per-step 120 --token-budget 16384 --max-path-tokens 2048 \
  --grad-ckpt --max-train-k 16 --lr 2e-5 --head-lr 3e-4 --warmup-frac 0.03 --weight-decay 0.1 --crps-weight 0.5 \
  --family-alpha 0.5 --eval-every 50 --eval-max 1000 --ckpt-every 100 --seed 55
```

The 1.7B release's first stage (`h200-17-fullk-v3d`), full candidate sets, one 141 GB GPU, about 8 hours:

```bash
python -m hunch.train --data data/sprint_v3d --out runs/hunch-1.7b-stage1 --model Qwen/Qwen3-1.7B \
  --amp bf16 --optimizer adamw --max-steps 1200 --questions-per-step 120 --token-budget 196608 --grad-ckpt \
  --max-train-k 0 --lr 2e-5 --head-lr 3e-4 --warmup-frac 0.03 --eval-every 100 --eval-max 1000 --ckpt-every 100 --seed 0
```

The 1.7B release (`g2-17-v4t`, config archived at
[`docs/results/day7/v1/g2-17-v4t_config.json`](docs/results/day7/v1/g2-17-v4t_config.json)) fine-tunes that checkpoint for
800 steps on `v4t`. The small token budget, path cap and frozen input embeddings are what let it fit a shared 32 GB card:

```bash
# weights only: --resume on a directory with train_state.json would continue that run's step count and schedule
mkdir -p runs/hunch-1.7b-init && cp runs/hunch-1.7b-stage1/best/model.safetensors runs/hunch-1.7b-init/
python -m hunch.train --data data/v4t --out runs/hunch-1.7b --model Qwen/Qwen3-1.7B --resume runs/hunch-1.7b-init \
  --amp bf16 --optimizer adamw8bit --max-steps 800 --questions-per-step 120 --token-budget 8192 --max-path-tokens 1024 \
  --freeze-embeddings --grad-ckpt --max-train-k 16 --lr 1e-5 --head-lr 3e-4 --warmup-frac 0.03 --weight-decay 0.1 \
  --crps-weight 0.5 --family-alpha 0.5 --eval-every 50 --eval-max 2000 --eval-token-budget 32768 --ckpt-every 50 --seed 0
```

On a 96 GB card use `--optimizer adamw8bit` and a smaller `--token-budget` for the first stage. The token budget sets how a
step's questions are packed into microbatches, and the trainer skips any question too large for it (it prints
`train_skipped_over_budget`), so a smaller budget can also change what is trained on. The 1.7B release's second stage
skipped none; its own log has no such line. The best checkpoint by dev NLL is kept in
`runs/<run>/best/`.

## 3. Evaluate

```bash
CK=runs/hunch-0.6b/best; M=Qwen/Qwen3-0.6B          # the 1.7B: CK=runs/hunch-1.7b/best; M=Qwen/Qwen3-1.7B
python -m hunch.evaluate --checkpoint $CK --model $M --amp bf16 --data data/sprint_v3d --split heldout     --max-per-family 2000 --out results/heldout2000.json
python -m hunch.evaluate --checkpoint $CK --model $M --amp bf16 --data data/sprint_v3d --split dev         --max-per-family 300  --out results/dev.json
python -m hunch.evaluate --checkpoint $CK --model $M --amp bf16 --data data/sprint_v3d --split calibration --max-per-family 300  --out results/calibration.json

# the release temperature: fitted on the calibration split only, written where Hunch.load reads it (without it, T = 1)
python scripts/fit_temperature.py --calibration results/calibration.json --dev results/dev.json --write-config $CK

# the in-family test split (docs/29), as each release checkpoint was read on it once, at that read's token budget; then the
# untrained backbone on the same 3,467 questions. sealed_report.py sets a read beside the shipped frozen read and the release
# temperature; sealed_contamination.py finds the questions whose state also occurs in the releases' training text
R=w16-06-v3dgp10-1200-s55; B=16384                  # the 1.7B: R=g2-17-v4t; B=32768
python -m hunch.evaluate --checkpoint $CK --model $M --amp bf16 --data data/sprint_v3d --split test --max-per-family 300 --token-budget $B --out results/test300.json
python -m hunch.baselines.frozen --model $M --dtype fp32 --data data/sprint_v3d --split test --max-per-family 300 --out results/frozen_test.json
python scripts/sealed_report.py --read $R=results/test300.json --out results/sealed.json
python scripts/sealed_contamination.py --train data/sprint_v3d/train.jsonl data/sprint_v3dgp10/train.jsonl data/teacher_v1/train.jsonl \
  --out results/sealed_contamination_ids.json

# public suites: typed-decisions (pinned revision; writes per-decision rows beside the summary) and its v2-system-one test
python scripts/bench_typed_decisions.py --checkpoint $CK --model $M --out results/td.json
python scripts/bench_td_v2_systemone.py --checkpoint $CK --model $M --out results/v2so.json
python scripts/td_bootstrap.py results/td.items.jsonl                      # case-level 95 % intervals; add a second file for a paired difference
python scripts/td_card_check.py                                            # the card's published rows, and our score of its Prior row

# the open Decider through its own interface, same scoring code, at the revisions measured here (0.8b: a0a01d6f8135298f400a8c856b355793012ae971)
python scripts/bench_td_decider.py --model Mapika/decider-2b --revision 9839cc9d908be16c5988c0d041034b5fdf82c7a2 --out results/td_decider-2b.json
python scripts/bench_td_decider.py --model Mapika/decider-2b --revision 9839cc9d908be16c5988c0d041034b5fdf82c7a2 --suite v2so --out results/v2so_decider-2b.json

# TypeSafe's Jev on the same suite, through OpenRouter (needs an OpenRouter key; about two cents): its answers and its round trip
OPENROUTER_API_KEY=... python scripts/bench_jev.py --out results/jev_typed_decisions.json

# latency: one request at a time on one GPU; the file records the GPU, dtype, token budget and torch version
python scripts/bench_latency.py --checkpoint $CK --model $M --out results/latency.json

# every public suite Laya reports on, rebuilt from Laya's code (dry run first: it tokenises and prices every suite)
python scripts/bench_laya_list.py --dry-run --out-dir results/laya
python scripts/bench_laya_list.py --checkpoint $CK --model $M --release-T 1.0 --out-dir results/laya

# injection and jaggedness probes (500 held-out questions), read against a precision floor: the checkpoint's bf16-vs-fp32
# difference, which its formats report measures (section 4). The 0.6B release's floor: max TV 7.68e-02, 28 of 6,000; the 1.7B's:
# max TV 3.08e-02, 28 of 6,000 (--floor-tv 3.08e-02 --floor-rate 0.00467).
python scripts/robustness_probes.py --checkpoint $CK --model $M --out-dir results/robust --floor-tv 7.68e-02 --floor-rate 0.00467

# fairness: every Civil Comments row of dev and calibration, then the identity-mention analysis
python -m hunch.evaluate --checkpoint $CK --model $M --amp bf16 --data data/sprint_v3d --split dev         --max-per-family 5000 --out results/run_devall.json
python -m hunch.evaluate --checkpoint $CK --model $M --amp bf16 --data data/sprint_v3d --split calibration --max-per-family 5000 --out results/run_calibrationall.json
python scripts/fairness_eval.py --preds results/run_devall.json results/run_calibrationall.json --out results/fairness.json
```

Held-out and dev reads at 2,000 and 300 questions per family are the sizes every table uses. The released 0.6B's probes were
run before its own floor was measured and used an earlier 0.6B checkpoint's (1.83e-02, 0.750 %). The 1.7B release's probes also
ran before its own floor was measured (3.08e-02, 0.467 %, section 4) and used its parent's (1.68e-02, 0.593 %). [docs/26](docs/26-injection-and-jaggedness.md) states which floor each
table is read against. Contamination and PII studies: `scripts/contamination_scan.py` and `scripts/pii_scan.py`
([docs/21](docs/21-contamination-report.md), [docs/25](docs/25-pii-scan.md)).

## 4. Formats

The same equivalence harness converts a checkpoint and scores every artifact on all 6,000 held-out questions against the
fp32 PyTorch run of the same checkpoint. `hunch/formats/common.py` pins each size's release checkpoint by sha256, the
temperature every artifact embeds, and the archived held-out read (in `docs/results/`) the harness compares against. For the
released 0.6B:

```bash
export HUNCH_BACKBONE=Qwen/Qwen3-0.6B
CK=$(python -c "from huggingface_hub import snapshot_download as s; print(s('antareslabs/hunch-0.6b-preview'))")
python -m hunch.formats.to_gguf --checkpoint $CK/model.safetensors --levels f16,Q8_0,Q4_K_M
python -m hunch.formats.to_mlx  --checkpoint $CK/model.safetensors
python -m hunch.formats.equivalence_test --reference --checkpoint $CK --device cpu --limit 0 --no-report
python -m hunch.formats.equivalence_test --gguf hunch/formats/out/hunch-0.6b-preview.f16.gguf --limit 0 --no-report   # each artifact
python -m hunch.formats.equivalence_test --mlx  hunch/formats/out/hunch-0.6b-preview-mlx-f16  --limit 0 --no-report
python -m hunch.formats.equivalence_test --report
```

The released 1.7B, f16 only. Its fp32 reference ran on CUDA, and its GGUF read on a CUDA build of llama-cpp-python
(`--tag cuda` names that read, which is the one its gate uses):

```bash
export HUNCH_BACKBONE=Qwen/Qwen3-1.7B
CK=$(python -c "from huggingface_hub import snapshot_download as s; print(s('antareslabs/hunch-1.7b-preview'))")
python -m hunch.formats.to_gguf --checkpoint $CK/model.safetensors --levels f16
python -m hunch.formats.to_mlx  --checkpoint $CK/model.safetensors --levels f16
python -m hunch.formats.equivalence_test --reference --checkpoint $CK --device cuda --limit 0 --no-report
python -m hunch.formats.equivalence_test --gguf hunch/formats/out/hunch-1.7b-preview.f16.gguf --tag cuda --limit 0 --no-report
python -m hunch.formats.equivalence_test --mlx  hunch/formats/out/hunch-1.7b-preview-mlx-f16 --token-budget 8192 --limit 0 --no-report
python -m hunch.formats.equivalence_test --report
```

Converting a checkpoint you trained yourself is a different file, so name it as one: `HUNCH_ALLOW_UNPINNED=1`,
`HUNCH_ARCHIVE=results/heldout2000.json` (its own held-out read) and `HUNCH_TEMPERATURE=<its fitted T>`. The pass/fail rule
and the floors are in [FORMATS](FORMATS.md).

## 5. Check the documents against the files

```bash
python scripts/launch_derived.py     # recomputes derived.json: fitted temperatures, pooled probe rates, intervals, Laya summaries
python scripts/verify_claims.py
```

Every decimal result the launch documents quote, and every count of changed answers in FORMATS, is listed in `claims.json`
with the file and field it comes from. The checker re-reads each one and fails on any difference. It also fails on a decimal number in those documents that no claim
produces and `claims.json` does not declare as a fixed value (a rule threshold or a quotation).

## What this repository cannot give you

- **The weights' exact bytes.** Retraining reproduces the recipe, not the same floating-point weights. The released files
  are identified by sha256 on their Hugging Face pages.
- **GPU nondeterminism.** Re-scoring the same checkpoint reproduced typed-decisions accuracy exactly and KL to 2e-4
  (an open Decider on a shared card). Expect agreement at that level, not bit for bit.
- **Backend-specific numbers.** GGUF and MLX reads depend on the llama.cpp / MLX build and the hardware. The versions are
  recorded in each report.
