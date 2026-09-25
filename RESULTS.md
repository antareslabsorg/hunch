# Results: where every number comes from

The model cards and [BENCHMARKS](BENCHMARKS.md) quote numbers; this page lists the files they come from. The dev,
calibration and held-out files hold every per-question prediction; the typed-decisions reads have per-decision rows in the
listed `.items.jsonl` files; the v2-system-one files are summaries. Derived numbers, such as fitted temperatures,
pooled probe rates and intervals, are computed by `scripts/launch_derived.py` into
[`docs/results/launch/derived.json`](docs/results/launch/derived.json) from files listed here, so running it in this repository
reproduces that file. `scripts/verify_claims.py` checks every quoted decimal result against these files, and fails on any
decimal number in the README, the cards, BENCHMARKS or FORMATS that is neither claimed nor declared a fixed value. The figures in
`docs/figures/` are drawn from the same files by `scripts/render_figures.py`, with no plotting library.

## Hunch 0.6B (`w16-06-v3dgp10-1200-s55`)

- In-family dev: [`docs/results/day6/w16-06-v3dgp10-1200-s55_dev.json`](docs/results/day6/w16-06-v3dgp10-1200-s55_dev.json)
- Calibration split (temperature fit): [`docs/results/day6/w16-06-v3dgp10-1200-s55_calibration.json`](docs/results/day6/w16-06-v3dgp10-1200-s55_calibration.json)
- Sealed in-family test split, read once: [`w16-06-v3dgp10-1200-s55_test300.json`](docs/results/launch/sealed/w16-06-v3dgp10-1200-s55_test300.json)
- Held-out families, 2,000 each: [`docs/results/day6/w16-06-v3dgp10-1200-s55_heldout2000.json`](docs/results/day6/w16-06-v3dgp10-1200-s55_heldout2000.json)
- typed-decisions: [`docs/results/day6/td_zeroshot_w16-06-v3dgp10-1200-s55.json`](docs/results/day6/td_zeroshot_w16-06-v3dgp10-1200-s55.json),
  with per-decision rows from a re-score that reproduced it exactly:
  [`td_w16-06-v3dgp10-1200-s55.items.jsonl`](docs/results/day7/v1/rescore/td_w16-06-v3dgp10-1200-s55.items.jsonl);
  v2-system-one: [`docs/results/day6/v2so_w16-06-v3dgp10-1200-s55.json`](docs/results/day6/v2so_w16-06-v3dgp10-1200-s55.json)
- Laya list: [`docs/results/day7/laya/laya_list_w16-06-v3dgp10-1200-s55.json`](docs/results/day7/laya/laya_list_w16-06-v3dgp10-1200-s55.json)
- Probes: [`injection`](docs/results/day6/robust/injection_w16-06-v3dgp10-1200-s55.json),
  [`jaggedness`](docs/results/day6/robust/jaggedness_w16-06-v3dgp10-1200-s55.json)
- Fairness: [`docs/results/day6/fairness.json`](docs/results/day6/fairness.json)
- On-device formats: [`hunch/formats/out/equivalence-0.6b.json`](hunch/formats/out/equivalence-0.6b.json)
- Latency: [GPU](docs/results/day7/v1/latency/latency_w16-06-v3dgp10-1200-s55.json), [laptop](docs/results/day7/v1/latency/latency_mlx_w16-06-v3dgp10-1200-s55.json)
- The candidates it was chosen over: [`v2so_g1-06-v4t.json`](docs/results/day7/v1/v2so_g1-06-v4t.json), [`v2so_g1b-06-v4t-s1.json`](docs/results/day7/v1/v2so_g1b-06-v4t-s1.json)
- Training configuration: [`w16-06-v3dgp10-1200-s55_config.json`](docs/results/run_configs/released/w16-06-v3dgp10-1200-s55_config.json)

## Hunch 1.7B (`g2-17-v4t`)

- In-family dev: [`docs/results/day7/v1/g2-17-v4t_dev.json`](docs/results/day7/v1/g2-17-v4t_dev.json)
- Calibration split: [`docs/results/day7/v1/g2-17-v4t_calibration.json`](docs/results/day7/v1/g2-17-v4t_calibration.json)
- Sealed in-family test split, read once: [`g2-17-v4t_test300.json`](docs/results/launch/sealed/g2-17-v4t_test300.json)
- Held-out families: [`docs/results/day7/v1/g2-17-v4t_heldout2000.json`](docs/results/day7/v1/g2-17-v4t_heldout2000.json)
- typed-decisions: [`docs/results/day7/v1/td_g2-17-v4t.json`](docs/results/day7/v1/td_g2-17-v4t.json), with per-decision rows
  from a re-score that reproduced it exactly: [`td_g2-17-v4t.items.jsonl`](docs/results/day7/v1/rescore/td_g2-17-v4t.items.jsonl);
  v2-system-one: [`docs/results/day7/v1/v2so_g2-17-v4t.json`](docs/results/day7/v1/v2so_g2-17-v4t.json)
- Laya list: [`docs/results/day7/v1/laya/laya_list_g2-17-v4t.json`](docs/results/day7/v1/laya/laya_list_g2-17-v4t.json)
- Probes: [`injection`](docs/results/day7/v1/robust/injection_g2-17-v4t.json), [`jaggedness`](docs/results/day7/v1/robust/jaggedness_g2-17-v4t.json)
- Fairness: [`docs/results/day7/v1/fairness_g2-17-v4t.json`](docs/results/day7/v1/fairness_g2-17-v4t.json)
- On-device formats: [`hunch/formats/out/equivalence-1.7b.json`](hunch/formats/out/equivalence-1.7b.json)
- Latency: [GPU](docs/results/day7/v1/latency/latency_g2-17-v4t.json), [laptop](docs/results/day7/v1/latency/latency_mlx_g2-17-v4t.json)
- Training configuration: [`g2-17-v4t_config.json`](docs/results/day7/v1/g2-17-v4t_config.json)
- Its replication seed `g2b-17-v4t-s1`: [typed-decisions](docs/results/day7/v1/td_g2b-17-v4t-s1.json),
  [v2-system-one](docs/results/day7/v1/v2so_g2b-17-v4t-s1.json), [held-out](docs/results/day7/v1/g2b-17-v4t-s1_heldout2000.json),
  [Laya list](docs/results/day7/v1/laya/laya_list_g2b-17-v4t-s1.json)

Both fairness files are computed from the full dev and calibration predictions (up to 5,000 per family), which are not
included here (about 33 MB per checkpoint). `hunch.evaluate --split dev --max-per-family 5000` and the same with
`--split calibration` regenerate them.

## Comparison checkpoints and references

- `h200-17-fullk-v3d`, the earlier 1.7B checkpoint the 1.7B release was fine-tuned from (its parent):
  [dev](docs/results/h200-full/results/h200-17-fullk-v3d_dev.json), [held-out](docs/results/h200-full/results/h200-17-fullk-v3d_heldout2000.json),
  [typed-decisions](docs/results/day6/td_zeroshot_h200-17-fullk-v3d.json), [v2-system-one](docs/results/day6/v2so_h200-17-fullk-v3d.json),
  [Laya list](docs/results/day7/laya/laya_list_h200-17-fullk-v3d.json)
- The open Decider, run by us with the same scoring code: [decider-2b](docs/results/day7/v1/td_decider-2b.json)
  ([per-decision rows](docs/results/day7/v1/rescore/td_decider-2b.items.jsonl)), [decider-0.8b](docs/results/day7/v1/td_decider-0.8b.json)
  ([per-decision rows](docs/results/day7/v1/rescore/td_decider-0.8b.items.jsonl)), and on v2-system-one
  [2b](docs/results/day7/v1/v2so_decider-2b.json) / [0.8b](docs/results/day7/v1/v2so_decider-0.8b.json)
- The typed-decisions card's published rows and our reproduction of its Prior row: [`td_card.json`](docs/results/launch/td_card.json)
- TypeSafe Jev 1.13, measured by us through OpenRouter on typed-decisions: [summary](docs/results/launch/jev/jev_typed_decisions.json),
  [per-decision rows](docs/results/launch/jev/jev_typed_decisions.items.jsonl), [per-request timings](docs/results/launch/jev/jev_typed_decisions.requests.jsonl)
- Laya's published results, verbatim: [`t4_colab_benchmark.json`](docs/results/launch/laya_published/t4_colab_benchmark.json),
  [`app_benchmark_results.json`](docs/results/launch/laya_published/app_benchmark_results.json)
- The 0.6B candidate that did not replace the 0.6B release (`g1-06-v4t`, two seeds): typed-decisions
  [seed 0](docs/results/day7/v1/td_g1-06-v4t.json) / [seed 1](docs/results/day7/v1/td_g1b-06-v4t-s1.json), held-out
  [seed 0](docs/results/day7/v1/g1-06-v4t_heldout2000.json) / [seed 1](docs/results/day7/v1/g1b-06-v4t-s1_heldout2000.json), Laya
  [seed 0](docs/results/day7/v1/laya/laya_list_g1-06-v4t.json) / [seed 1](docs/results/day7/v1/laya/laya_list_g1b-06-v4t-s1.json)
- The constant predictor on the held-out families: [`heldout-majority.json`](docs/results/launch/heldout-majority.json)
- The sealed test read ([docs/29](docs/29-sealed-test.md)), both checkpoints beside the untrained backbones:
  [`sealed.json`](docs/results/launch/sealed.json), from the frozen reads
  [0.6B](docs/results/frozen-test-full/frozen_test_Qwen3-0.6B_v3d.json) / [1.7B](docs/results/frozen-test-full/frozen_test_Qwen3-1.7B_v3d.json);
  its questions whose state also occurs in training: [`sealed_contamination_ids.json`](docs/results/launch/sealed_contamination_ids.json)
