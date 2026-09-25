# Hunch routing a message as it is typed

A customer message appears word by word. After every word, one `decide()` call scores every candidate of the question
again: all 77 Banking77 intents for the first message, then all 60 MASSIVE intents for the second, each with the
description the training data gives it. The panel shows the six most probable, the time the call took and, once the
message is complete, the label the record carries.

The launch video and GIF are a rendered replay of one recorded run of [`live_routing.py`](live_routing.py) on an Apple
M4 Pro laptop, at the recording's own speed. Nothing on screen was typed by hand or edited. Every word, probability and
time comes from the recording. The renderer and its check are kept with the launch material, not in this repository,
because they use the website's fonts. The check decodes every frame of the video and the GIF, and compares each with
what the recording puts there at that moment. It also reads the numbers back from the pixels.

## Run it

You need Apple silicon, this repository installed (`pip install -e .`), MLX (`pip install mlx==0.32.2 mlx-lm==0.31.3`,
the pins in `hunch/formats/requirements.txt`) and three files:

- **The MLX f16 build of Hunch 0.6B.** Get it with `hf download antareslabs/hunch-0.6b-preview-MLX --local-dir hunch-0.6b-mlx`
  ([FORMATS](../FORMATS.md)).
- **The calibration split** at `data/sprint_v3d/calibration.jsonl`, from the data build in [REPRODUCE](../REPRODUCE.md).
- **The Qwen3-0.6B tokenizer.** The loader fetches it from the Hub on first use; after that, `HF_HUB_OFFLINE=1` keeps it
  offline.

```bash
M=hunch-0.6b-mlx/hunch-0.6b-preview-mlx-f16
python examples/live_routing.py --model $M                        # live, in this terminal
python examples/live_routing.py --model $M --record run.jsonl     # headless: every call to a file
python examples/live_routing.py --model $M --recheck run.jsonl    # every recorded call again, candidates reversed
```

The recorded run was made from the repository root. The network was denied to the process by the macOS sandbox:

```bash
nice -n 15 sandbox-exec -p '(version 1)(allow default)(deny network*)' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false python examples/live_routing.py --record run.jsonl
```

`--record` first waits, for at most 45 minutes, until the 1-minute load average is below 5, so that the times are not
another program's. It never overwrites a recording.

## What each call is

- **The state** is the record's state with its `message` cut after the current word. The last call reads the record's
  message verbatim.
- **The question** is the record's own: its instruction and every candidate with its label and description, verbatim.
  The candidates are shuffled before every call (`random.Random(0)`), and the panel sorts them by probability.
- **Nothing carries over.** Each call is a whole request, from scratch. The fast engine (`hunch_mlx.load(path, fast=True)`)
  reads the state once per call instead of once per candidate. Nothing is reused from one word to the next.
- **The time** is wall clock around `decide()`, so it includes tokenisation, the forward pass and the softmax. The first
  calls compile Metal kernels, so before the recorded pass there is one warm-up call per message, on its first word.
  Their times are in the recording's header and below.
- **The temperature** is the release temperature embedded in the build (T = 0.8706), because both families are among
  those Hunch was trained on.

## How the two messages were chosen

The rule was written down in the project log before any recording, and before any message text had been read. For
Banking77 (a choice over 77 intents) and then MASSIVE (its choice over 60 intents), it takes the first record in file
order of `data/sprint_v3d/calibration.jsonl` that meets two conditions:

- its message has 9 to 18 words;
- its id is in none of the calibration split's lists of overlap with the training text (exact, Jaccard and near-duplicate)
  in [`docs/results/launch/contamination_ids.json`](../docs/results/launch/contamination_ids.json).

The calibration split was used only to fit Hunch's temperature, which cannot change an answer. Both messages are shown
whatever Hunch answers. Had a final answer differed from the record's label, the replay would say so on screen, and so
would the table below.

A first recording used Bitext (27 intents) as the second family. It was replaced by MASSIVE, under the same rule,
because Bitext's licence (CDLA-Sharing 1.0) is share-alike and this repository redistributes no Bitext rows; its final
answer had matched its label. CLINC150, with 151 intents, was left out before any recording. In a probe with a made-up message, one call over its
candidates took more than a second on this laptop, too slow to read as typing.

## How the replay is paced

"Recorded run · original speed" means the replay runs on the recording's clock. A word appears when its call started,
and the bars change when the call returned, so the gap between the two is the measured time of that decision. After
each decision the recorder paused 0.3 s so that the bars can be read, and the replay says so on screen.

The replay adds only three things: 0.5 s before a message's first word, 1.5 s on the final answer, and a 0.5 s
cross-fade between messages and back to the first, so that the GIF loops. The MP4 runs at 60 frames per second, so every
change appears within half a frame of its recorded time. The GIF's delays are rounded to 10 ms.

## Results

| Message | Candidates | Words, one call each | Final answer (its probability) | Record's label | Median call | p90 | Slowest |
|---|---|---|---|---|---|---|---|
| Banking77 | 77 | 9 | card arrival (0.62) | card arrival | 551.2 ms | 569.8 ms | 576.0 ms |
| MASSIVE | 60 | 9 | alarm query (0.98) | alarm query | 365.5 ms | 369.0 ms | 372.0 ms |

Records chosen by the rule: `mteb/banking77:train:35:intent` (Banking77, line 3 of the file) and `mteb/amazon_massive_intent:train:161:intent` (MASSIVE, line 1,199 of the file). Recorded 2026-09-25T21:00:16+02:00 on an Apple M4 Pro (MLX 0.32.2, float32 activations, FastHunchOnDevice), 1-minute load 2.74 at the start, network access denied to the process. Warm-up calls, not shown: 563 ms, 345 ms. Pauses after a decision: 0.302–0.305 s. Tokens per call, packed: 1,916–1,924 (Banking77) and 1,356–1,364 (MASSIVE).

Re-scored afterwards with every call's candidates in reverse order (`--recheck`): 18 of 18 answers unchanged, and no probability moved by more than 2.0e-06.

The `train` in the first record's id names the upstream dataset's own split. Both records are in Hunch's
calibration split, which it was not trained on. The block above is generated from the recording, and the replay's
check fails unless this document contains it verbatim.

## What this shows, and what it does not

It shows real output from one run: the probability Hunch gave every candidate after every word, and the time each call
took on one laptop. The reversed-order recheck above shows that the order of the candidates made no difference.

It does not show:

- **Accuracy.** Two messages are two data points. The measured accuracy is in [BENCHMARKS](../BENCHMARKS.md).
- **A trained skill for partial messages.** Hunch was trained on whole messages. After one or two words there is little
  to go on, and the early probabilities are diffuse by design. The rankings along the way show what a scorer calibrated
  on whole messages does with a fragment. They are not a claim that it predicts intent from fragments. Only the last
  call reads the text the record's label was given for.
- **Behaviour out of family.** Both families are among Hunch's training families, which is the easy case. On public
  benchmarks it was not trained on it is weaker ([BENCHMARKS](../BENCHMARKS.md)).
- **Speed elsewhere.** This is one laptop and one run. A call's cost grows with the number of candidates and the length
  of their descriptions (the token counts above), which is why the Banking77 calls take longer than the MASSIVE calls.
- **Streaming.** Each word is a new request, and the model keeps nothing between calls.
- **A person typing.** The message is replayed word by word from the record, with the stated pause.

"No network" on screen means the recording process was denied all network access by the macOS sandbox, and the Hub
client was set offline. The recording's header shows the sandbox was in force. The recorder tried to connect to a closed
port on the same machine, and the connection was refused by the sandbox, not by the port.

## Sources

The first message is from Banking77 by PolyAI, under CC BY 4.0. The second is from Amazon's MASSIVE dataset, under CC BY 4.0. The records are named above. The intents'
descriptions are those of Hunch's data build. The licences of every source are in
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).
