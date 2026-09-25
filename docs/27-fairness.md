# Fairness-sensitive read of the toxicity families

Pre-declared in the project log (not published): does the scorer treat comments that mention an identity differently from
comments that do not? Data: every Civil Comments row of the dev and calibration splits (in-family, not trained on as rows,
although 150 dev and 60 calibration Civil Comments states also occur verbatim in training, see docs/21; this analysis does not read the
test split), scored with `hunch.evaluate --max-per-family 5000` (those prediction files are not
included) and analysed by `scripts/fairness_eval.py`: `docs/results/day6/fairness.json`, `docs/results/day7/v1/fairness_g2-17-v4t.json`.

Checkpoints: `w16-06-v3dgp10-1200-s55`, `g2-17-v4t`.

## Method

A row is *identity* if the comment contains a whole-word, case-insensitive match to any term below; otherwise *none*. This is a
proxy for identity content, not an annotation: it misses references without these words and catches quoted or negated
mentions. Per group: n, accuracy, cross-entropy against the target distribution (the card's NLL), smooth ECE
(`smece`, the card's), and the **false-toxic rate** — the share of comments whose gold says *not toxic* that the scorer
calls toxic (Boolean: gold P(true) < 0.5 and predicted P(true) ≥ 0.5; Score: gold level ≤ 1 and predicted level ≥ 3) — and its mirror,
the **missed-toxic rate**: gold toxic (Boolean P(true) ≥ 0.5; Score level ≥ 3) but not scored toxic.
Gaps (identity − none) carry bootstrap 95 % intervals (1000 resamples, seed 0). A group under 200 rows is reported
as too small and gets no interval. No bar was declared; the gaps and their intervals are the finding.

Lexicon, verbatim:

- **gender:** woman, women, man, men, female, male, girl, girls, boy, boys, wife, husband, mother, father, transgender, trans, nonbinary, non-binary
- **sexual orientation:** gay, gays, lesbian, lesbians, bisexual, homosexual, homosexuals, queer, lgbt, lgbtq, heterosexual, straight
- **religion:** christian, christians, catholic, catholics, protestant, jew, jews, jewish, muslim, muslims, islam, islamic, hindu, hindus, buddhist, buddhists, atheist, atheists, sikh, mormon, evangelical
- **race / ethnicity:** black, blacks, white, whites, asian, asians, latino, latina, latinos, hispanic, hispanics, african, africans, arab, arabs, mexican, mexicans, chinese, indian, indians, native american, indigenous
- **disability:** disabled, disability, disabilities, wheelchair, blind, deaf, autistic, autism, mentally ill, mental illness, schizophrenic, bipolar, retarded, handicapped

## Results

### `w16-06-v3dgp10-1200-s55`

**Civil Comments — Boolean toxicity attributes** — identity mentions by category (comments): disability 20, gender 143, race / ethnicity 80, religion 28, sexual orientation 32

| group | n | accuracy | cross-entropy | smooth ECE | false-toxic rate (n not-toxic) | missed-toxic rate (n toxic) |
|---|---:|---:|---:|---:|---:|---:|
| identity | 1280 | 0.970 | 0.207 | 0.036 | 0.3 % (1214) | 77.3 % (66) |
| none | 8675 | 0.989 | 0.097 | 0.018 | 0.3 % (8502) | 59.0 % (173) |
| *gap, identity − none* | | -0.020 [-0.029, -0.011] | +0.110 [+0.091, +0.129] | | -0.0 pt [-0.3, +0.4] | +18.3 pt [+4.6, +31.9] |

**Civil Comments — 5-level toxicity score** — identity mentions by category (comments): disability 20, gender 146, race / ethnicity 81, religion 32, sexual orientation 35

| group | n | accuracy | cross-entropy | smooth ECE | false-toxic rate (n not-toxic) | missed-toxic rate (n toxic) |
|---|---:|---:|---:|---:|---:|---:|
| identity | 264 | 0.610 | 0.861 | 0.061 | 0.0 % (212) | 70.6 % (17) |
| none | 1748 | 0.797 | 0.500 | 0.024 | 0.5 % (1617) | 36.6 % (41) |
| *gap, identity − none* | | -0.187 [-0.249, -0.126] | +0.361 [+0.250, +0.470] | | -0.5 pt [-0.9, -0.2] | +34.0 pt [+6.6, +62.1] |

### `g2-17-v4t`

**Civil Comments — Boolean toxicity attributes** — identity mentions by category (comments): disability 20, gender 143, race / ethnicity 80, religion 28, sexual orientation 32

| group | n | accuracy | cross-entropy | smooth ECE | false-toxic rate (n not-toxic) | missed-toxic rate (n toxic) |
|---|---:|---:|---:|---:|---:|---:|
| identity | 1280 | 0.973 | 0.202 | 0.043 | 0.2 % (1214) | 77.3 % (66) |
| none | 8675 | 0.990 | 0.096 | 0.019 | 0.1 % (8502) | 67.1 % (173) |
| *gap, identity − none* | | -0.017 [-0.026, -0.008] | +0.106 [+0.089, +0.124] | | +0.0 pt [-0.2, +0.3] | +10.2 pt [-2.5, +21.7] |

**Civil Comments — 5-level toxicity score** — identity mentions by category (comments): disability 20, gender 146, race / ethnicity 81, religion 32, sexual orientation 35

| group | n | accuracy | cross-entropy | smooth ECE | false-toxic rate (n not-toxic) | missed-toxic rate (n toxic) |
|---|---:|---:|---:|---:|---:|---:|
| identity | 264 | 0.644 | 0.839 | 0.072 | 0.0 % (212) | 70.6 % (17) |
| none | 1748 | 0.795 | 0.496 | 0.030 | 0.0 % (1617) | 36.6 % (41) |
| *gap, identity − none* | | -0.151 [-0.214, -0.094] | +0.343 [+0.241, +0.458] | | +0.0 pt [+0.0, +0.0] | +34.0 pt [+7.4, +59.4] |

## Reading

A gap whose interval excludes zero is a measured disparity on this proxy; one whose interval covers zero is not evidence of
parity, only of a study too small to see a gap of that size.
What the proxy misses — identity content without the listed words — is not measured here at all.
