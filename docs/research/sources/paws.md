# PAWS

## What changed vs. the current plan

Pins `labeled_final` and distinguishes human-labeled paraphrases from the noisily labeled `unlabeled_final` configuration.

All instructions, paraphrases, candidate descriptions, and proposed conversion rules below are authored for Hunch; they are not quoted upstream prompts. Upstream field names, label names and encodings are cited separately. Retrieval date: **2026-09-19**.

Use the candidate `id` as a stable key; label shuffling must move the target with it. Keep labels and label-derived metadata out of `state`. Natural hard annotations are one-hot **label distributions**, not certainty about an objective event. Preserve native holdouts; derive dev/calibration only from eligible training groups; every paraphrase and related example stays with its base group. These are proposed applications of the record and split rules.

Implementation note: the documented schema uses probability maps; [the current runtime schema](../../../hunch/schema.py) uses arrays aligned to candidate order and integer `hard` indices, Boolean `criteria`, and ordered Score `levels`. The JSON candidate lists below are semantic specifications; serialize them through those existing runtime fields.

## Upstream facts and mapping

Observed upstream revision: [`161ece9501cf0a11f3e48bd356eaa82de46d6a09`](https://huggingface.co/datasets/google-research-datasets/paws/tree/161ece9501cf0a11f3e48bd356eaa82de46d6a09), verified through the [repository metadata API](https://huggingface.co/api/datasets/google-research-datasets/paws) on 2026-09-19. Pin this revision (or record a deliberate replacement) before ingestion; the research did not download the dataset.

[google-research-datasets/paws](https://huggingface.co/datasets/google-research-datasets/paws/blob/main/README.md) exposes `id`, `sentence1`, `sentence2`, `label`; label 0 means different meaning and 1 means paraphrase. `labeled_final` has train/validation/test; `labeled_swap` is auxiliary training and `unlabeled_final` uses noisy labels.

Proposed `state = {"sentence1": row.sentence1, "sentence2": row.sentence2}`; Boolean K=2, `family = paraphrase.paws`; label 1 -> true and 0 -> false. Descriptions below become Boolean criteria. Prefer original Wikipedia source/seed sentence groups; these are absent from the HF schema. Build connected groups over shared normalized sentences, including reversed pairs, and retain native splits. Quarantine train rows connected to held-out rows. Do not mix the noisy configuration into a human-gold provenance bucket.

## Exact instruction and ten paraphrases

0. Do the two sentences express the same meaning?
1. Are these two sentences paraphrases of one another?
2. Do both sentences convey equivalent information?
3. Is the meaning of the first sentence equivalent to the second?
4. Decide whether the sentences say the same thing.
5. Can the two sentences be treated as semantic paraphrases?
6. Does each sentence preserve the meaning of the other?
7. Determine whether the sentences have equivalent meanings.
8. Are the two statements meaning-equivalent despite differences in wording?
9. Would these sentences communicate the same information?
10. Is this pair a paraphrase pair?

## Candidate semantics

```json
[
  {
    "id": "false",
    "label": "false",
    "description": "The sentences differ in meaning despite any shared words."
  },
  {
    "id": "true",
    "label": "true",
    "description": "The sentences express equivalent meaning."
  }
]
```

## Label-quality caveats

The [PAWS paper](https://aclanthology.org/N19-1131/) and [dataset documentation](https://github.com/google-research-datasets/paws) construct high-overlap pairs using word swapping and back translation. Word overlap is not a valid paraphrase rule; roles, negation, direction, and argument order matter. The card distinguishes human judgments from noisy labels. Sentence-pair IDs alone do not establish parent-document independence. Keep PAWS-QQP reconstruction and its separate terms outside this pinned PAWS-Wiki task.
