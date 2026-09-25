# BoolQ

## What changed vs. the current plan

Pins `google/boolq` Boolean polarity and separates the question text from the answer; the public HF release has no labeled test split.

All instructions, paraphrases, candidate descriptions, and proposed conversion rules below are authored for Hunch; they are not quoted upstream prompts. Upstream field names, label names and encodings are cited separately. Retrieval date: **2026-09-19**.

Use the candidate `id` as a stable key; label shuffling must move the target with it. Keep labels and label-derived metadata out of `state`. Natural hard annotations are one-hot **label distributions**, not certainty about an objective event. Preserve native holdouts; derive dev/calibration only from eligible training groups; every paraphrase and related example stays with its base group. These are proposed applications of the record and split rules.

Implementation note: the documented schema uses probability maps; [the current runtime schema](../../../hunch/schema.py) uses arrays aligned to candidate order and integer `hard` indices, Boolean `criteria`, and ordered Score `levels`. The JSON candidate lists below are semantic specifications; serialize them through those existing runtime fields.

## Upstream facts and mapping

Observed upstream revision: [`35b264d03638db9f4ce671b711558bf7ff0f80d5`](https://huggingface.co/datasets/google/boolq/tree/35b264d03638db9f4ce671b711558bf7ff0f80d5), verified through the [repository metadata API](https://huggingface.co/api/datasets/google/boolq) on 2026-09-19. Pin this revision (or record a deliberate replacement) before ingestion; the research did not download the dataset.

The [card and features](https://huggingface.co/datasets/google/boolq/blob/main/README.md) expose `question`, `passage`, Boolean `answer`, train (9,427) and validation (3,270). The [authors' repository](https://github.com/google-research-datasets/boolean-questions) and [paper](https://aclanthology.org/N19-1300/) describe naturally occurring yes/no questions paired with passages.

Proposed `state = {"passage": row.passage, "question": row.question}`; Boolean K=2, `family = qa.boolq`. Boolean `answer == true` maps to Hunch `true`; false maps to `false`. The two candidate descriptions become Boolean `criteria`. Do not treat Python truthiness of string `"false"` as a label. Use passage hash or original page identity as source group, merging repeated questions and passages; page title is not present in the pinned HF schema. Reserve native validation as final evaluation if no other labeled holdout is chosen; derive dev/calibration from train groups.

## Exact instruction and ten paraphrases

0. Using the passage, is the answer to the question yes?
1. Does the passage support answering yes to the question?
2. Based on the passage, should the question be answered yes?
3. Determine whether the passage's answer to the question is yes.
4. Answer whether yes is the correct response to the question given the passage.
5. Is yes the answer warranted by the passage for this question?
6. Read the passage and decide if the question's answer is yes.
7. According to the passage, is the answer yes rather than no?
8. Does the supplied passage imply a yes answer to the question?
9. Judge whether the passage answers the question affirmatively.
10. Is the correct yes-or-no answer to the question yes, using the passage?

## Candidate semantics

```json
[
  {
    "id": "false",
    "label": "false",
    "description": "The answer to the question, using the passage, is no."
  },
  {
    "id": "true",
    "label": "true",
    "description": "The answer to the question, using the passage, is yes."
  }
]
```

## Label-quality caveats

The [BoolQ paper](https://aclanthology.org/N19-1300/) emphasizes the difficulty of natural yes/no questions compared with automatically generated tasks. A no answer is not the same thing as an unsupported proposition: this is an answerability-filtered binary dataset with no unknown class. Do not add unknown labels or convert retrieval failure to no. Per-example answer vote counts in this HF release: **not found**.
