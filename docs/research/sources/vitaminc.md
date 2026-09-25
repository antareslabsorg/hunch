# VitaminC

## What changed vs. the current plan

Preserves contrastive case/revision groups and real-versus-synthetic metadata; adds cross-source FEVER overlap protection. Uses exact string labels rather than an assumed integer order.

All instructions, paraphrases, candidate descriptions, and proposed conversion rules below are authored for Hunch; they are not quoted upstream prompts. Upstream field names, label names and encodings are cited separately. Retrieval date: **2026-09-19**.

Use the candidate `id` as a stable key; label shuffling must move the target with it. Keep labels and label-derived metadata out of `state`. Natural hard annotations are one-hot **label distributions**, not certainty about an objective event. Preserve native holdouts; derive dev/calibration only from eligible training groups; every paraphrase and related example stays with its base group. These are proposed applications of the record and split rules.

Implementation note: the documented schema uses probability maps; [the current runtime schema](../../../hunch/schema.py) uses arrays aligned to candidate order and integer `hard` indices, Boolean `criteria`, and ordered Score `levels`. The JSON candidate lists below are semantic specifications; serialize them through those existing runtime fields.

## Upstream facts and mapping

Observed upstream revision: [`be6febb761b0b2807687e61e0b5282e459df2fa0`](https://huggingface.co/datasets/tals/vitaminc/tree/be6febb761b0b2807687e61e0b5282e459df2fa0), verified through the [repository metadata API](https://huggingface.co/api/datasets/tals/vitaminc) on 2026-09-19. Pin this revision (or record a deliberate replacement) before ingestion; the research did not download the dataset.

Source `tals/vitaminc`, default. The [dataset card](https://huggingface.co/datasets/tals/vitaminc), [bounded viewer schema](https://datasets-server.huggingface.co/first-rows?dataset=tals%2Fvitaminc&config=default&split=train), and [column statistics](https://datasets-server.huggingface.co/statistics?dataset=tals%2Fvitaminc&config=default&split=train) expose `unique_id`, `case_id`, `wiki_revision_id`, `label`, `claim`, `evidence`, `page`, `revision_type`, `FEVER_id`, and `big_bench_canary`. The exact label values are SUPPORTS, REFUTES, and NOT ENOUGH INFO; revision type is real or synthetic.

Proposed `state = {"claim": row.claim, "evidence": row.evidence}`; Choice K=3, `family = verification.vitaminc`. Map exact string labels to the candidate IDs below. Keep revision metadata, identifiers, labels, and the canary field outside model state. Preserve the canary in dataset provenance and any permitted redistributed source artifacts; omitting it from a classification input is not permission to remove provenance from the source.

Use connected source groups across shared `case_id`, `wiki_revision_id`, and nonempty `FEVER_id`; all before/after and synthetic contrastive variants stay together. Optionally strengthen to page-level groups if the evaluation is explicitly document-disjoint. Never key independently by `unique_id` alone. Preserve native train/validation/test and quarantine overlapping training groups. The [project](https://github.com/TalSchuster/VitaminC) distinguishes real and synthetic test slices; report them separately and report FEVER overlap separately.

## Exact instruction and ten paraphrases

0. Using only the supplied evidence, classify the claim as supported, refuted, or not enough information.
1. Does the provided evidence support the claim, refute it, or leave it unresolved?
2. Choose whether the claim is supported, contradicted, or not established by the supplied evidence.
3. Determine the claim's relationship to the evidence: supports, refutes, or not enough information.
4. Based solely on the evidence, should the claim be accepted, rejected, or left undecided?
5. Select the evidence-grounded verdict for the claim from supports, refutes, and not enough information.
6. Judge whether the evidence establishes the claim, establishes its opposite, or does neither.
7. What verdict does the supplied evidence justify for this claim: supported, refuted, or unresolved?
8. Assess the claim using the evidence and choose support, refutation, or insufficient information.
9. Is the claim shown true, shown false, or left undetermined by the provided evidence?
10. Classify the claim against this evidence without adding outside facts: supports, refutes, or not enough information.

## Candidate list

```json
[
  {
    "id": "SUPPORTS",
    "label": "supports",
    "description": "The supplied evidence establishes the claim as true."
  },
  {
    "id": "REFUTES",
    "label": "refutes",
    "description": "The supplied evidence establishes that the claim is false."
  },
  {
    "id": "NOT ENOUGH INFO",
    "label": "not enough information",
    "description": "The supplied evidence does not establish either the claim or its negation."
  }
]
```

## Label-quality caveats

The [paper](https://aclanthology.org/2021.naacl-main.52/) builds contrastive evidence from Wikipedia revisions and synthetic changes. Nearly identical evidence can reverse the verdict; topical similarity or remembered world facts must not override the supplied version. Synthetic edits do not have the same provenance as naturally occurring edits. Model confidence against hard dataset labels is not a measurement of annotator disagreement; per-example vote counts in this export were **not found**. A Wikipedia revision itself is not an external guarantee of factual truth.
