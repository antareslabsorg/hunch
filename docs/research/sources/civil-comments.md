# Civil Comments

## What changed vs. the current plan

Emits **seven separate Boolean questions with soft targets**, not one seven-way class or thresholded hard toxicity target. The HF mirror lacks article and thread IDs; stronger source grouping requires an authorized upstream metadata join.

All instructions, paraphrases, candidate descriptions, and proposed conversion rules below are authored for Hunch; they are not quoted upstream prompts. Upstream field names, label names and encodings are cited separately. Retrieval date: **2026-09-19**.

Use the candidate `id` as a stable key; label shuffling must move the target with it. Keep labels and label-derived metadata out of `state`. Natural hard annotations are one-hot **label distributions**, not certainty about an objective event. Preserve native holdouts; derive dev/calibration only from eligible training groups; every paraphrase and related example stays with its base group. These are proposed applications of the record and split rules.

Implementation note: the documented schema uses probability maps; [the current runtime schema](../../../hunch/schema.py) uses arrays aligned to candidate order and integer `hard` indices, Boolean `criteria`, and ordered Score `levels`. The JSON candidate lists below are semantic specifications; serialize them through those existing runtime fields.

## Upstream facts and mapping

Observed upstream revision: [`f2970eb3a55777454c94069077cc8d9b5866312d`](https://huggingface.co/datasets/google/civil_comments/tree/f2970eb3a55777454c94069077cc8d9b5866312d), verified through the [repository metadata API](https://huggingface.co/api/datasets/google/civil_comments) on 2026-09-19. Pin this revision (or record a deliberate replacement) before ingestion; the research did not download the dataset.

The [HF schema](https://huggingface.co/datasets/google/civil_comments/blob/main/README.md) contains `text` and seven float fields: `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit`. The [maintainer's TFDS documentation](https://www.tensorflow.org/datasets/catalog/civil_comments) explicitly defines those values as fractions of annotators assigning each attribute. Different attributes can apply simultaneously.

Proposed state: `{"comment": row.text}`. For each field f, create `family = "moderation.civil." + f`, Boolean K=2 and `probs = [1-row[f], row[f]]` in false/true order. Set `semantic_kind = label_distribution`, `provenance = human_annotation_distribution`, and `hard = null` for the soft-target training task. Validate finite values in [0,1]; missing labels are omitted individually, not turned into zero. Do not reconstruct annotator counts by assuming a common denominator: denominators are **not found** in this HF schema. For a separate hard-label metric, preregister threshold and tie behavior; never replace the stored fraction.

All seven questions from one comment share a source group. Preserve native train/validation/test. Prefer original `article_id` and thread components recovered through stable upstream IDs; neither appears in this HF schema. If unavailable, normalize duplicate comment text, document the weaker grouping, and avoid a claim of article-disjoint testing. The [TFDS note](https://www.tensorflow.org/datasets/catalog/civil_comments) warns that parent text can cross split boundaries and that original annotators did not see it. Consequently do not add parent context while silently retaining the context-free targets.

Definitions below are concise Hunch working glosses of the categories, not a recovered verbatim annotation manual. A full release-matched annotation rubric was **not found** in the HF card; confirm these glosses against the audit protocol before using them as deployment policy.

## toxicity: exact instruction and ten paraphrases

0. Does the comment contain toxicity? Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
1. Is toxicity present in the comment? Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
2. Would this comment be annotated as containing toxicity? Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
3. Determine whether the comment expresses toxicity. Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
4. Assess whether the comment includes toxicity. Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
5. Does the comment meet the definition of toxicity? Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
6. Classify whether this comment contains toxicity. Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
7. Identify whether toxicity occurs in the comment. Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
8. Should the comment receive the toxicity annotation? Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
9. Judge whether this comment demonstrates toxicity. Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.
10. Is the toxicity label applicable to this comment? Definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion.

### Candidate semantics

```json
[
  {
    "id": "false",
    "label": "no toxicity",
    "description": "The comment does not satisfy this definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion."
  },
  {
    "id": "true",
    "label": "toxicity",
    "description": "The comment satisfies this definition: Rude, disrespectful, or unreasonable language likely to make someone leave a discussion."
  }
]
```

## severe_toxicity: exact instruction and ten paraphrases

0. Does the comment contain severe toxicity? Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
1. Is severe toxicity present in the comment? Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
2. Would this comment be annotated as containing severe toxicity? Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
3. Determine whether the comment expresses severe toxicity. Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
4. Assess whether the comment includes severe toxicity. Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
5. Does the comment meet the definition of severe toxicity? Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
6. Classify whether this comment contains severe toxicity. Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
7. Identify whether severe toxicity occurs in the comment. Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
8. Should the comment receive the severe_toxicity annotation? Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
9. Judge whether this comment demonstrates severe toxicity. Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.
10. Is the severe_toxicity label applicable to this comment? Definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement.

### Candidate semantics

```json
[
  {
    "id": "false",
    "label": "no severe_toxicity",
    "description": "The comment does not satisfy this definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement."
  },
  {
    "id": "true",
    "label": "severe_toxicity",
    "description": "The comment satisfies this definition: An extreme degree of toxic or abusive language, beyond ordinary rudeness or disagreement."
  }
]
```

## obscene: exact instruction and ten paraphrases

0. Does the comment contain obscenity? Definition: Obscene, vulgar, or profane language.
1. Is obscenity present in the comment? Definition: Obscene, vulgar, or profane language.
2. Would this comment be annotated as containing obscenity? Definition: Obscene, vulgar, or profane language.
3. Determine whether the comment expresses obscenity. Definition: Obscene, vulgar, or profane language.
4. Assess whether the comment includes obscenity. Definition: Obscene, vulgar, or profane language.
5. Does the comment meet the definition of obscenity? Definition: Obscene, vulgar, or profane language.
6. Classify whether this comment contains obscenity. Definition: Obscene, vulgar, or profane language.
7. Identify whether obscenity occurs in the comment. Definition: Obscene, vulgar, or profane language.
8. Should the comment receive the obscene annotation? Definition: Obscene, vulgar, or profane language.
9. Judge whether this comment demonstrates obscenity. Definition: Obscene, vulgar, or profane language.
10. Is the obscene label applicable to this comment? Definition: Obscene, vulgar, or profane language.

### Candidate semantics

```json
[
  {
    "id": "false",
    "label": "no obscene",
    "description": "The comment does not satisfy this definition: Obscene, vulgar, or profane language."
  },
  {
    "id": "true",
    "label": "obscene",
    "description": "The comment satisfies this definition: Obscene, vulgar, or profane language."
  }
]
```

## threat: exact instruction and ten paraphrases

0. Does the comment contain a threat? Definition: An expressed intention or wish to inflict harm on a person or group.
1. Is a threat present in the comment? Definition: An expressed intention or wish to inflict harm on a person or group.
2. Would this comment be annotated as containing a threat? Definition: An expressed intention or wish to inflict harm on a person or group.
3. Determine whether the comment expresses a threat. Definition: An expressed intention or wish to inflict harm on a person or group.
4. Assess whether the comment includes a threat. Definition: An expressed intention or wish to inflict harm on a person or group.
5. Does the comment meet the definition of a threat? Definition: An expressed intention or wish to inflict harm on a person or group.
6. Classify whether this comment contains a threat. Definition: An expressed intention or wish to inflict harm on a person or group.
7. Identify whether a threat occurs in the comment. Definition: An expressed intention or wish to inflict harm on a person or group.
8. Should the comment receive the threat annotation? Definition: An expressed intention or wish to inflict harm on a person or group.
9. Judge whether this comment demonstrates a threat. Definition: An expressed intention or wish to inflict harm on a person or group.
10. Is the threat label applicable to this comment? Definition: An expressed intention or wish to inflict harm on a person or group.

### Candidate semantics

```json
[
  {
    "id": "false",
    "label": "no threat",
    "description": "The comment does not satisfy this definition: An expressed intention or wish to inflict harm on a person or group."
  },
  {
    "id": "true",
    "label": "threat",
    "description": "The comment satisfies this definition: An expressed intention or wish to inflict harm on a person or group."
  }
]
```

## insult: exact instruction and ten paraphrases

0. Does the comment contain an insult? Definition: Language that demeans or attacks a person or group.
1. Is an insult present in the comment? Definition: Language that demeans or attacks a person or group.
2. Would this comment be annotated as containing an insult? Definition: Language that demeans or attacks a person or group.
3. Determine whether the comment expresses an insult. Definition: Language that demeans or attacks a person or group.
4. Assess whether the comment includes an insult. Definition: Language that demeans or attacks a person or group.
5. Does the comment meet the definition of an insult? Definition: Language that demeans or attacks a person or group.
6. Classify whether this comment contains an insult. Definition: Language that demeans or attacks a person or group.
7. Identify whether an insult occurs in the comment. Definition: Language that demeans or attacks a person or group.
8. Should the comment receive the insult annotation? Definition: Language that demeans or attacks a person or group.
9. Judge whether this comment demonstrates an insult. Definition: Language that demeans or attacks a person or group.
10. Is the insult label applicable to this comment? Definition: Language that demeans or attacks a person or group.

### Candidate semantics

```json
[
  {
    "id": "false",
    "label": "no insult",
    "description": "The comment does not satisfy this definition: Language that demeans or attacks a person or group."
  },
  {
    "id": "true",
    "label": "insult",
    "description": "The comment satisfies this definition: Language that demeans or attacks a person or group."
  }
]
```

## identity_attack: exact instruction and ten paraphrases

0. Does the comment contain an identity attack? Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
1. Is an identity attack present in the comment? Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
2. Would this comment be annotated as containing an identity attack? Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
3. Determine whether the comment expresses an identity attack. Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
4. Assess whether the comment includes an identity attack. Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
5. Does the comment meet the definition of an identity attack? Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
6. Classify whether this comment contains an identity attack. Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
7. Identify whether an identity attack occurs in the comment. Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
8. Should the comment receive the identity_attack annotation? Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
9. Judge whether this comment demonstrates an identity attack. Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.
10. Is the identity_attack label applicable to this comment? Definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity.

### Candidate semantics

```json
[
  {
    "id": "false",
    "label": "no identity_attack",
    "description": "The comment does not satisfy this definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity."
  },
  {
    "id": "true",
    "label": "identity_attack",
    "description": "The comment satisfies this definition: An attack directed at someone because of an identity characteristic rather than mere mention of that identity."
  }
]
```

## sexual_explicit: exact instruction and ten paraphrases

0. Does the comment contain sexually explicit content? Definition: Explicit descriptions of sexual acts or sexual anatomy.
1. Is sexually explicit content present in the comment? Definition: Explicit descriptions of sexual acts or sexual anatomy.
2. Would this comment be annotated as containing sexually explicit content? Definition: Explicit descriptions of sexual acts or sexual anatomy.
3. Determine whether the comment expresses sexually explicit content. Definition: Explicit descriptions of sexual acts or sexual anatomy.
4. Assess whether the comment includes sexually explicit content. Definition: Explicit descriptions of sexual acts or sexual anatomy.
5. Does the comment meet the definition of sexually explicit content? Definition: Explicit descriptions of sexual acts or sexual anatomy.
6. Classify whether this comment contains sexually explicit content. Definition: Explicit descriptions of sexual acts or sexual anatomy.
7. Identify whether sexually explicit content occurs in the comment. Definition: Explicit descriptions of sexual acts or sexual anatomy.
8. Should the comment receive the sexual_explicit annotation? Definition: Explicit descriptions of sexual acts or sexual anatomy.
9. Judge whether this comment demonstrates sexually explicit content. Definition: Explicit descriptions of sexual acts or sexual anatomy.
10. Is the sexual_explicit label applicable to this comment? Definition: Explicit descriptions of sexual acts or sexual anatomy.

### Candidate semantics

```json
[
  {
    "id": "false",
    "label": "no sexual_explicit",
    "description": "The comment does not satisfy this definition: Explicit descriptions of sexual acts or sexual anatomy."
  },
  {
    "id": "true",
    "label": "sexual_explicit",
    "description": "The comment satisfies this definition: Explicit descriptions of sexual acts or sexual anatomy."
  }
]
```

## Label-quality caveats

The [dataset's bias paper](https://arxiv.org/abs/1903.04561) motivates subgroup-sensitive evaluation: aggregate quality can conceal errors associated with identity terms. Identity mention is not itself identity attack. The labels measure annotator judgments, not a frequency of downstream harm or the probability that a platform must remove a comment. News-comment text from 2015–2017 differs from current assistant conversations. Because votes are aggregated and counts are missing, equal weighting of rows is a documented choice, not inverse-variance weighting. Adding conversation context changes the annotator's information and therefore can change the target definition.
