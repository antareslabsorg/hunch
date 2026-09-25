# Bitext customer support

## What changed vs. the current plan

Uses all **27 intents and 11 actual categories**. The card's prose currently claims ten categories and shows an incomplete/outdated label list. The live [column statistics](https://datasets-server.huggingface.co/statistics?dataset=bitext%2FBitext-customer-support-llm-chatbot-training-dataset&config=default&split=train) establish the names below. Treat the released `instruction` as customer content, not as the Hunch system instruction.

All instructions, paraphrases, candidate descriptions, and proposed conversion rules below are authored for Hunch; they are not quoted upstream prompts. Upstream field names, label names and encodings are cited separately. Retrieval date: **2026-09-19**.

Use the candidate `id` as a stable key; label shuffling must move the target with it. Keep labels and label-derived metadata out of `state`. Natural hard annotations are one-hot **label distributions**, not certainty about an objective event. Preserve native holdouts; derive dev/calibration only from eligible training groups; every paraphrase and related example stays with its base group. These are proposed applications of the record and split rules.

Implementation note: the documented schema uses probability maps; [the current runtime schema](../../../hunch/schema.py) uses arrays aligned to candidate order and integer `hard` indices, Boolean `criteria`, and ordered Score `levels`. The JSON candidate lists below are semantic specifications; serialize them through those existing runtime fields.

## Upstream facts and mapping

Observed upstream revision: [`430d1a89bd93bd1fa23c16f29dd53e73f0087443`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/tree/430d1a89bd93bd1fa23c16f29dd53e73f0087443), verified through the [repository metadata API](https://huggingface.co/api/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) on 2026-09-19. Pin this revision (or record a deliberate replacement) before ingestion; the research did not download the dataset.

The [card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) lists `flags`, `instruction`, `category`, `intent`, and `response`; it describes a hybrid synthetic generation process. The [statistics endpoint](https://datasets-server.huggingface.co/statistics?dataset=bitext%2FBitext-customer-support-llm-chatbot-training-dataset&config=default&split=train) returns 27 distinct string intents and 11 string categories for `default/train`.

Proposed `state = {"customer_message": row.instruction}`. Emit separate Choice questions for intent (K=27) and category (K=11), using exact string equality for targets. Neither `response`, `intent`, `category` nor generation flags belongs in state. The lists are alphabetically ordered for Hunch; they are not upstream integer encodings. Keep both questions, paraphrases, and near-duplicate generated utterances in the same group. No upstream train/dev/test separation or seed-family ID is provided in the inspected release; construct grouped train/dev/calibration/test partitions and disclose that seed-family independence cannot be guaranteed.

The following category-to-intent table is an **authored proposed consistency check** based on label meanings; verify every observed pair during engineering ingestion, and fail on mismatches rather than overwrite upstream categories. CANCEL covers cancellation fees while cancel_order belongs to ORDER.

```json
{
  "ACCOUNT": [
    "create_account",
    "delete_account",
    "edit_account",
    "recover_password",
    "registration_problems",
    "switch_account"
  ],
  "CANCEL": [
    "check_cancellation_fee"
  ],
  "CONTACT": [
    "contact_customer_service",
    "contact_human_agent"
  ],
  "DELIVERY": [
    "delivery_options",
    "delivery_period"
  ],
  "FEEDBACK": [
    "complaint",
    "review"
  ],
  "INVOICE": [
    "check_invoice",
    "get_invoice"
  ],
  "ORDER": [
    "cancel_order",
    "change_order",
    "place_order",
    "track_order"
  ],
  "PAYMENT": [
    "check_payment_methods",
    "payment_issue"
  ],
  "REFUND": [
    "check_refund_policy",
    "get_refund",
    "track_refund"
  ],
  "SHIPPING": [
    "change_shipping_address",
    "set_up_shipping_address"
  ],
  "SUBSCRIPTION": [
    "newsletter_subscription"
  ]
}
```

## Intent instruction and ten paraphrases

0. Choose the candidate intent that best matches the user's request.
1. Which listed intent most accurately describes what the user wants?
2. Identify the user's main intent from the candidate descriptions.
3. Select the intent that best captures the purpose of this message.
4. Match the user's request to the most appropriate listed intent.
5. What is the user trying to do, among the candidate intents?
6. Assign this message to its best-fitting intent.
7. Determine which candidate intent the message expresses.
8. Choose the intent whose description most closely fits this request.
9. Classify the user's goal using the available intent labels.
10. Which candidate best describes the action or information the user is seeking?

## Intent candidates

```json
[
  {
    "id": "cancel_order",
    "label": "cancel_order",
    "description": "The customer wants to cancel an existing order."
  },
  {
    "id": "change_order",
    "label": "change_order",
    "description": "The customer wants to modify an existing order."
  },
  {
    "id": "change_shipping_address",
    "label": "change_shipping_address",
    "description": "The customer wants to change an order's delivery address."
  },
  {
    "id": "check_cancellation_fee",
    "label": "check_cancellation_fee",
    "description": "The customer asks whether cancellation incurs a charge."
  },
  {
    "id": "check_invoice",
    "label": "check_invoice",
    "description": "The customer asks to inspect or clarify an invoice."
  },
  {
    "id": "check_payment_methods",
    "label": "check_payment_methods",
    "description": "The customer asks which payment methods are accepted."
  },
  {
    "id": "check_refund_policy",
    "label": "check_refund_policy",
    "description": "The customer asks about the conditions for obtaining a refund."
  },
  {
    "id": "complaint",
    "label": "complaint",
    "description": "The customer wants to complain about a product or service."
  },
  {
    "id": "contact_customer_service",
    "label": "contact_customer_service",
    "description": "The customer asks how to reach customer support."
  },
  {
    "id": "contact_human_agent",
    "label": "contact_human_agent",
    "description": "The customer asks to speak with a human representative."
  },
  {
    "id": "create_account",
    "label": "create_account",
    "description": "The customer wants to create an account."
  },
  {
    "id": "delete_account",
    "label": "delete_account",
    "description": "The customer wants to delete an account."
  },
  {
    "id": "delivery_options",
    "label": "delivery_options",
    "description": "The customer asks which delivery services or methods are available."
  },
  {
    "id": "delivery_period",
    "label": "delivery_period",
    "description": "The customer asks how long delivery takes."
  },
  {
    "id": "edit_account",
    "label": "edit_account",
    "description": "The customer wants to change account details."
  },
  {
    "id": "get_invoice",
    "label": "get_invoice",
    "description": "The customer wants a copy of an invoice."
  },
  {
    "id": "get_refund",
    "label": "get_refund",
    "description": "The customer wants to request or obtain a refund."
  },
  {
    "id": "newsletter_subscription",
    "label": "newsletter_subscription",
    "description": "The customer asks to manage a newsletter subscription."
  },
  {
    "id": "payment_issue",
    "label": "payment_issue",
    "description": "The customer reports difficulty completing or processing a payment."
  },
  {
    "id": "place_order",
    "label": "place_order",
    "description": "The customer wants to make a new purchase or order."
  },
  {
    "id": "recover_password",
    "label": "recover_password",
    "description": "The customer needs to recover or reset an account password."
  },
  {
    "id": "registration_problems",
    "label": "registration_problems",
    "description": "The customer reports difficulty registering for an account."
  },
  {
    "id": "review",
    "label": "review",
    "description": "The customer wants to leave or discuss a review."
  },
  {
    "id": "set_up_shipping_address",
    "label": "set_up_shipping_address",
    "description": "The customer wants to add or establish a shipping address."
  },
  {
    "id": "switch_account",
    "label": "switch_account",
    "description": "The customer wants to switch between accounts or account types."
  },
  {
    "id": "track_order",
    "label": "track_order",
    "description": "The customer asks for the progress or location of an order."
  },
  {
    "id": "track_refund",
    "label": "track_refund",
    "description": "The customer asks for the status of a refund."
  }
]
```

## Category instruction and ten paraphrases

0. Choose the candidate support category that best matches the user's request.
1. Which listed support category most accurately describes what the user wants?
2. Identify the user's main support category from the candidate descriptions.
3. Select the support category that best captures the purpose of this message.
4. Match the user's request to the most appropriate listed support category.
5. What is the user trying to do, among the candidate support categorys?
6. Assign this message to its best-fitting support category.
7. Determine which candidate support category the message expresses.
8. Choose the support category whose description most closely fits this request.
9. Classify the user's goal using the available support category labels.
10. Which candidate best describes the action or information the user is seeking?

## Category candidates

```json
[
  {
    "id": "ACCOUNT",
    "label": "ACCOUNT",
    "description": "The request concerns account creation, access, details, deletion, or switching."
  },
  {
    "id": "CANCEL",
    "label": "CANCEL",
    "description": "The request concerns the fee charged for cancellation."
  },
  {
    "id": "CONTACT",
    "label": "CONTACT",
    "description": "The request concerns reaching customer service or a human representative."
  },
  {
    "id": "DELIVERY",
    "label": "DELIVERY",
    "description": "The request concerns delivery options or delivery duration."
  },
  {
    "id": "FEEDBACK",
    "label": "FEEDBACK",
    "description": "The request concerns a complaint or a review."
  },
  {
    "id": "INVOICE",
    "label": "INVOICE",
    "description": "The request concerns checking or obtaining an invoice."
  },
  {
    "id": "ORDER",
    "label": "ORDER",
    "description": "The request concerns placing, modifying, cancelling, or tracking an order."
  },
  {
    "id": "PAYMENT",
    "label": "PAYMENT",
    "description": "The request concerns payment methods or payment problems."
  },
  {
    "id": "REFUND",
    "label": "REFUND",
    "description": "The request concerns refund policy, obtaining a refund, or tracking it."
  },
  {
    "id": "SHIPPING",
    "label": "SHIPPING",
    "description": "The request concerns setting or changing a shipping address."
  },
  {
    "id": "SUBSCRIPTION",
    "label": "SUBSCRIPTION",
    "description": "The request concerns newsletter subscription management."
  }
]
```

## Label-quality caveats

The [publisher's methodology](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset#overview) uses language-generation transformations and linguistic curation; these labels should carry synthetic provenance metadata, not be described as naturally occurring customer outcomes. Template expansion creates a leakage risk across random row splits. Published per-label human error rates and independent annotation distributions: **not found**. Placeholder entities and assistant responses can reveal intent; audit on real independently authored support requests before using headline accuracy as a product claim.
