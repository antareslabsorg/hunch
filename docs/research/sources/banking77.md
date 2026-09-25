# Banking77

## What changed vs. the current plan

Provides all 77 descriptions and fixes exact casing/punctuation: `Refund_not_showing_up` and `reverted_card_payment?` are upstream labels. No undocumented numerical relabeling is permitted. The potentially overlapping physical-card and transfer labels need human boundary review, not invented rules.

All instructions, paraphrases, candidate descriptions, and proposed conversion rules below are authored for Hunch; they are not quoted upstream prompts. Upstream field names, label names and encodings are cited separately. Retrieval date: **2026-09-19**.

Use the candidate `id` as a stable key; label shuffling must move the target with it. Keep labels and label-derived metadata out of `state`. Natural hard annotations are one-hot **label distributions**, not certainty about an objective event. Preserve native holdouts; derive dev/calibration only from eligible training groups; every paraphrase and related example stays with its base group. These are proposed applications of the record and split rules.

Implementation note: the documented schema uses probability maps; [the current runtime schema](../../../hunch/schema.py) uses arrays aligned to candidate order and integer `hard` indices, Boolean `criteria`, and ordered Score `levels`. The JSON candidate lists below are semantic specifications; serialize them through those existing runtime fields.

## Upstream facts and mapping

Observed upstream revision: [`90d4e2ee5521c04fc1488f065b8b083658768c57`](https://huggingface.co/datasets/PolyAI/banking77/tree/90d4e2ee5521c04fc1488f065b8b083658768c57), verified through the [repository metadata API](https://huggingface.co/api/datasets/PolyAI/banking77) on 2026-09-19. Pin this revision (or record a deliberate replacement) before ingestion; the research did not download the dataset.

Current engineering compatibility: [the adapter](../../../hunch/data/families.py) loads `mteb/banking77` rather than PolyAI. The [mirror card](https://huggingface.co/datasets/mteb/banking77/tree/18072d2685ea682290f7b8924d94c62acc19c0b2) at revision `18072d2685ea682290f7b8924d94c62acc19c0b2` exposes `text`, integer `label`, and `label_text`, with 9,993 train / 3,076 test rows, so it is not the identical 10,003/3,080-row benchmark. Its [label-text statistics](https://datasets-server.huggingface.co/statistics?dataset=mteb%2Fbanking77&config=default&split=train) contain the same 77 canonical names. Join the descriptions below by exact `label_text`, validate the mirror's integer/name bijection, and retain its actual counts and revision in reports. Do not use the original numeric order blindly on a changed mirror.

Source: `PolyAI/banking77`, English, `text` and ClassLabel `label`; integer 0–76 maps to the following list in order. Train has 10,003 rows; test has 3,080. The [loader](https://huggingface.co/datasets/PolyAI/banking77/blob/main/banking77.py) and [card](https://huggingface.co/datasets/PolyAI/banking77) specify that mapping and splits.

Proposed `state = {"utterance": row.text}`; `family = support.intent.banking77`; `question.type = choice`, K=77; target is the candidate at `row.label`. Group by normalized utterance hash, with exact/near duplicates merged before splitting; the release does not expose customer/conversation identifiers, so customer-disjoint guarantees are **not found**. Keep the native test holdout.

## Exact instruction and ten paraphrases

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

## Candidate list

Order is the upstream integer encoding; descriptions are authored.

```json
[
  {
    "id": "activate_my_card",
    "label": "activate_my_card",
    "description": "The user wants to activate a received payment card."
  },
  {
    "id": "age_limit",
    "label": "age_limit",
    "description": "The user asks about the minimum or maximum age for using the banking service."
  },
  {
    "id": "apple_pay_or_google_pay",
    "label": "apple_pay_or_google_pay",
    "description": "The user asks about adding or using a card with Apple Pay or Google Pay."
  },
  {
    "id": "atm_support",
    "label": "atm_support",
    "description": "The user asks which cash machines support their card."
  },
  {
    "id": "automatic_top_up",
    "label": "automatic_top_up",
    "description": "The user wants to configure or understand automatic balance top-ups."
  },
  {
    "id": "balance_not_updated_after_bank_transfer",
    "label": "balance_not_updated_after_bank_transfer",
    "description": "The user reports that an incoming bank transfer has not appeared in the balance."
  },
  {
    "id": "balance_not_updated_after_cheque_or_cash_deposit",
    "label": "balance_not_updated_after_cheque_or_cash_deposit",
    "description": "The user reports a missing balance update after depositing cash or a cheque."
  },
  {
    "id": "beneficiary_not_allowed",
    "label": "beneficiary_not_allowed",
    "description": "The user cannot send money to a particular beneficiary because that recipient is not allowed."
  },
  {
    "id": "cancel_transfer",
    "label": "cancel_transfer",
    "description": "The user wants to cancel a money transfer."
  },
  {
    "id": "card_about_to_expire",
    "label": "card_about_to_expire",
    "description": "The user asks what to do about a card nearing its expiry date."
  },
  {
    "id": "card_acceptance",
    "label": "card_acceptance",
    "description": "The user asks where or in what circumstances the card can be accepted."
  },
  {
    "id": "card_arrival",
    "label": "card_arrival",
    "description": "The user is waiting for an ordered card and asks where it is."
  },
  {
    "id": "card_delivery_estimate",
    "label": "card_delivery_estimate",
    "description": "The user asks how long card delivery normally takes."
  },
  {
    "id": "card_linking",
    "label": "card_linking",
    "description": "The user wants to link an existing card to an account or app."
  },
  {
    "id": "card_not_working",
    "label": "card_not_working",
    "description": "The user reports that the physical card does not work without specifying a narrower failure."
  },
  {
    "id": "card_payment_fee_charged",
    "label": "card_payment_fee_charged",
    "description": "The user questions a fee charged for paying by card."
  },
  {
    "id": "card_payment_not_recognised",
    "label": "card_payment_not_recognised",
    "description": "The user does not recognize a card payment shown on the account."
  },
  {
    "id": "card_payment_wrong_exchange_rate",
    "label": "card_payment_wrong_exchange_rate",
    "description": "The user disputes the exchange rate applied to a card purchase."
  },
  {
    "id": "card_swallowed",
    "label": "card_swallowed",
    "description": "The user reports that an ATM retained the card."
  },
  {
    "id": "cash_withdrawal_charge",
    "label": "cash_withdrawal_charge",
    "description": "The user questions a fee charged for an ATM cash withdrawal."
  },
  {
    "id": "cash_withdrawal_not_recognised",
    "label": "cash_withdrawal_not_recognised",
    "description": "The user does not recognize a cash withdrawal on the account."
  },
  {
    "id": "change_pin",
    "label": "change_pin",
    "description": "The user wants to change the card PIN."
  },
  {
    "id": "compromised_card",
    "label": "compromised_card",
    "description": "The user suspects the card or its details have been compromised."
  },
  {
    "id": "contactless_not_working",
    "label": "contactless_not_working",
    "description": "The user reports that contactless card payments are failing."
  },
  {
    "id": "country_support",
    "label": "country_support",
    "description": "The user asks whether the banking service is available to residents of a country."
  },
  {
    "id": "declined_card_payment",
    "label": "declined_card_payment",
    "description": "The user reports that a card payment was declined."
  },
  {
    "id": "declined_cash_withdrawal",
    "label": "declined_cash_withdrawal",
    "description": "The user reports that an ATM cash withdrawal was declined."
  },
  {
    "id": "declined_transfer",
    "label": "declined_transfer",
    "description": "The user reports that a money transfer was declined."
  },
  {
    "id": "direct_debit_payment_not_recognised",
    "label": "direct_debit_payment_not_recognised",
    "description": "The user does not recognize a direct debit taken from the account."
  },
  {
    "id": "disposable_card_limits",
    "label": "disposable_card_limits",
    "description": "The user asks about restrictions or limits on disposable virtual cards."
  },
  {
    "id": "edit_personal_details",
    "label": "edit_personal_details",
    "description": "The user wants to update personal account information."
  },
  {
    "id": "exchange_charge",
    "label": "exchange_charge",
    "description": "The user asks about a fee for exchanging currencies."
  },
  {
    "id": "exchange_rate",
    "label": "exchange_rate",
    "description": "The user asks what exchange rate is offered or how it is determined."
  },
  {
    "id": "exchange_via_app",
    "label": "exchange_via_app",
    "description": "The user asks how to exchange currencies inside the app."
  },
  {
    "id": "extra_charge_on_statement",
    "label": "extra_charge_on_statement",
    "description": "The user questions an additional or unexpected charge on a statement."
  },
  {
    "id": "failed_transfer",
    "label": "failed_transfer",
    "description": "The user reports that a transfer failed to complete."
  },
  {
    "id": "fiat_currency_support",
    "label": "fiat_currency_support",
    "description": "The user asks which ordinary currencies the service supports."
  },
  {
    "id": "get_disposable_virtual_card",
    "label": "get_disposable_virtual_card",
    "description": "The user wants to obtain a disposable virtual payment card."
  },
  {
    "id": "get_physical_card",
    "label": "get_physical_card",
    "description": "The user asks how to obtain or receive a physical payment card."
  },
  {
    "id": "getting_spare_card",
    "label": "getting_spare_card",
    "description": "The user wants an additional or spare payment card."
  },
  {
    "id": "getting_virtual_card",
    "label": "getting_virtual_card",
    "description": "The user wants to obtain a reusable virtual payment card."
  },
  {
    "id": "lost_or_stolen_card",
    "label": "lost_or_stolen_card",
    "description": "The user reports a lost or stolen payment card."
  },
  {
    "id": "lost_or_stolen_phone",
    "label": "lost_or_stolen_phone",
    "description": "The user reports losing a phone used to access the banking account."
  },
  {
    "id": "order_physical_card",
    "label": "order_physical_card",
    "description": "The user asks about placing an order for a physical payment card."
  },
  {
    "id": "passcode_forgotten",
    "label": "passcode_forgotten",
    "description": "The user has forgotten the passcode used to access the account."
  },
  {
    "id": "pending_card_payment",
    "label": "pending_card_payment",
    "description": "The user asks why a card payment remains pending."
  },
  {
    "id": "pending_cash_withdrawal",
    "label": "pending_cash_withdrawal",
    "description": "The user asks why a cash withdrawal remains pending."
  },
  {
    "id": "pending_top_up",
    "label": "pending_top_up",
    "description": "The user asks why a balance top-up remains pending."
  },
  {
    "id": "pending_transfer",
    "label": "pending_transfer",
    "description": "The user asks why a money transfer remains pending."
  },
  {
    "id": "pin_blocked",
    "label": "pin_blocked",
    "description": "The user needs help with a blocked card PIN."
  },
  {
    "id": "receiving_money",
    "label": "receiving_money",
    "description": "The user asks how to receive money from another person."
  },
  {
    "id": "Refund_not_showing_up",
    "label": "Refund_not_showing_up",
    "description": "The user is waiting for a refund that has not appeared in the account."
  },
  {
    "id": "request_refund",
    "label": "request_refund",
    "description": "The user wants to request a refund for a transaction."
  },
  {
    "id": "reverted_card_payment?",
    "label": "reverted_card_payment?",
    "description": "The user asks about a card payment that was reversed or reverted."
  },
  {
    "id": "supported_cards_and_currencies",
    "label": "supported_cards_and_currencies",
    "description": "The user asks which payment cards or currencies can be used for card top-ups."
  },
  {
    "id": "terminate_account",
    "label": "terminate_account",
    "description": "The user wants to close the banking account."
  },
  {
    "id": "top_up_by_bank_transfer_charge",
    "label": "top_up_by_bank_transfer_charge",
    "description": "The user asks about charges for funding the account by bank transfer."
  },
  {
    "id": "top_up_by_card_charge",
    "label": "top_up_by_card_charge",
    "description": "The user asks about charges for topping up with a payment card."
  },
  {
    "id": "top_up_by_cash_or_cheque",
    "label": "top_up_by_cash_or_cheque",
    "description": "The user asks whether or how to fund the account with cash or a cheque."
  },
  {
    "id": "top_up_failed",
    "label": "top_up_failed",
    "description": "The user reports a top-up that failed."
  },
  {
    "id": "top_up_limits",
    "label": "top_up_limits",
    "description": "The user asks about minimum or maximum top-up amounts."
  },
  {
    "id": "top_up_reverted",
    "label": "top_up_reverted",
    "description": "The user reports a top-up that was reversed after initiation."
  },
  {
    "id": "topping_up_by_card",
    "label": "topping_up_by_card",
    "description": "The user asks how to add money using a payment card."
  },
  {
    "id": "transaction_charged_twice",
    "label": "transaction_charged_twice",
    "description": "The user reports being charged twice for one transaction."
  },
  {
    "id": "transfer_fee_charged",
    "label": "transfer_fee_charged",
    "description": "The user questions a fee charged for sending a transfer."
  },
  {
    "id": "transfer_into_account",
    "label": "transfer_into_account",
    "description": "The user asks how to transfer money from another bank into this account."
  },
  {
    "id": "transfer_not_received_by_recipient",
    "label": "transfer_not_received_by_recipient",
    "description": "The user reports that the intended recipient has not received a sent transfer."
  },
  {
    "id": "transfer_timing",
    "label": "transfer_timing",
    "description": "The user asks how long a money transfer takes."
  },
  {
    "id": "unable_to_verify_identity",
    "label": "unable_to_verify_identity",
    "description": "The user reports difficulty completing identity verification."
  },
  {
    "id": "verify_my_identity",
    "label": "verify_my_identity",
    "description": "The user asks how to complete identity verification."
  },
  {
    "id": "verify_source_of_funds",
    "label": "verify_source_of_funds",
    "description": "The user asks how to verify where deposited money came from."
  },
  {
    "id": "verify_top_up",
    "label": "verify_top_up",
    "description": "The user asks how to verify a card top-up."
  },
  {
    "id": "virtual_card_not_working",
    "label": "virtual_card_not_working",
    "description": "The user reports a virtual payment card that does not work."
  },
  {
    "id": "visa_or_mastercard",
    "label": "visa_or_mastercard",
    "description": "The user asks whether the supplied card uses Visa or Mastercard."
  },
  {
    "id": "why_verify_identity",
    "label": "why_verify_identity",
    "description": "The user asks why identity verification is required."
  },
  {
    "id": "wrong_amount_of_cash_received",
    "label": "wrong_amount_of_cash_received",
    "description": "The user reports receiving the wrong cash amount from an ATM."
  },
  {
    "id": "wrong_exchange_rate_for_cash_withdrawal",
    "label": "wrong_exchange_rate_for_cash_withdrawal",
    "description": "The user disputes the exchange rate applied to an ATM withdrawal."
  }
]
```

## Label-quality caveats

The [original paper](https://aclanthology.org/2020.nlp4convai-1.5/) motivates fine-grained distinctions within one banking domain; this does not establish that every label boundary is unambiguous. The card has no released rater counts or comprehensive bias analysis. In particular, the names `get_physical_card` and `order_physical_card` do not by themselves prove a reliable semantic distinction; the broad descriptions above preserve that uncertainty. Production escalation thresholds require separate operator data. The mapping is a closed-set intent task, so no out-of-scope gold should be fabricated.
