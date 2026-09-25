# CLINC150

## What changed vs. the current plan

Defines the complete 151-way task, including out-of-scope; the Hugging Face encoding places `oos` at **42**, not the last index. Pin the configuration: `plus` is the proposed sprint choice; `small` and `imbalanced` are different experimental distributions.

All instructions, paraphrases, candidate descriptions, and proposed conversion rules below are authored for Hunch; they are not quoted upstream prompts. Upstream field names, label names and encodings are cited separately. Retrieval date: **2026-09-19**.

Use the candidate `id` as a stable key; label shuffling must move the target with it. Keep labels and label-derived metadata out of `state`. Natural hard annotations are one-hot **label distributions**, not certainty about an objective event. Preserve native holdouts; derive dev/calibration only from eligible training groups; every paraphrase and related example stays with its base group. These are proposed applications of the record and split rules.

Implementation note: the documented schema uses probability maps; [the current runtime schema](../../../hunch/schema.py) uses arrays aligned to candidate order and integer `hard` indices, Boolean `criteria`, and ordered Score `levels`. The JSON candidate lists below are semantic specifications; serialize them through those existing runtime fields.

## Upstream facts and mapping

Observed upstream revision: [`155b9c710419136e17307b80d0a13e68cd46b4ec`](https://huggingface.co/datasets/clinc/clinc_oos/tree/155b9c710419136e17307b80d0a13e68cd46b4ec), verified through the [repository metadata API](https://huggingface.co/api/datasets/clinc/clinc_oos) on 2026-09-19. Pin this revision (or record a deliberate replacement) before ingestion; the research did not download the dataset.

The [card and canonical feature metadata](https://huggingface.co/datasets/clinc/clinc_oos/blob/main/README.md) define `text`, integer `intent`, the following ordered label names, and native train/validation/test splits. The [original project](https://github.com/clinc/oos-eval) and [paper](https://aclanthology.org/D19-1131/) define 150 supported intents and a separate out-of-scope evaluation problem.

Proposed `state = {"utterance": row.text}`; `family = support.intent.clinc151`; Choice K=151; map `row.intent` through the pinned configuration's feature names, then to candidate IDs. Validate all names and OOS position before conversion. Group by normalized utterance hash, consolidating duplicates across configurations; train on only one configuration. Author/rater/conversation group identifiers and released annotator vote distributions: **not found**. Keep test untouched; split validation groups into dev/calibration only if that choice is recorded before model selection.

## Exact instruction and ten paraphrases

Every rendered instruction below already includes the required out-of-scope suffix; use it exactly once.

0. Choose the candidate intent that best matches the user's request. If none of the 150 supported intents applies, choose out-of-scope.
1. Which listed intent most accurately describes what the user wants? If none of the 150 supported intents applies, choose out-of-scope.
2. Identify the user's main intent from the candidate descriptions. If none of the 150 supported intents applies, choose out-of-scope.
3. Select the intent that best captures the purpose of this message. If none of the 150 supported intents applies, choose out-of-scope.
4. Match the user's request to the most appropriate listed intent. If none of the 150 supported intents applies, choose out-of-scope.
5. What is the user trying to do, among the candidate intents? If none of the 150 supported intents applies, choose out-of-scope.
6. Assign this message to its best-fitting intent. If none of the 150 supported intents applies, choose out-of-scope.
7. Determine which candidate intent the message expresses. If none of the 150 supported intents applies, choose out-of-scope.
8. Choose the intent whose description most closely fits this request. If none of the 150 supported intents applies, choose out-of-scope.
9. Classify the user's goal using the available intent labels. If none of the 150 supported intents applies, choose out-of-scope.
10. Which candidate best describes the action or information the user is seeking? If none of the 150 supported intents applies, choose out-of-scope.

## Candidate list

Order is the upstream integer encoding; `oos` is displayed as out-of-scope if a human-readable rename is needed.

```json
[
  {
    "id": "restaurant_reviews",
    "label": "restaurant_reviews",
    "description": "The user asks for reviews or ratings of a restaurant."
  },
  {
    "id": "nutrition_info",
    "label": "nutrition_info",
    "description": "The user asks about a food's nutrients or nutritional composition."
  },
  {
    "id": "account_blocked",
    "label": "account_blocked",
    "description": "The user reports that access to a financial account is blocked."
  },
  {
    "id": "oil_change_how",
    "label": "oil_change_how",
    "description": "The user asks how to change a vehicle's engine oil."
  },
  {
    "id": "time",
    "label": "time",
    "description": "The user asks for the current time."
  },
  {
    "id": "weather",
    "label": "weather",
    "description": "The user asks about weather conditions or a forecast."
  },
  {
    "id": "redeem_rewards",
    "label": "redeem_rewards",
    "description": "The user wants to redeem accumulated rewards."
  },
  {
    "id": "interest_rate",
    "label": "interest_rate",
    "description": "The user asks about an account's interest rate."
  },
  {
    "id": "gas_type",
    "label": "gas_type",
    "description": "The user asks which type of fuel a vehicle requires."
  },
  {
    "id": "accept_reservations",
    "label": "accept_reservations",
    "description": "The user asks whether a restaurant accepts reservations."
  },
  {
    "id": "smart_home",
    "label": "smart_home",
    "description": "The user wants to control a connected home device."
  },
  {
    "id": "user_name",
    "label": "user_name",
    "description": "The user asks what name the assistant has stored for them."
  },
  {
    "id": "report_lost_card",
    "label": "report_lost_card",
    "description": "The user wants to report a lost payment card."
  },
  {
    "id": "repeat",
    "label": "repeat",
    "description": "The user asks the assistant to repeat its previous message."
  },
  {
    "id": "whisper_mode",
    "label": "whisper_mode",
    "description": "The user wants the assistant to speak in whisper mode."
  },
  {
    "id": "what_are_your_hobbies",
    "label": "what_are_your_hobbies",
    "description": "The user asks what hobbies or interests the assistant has."
  },
  {
    "id": "order",
    "label": "order",
    "description": "The user wants to place an order for goods or food."
  },
  {
    "id": "jump_start",
    "label": "jump_start",
    "description": "The user asks how to jump-start a vehicle."
  },
  {
    "id": "schedule_meeting",
    "label": "schedule_meeting",
    "description": "The user wants to arrange a meeting."
  },
  {
    "id": "meeting_schedule",
    "label": "meeting_schedule",
    "description": "The user asks when a meeting is scheduled."
  },
  {
    "id": "freeze_account",
    "label": "freeze_account",
    "description": "The user wants to freeze a financial account."
  },
  {
    "id": "what_song",
    "label": "what_song",
    "description": "The user asks to identify a song being played or described."
  },
  {
    "id": "meaning_of_life",
    "label": "meaning_of_life",
    "description": "The user asks about the meaning or purpose of life."
  },
  {
    "id": "restaurant_reservation",
    "label": "restaurant_reservation",
    "description": "The user wants to reserve a table at a restaurant."
  },
  {
    "id": "traffic",
    "label": "traffic",
    "description": "The user asks about road traffic conditions."
  },
  {
    "id": "make_call",
    "label": "make_call",
    "description": "The user asks the assistant to place a telephone call."
  },
  {
    "id": "text",
    "label": "text",
    "description": "The user asks the assistant to send a text message."
  },
  {
    "id": "bill_balance",
    "label": "bill_balance",
    "description": "The user asks how much is owed on a bill."
  },
  {
    "id": "improve_credit_score",
    "label": "improve_credit_score",
    "description": "The user asks how to improve their credit score."
  },
  {
    "id": "change_language",
    "label": "change_language",
    "description": "The user wants to change the assistant's language."
  },
  {
    "id": "no",
    "label": "no",
    "description": "The user expresses a negative answer or disagreement."
  },
  {
    "id": "measurement_conversion",
    "label": "measurement_conversion",
    "description": "The user asks to convert between measurement units."
  },
  {
    "id": "timer",
    "label": "timer",
    "description": "The user wants to set or manage a countdown timer."
  },
  {
    "id": "flip_coin",
    "label": "flip_coin",
    "description": "The user asks for a random coin toss."
  },
  {
    "id": "do_you_have_pets",
    "label": "do_you_have_pets",
    "description": "The user asks whether the assistant has pets."
  },
  {
    "id": "balance",
    "label": "balance",
    "description": "The user asks for the balance of a financial account."
  },
  {
    "id": "tell_joke",
    "label": "tell_joke",
    "description": "The user asks the assistant to tell a joke."
  },
  {
    "id": "last_maintenance",
    "label": "last_maintenance",
    "description": "The user asks when a vehicle was last serviced."
  },
  {
    "id": "exchange_rate",
    "label": "exchange_rate",
    "description": "The user asks for a currency exchange rate."
  },
  {
    "id": "uber",
    "label": "uber",
    "description": "The user wants to arrange a ride through a ride-hailing service."
  },
  {
    "id": "car_rental",
    "label": "car_rental",
    "description": "The user wants information about renting a vehicle."
  },
  {
    "id": "credit_limit",
    "label": "credit_limit",
    "description": "The user asks for the current credit limit on an account."
  },
  {
    "id": "oos",
    "label": "oos",
    "description": "The request falls outside all 150 supported intent definitions."
  },
  {
    "id": "shopping_list",
    "label": "shopping_list",
    "description": "The user asks to view or retrieve a shopping list."
  },
  {
    "id": "expiration_date",
    "label": "expiration_date",
    "description": "The user asks when a payment card expires."
  },
  {
    "id": "routing",
    "label": "routing",
    "description": "The user asks for a bank account's routing number."
  },
  {
    "id": "meal_suggestion",
    "label": "meal_suggestion",
    "description": "The user asks for an idea of what to eat."
  },
  {
    "id": "tire_change",
    "label": "tire_change",
    "description": "The user asks how to replace or change a vehicle tire."
  },
  {
    "id": "todo_list",
    "label": "todo_list",
    "description": "The user asks to view a task or to-do list."
  },
  {
    "id": "card_declined",
    "label": "card_declined",
    "description": "The user reports that a card transaction was declined."
  },
  {
    "id": "rewards_balance",
    "label": "rewards_balance",
    "description": "The user asks how many rewards or points they have."
  },
  {
    "id": "change_accent",
    "label": "change_accent",
    "description": "The user wants to change the assistant's spoken accent."
  },
  {
    "id": "vaccines",
    "label": "vaccines",
    "description": "The user asks which vaccinations are needed for travel."
  },
  {
    "id": "reminder_update",
    "label": "reminder_update",
    "description": "The user wants to change an existing reminder."
  },
  {
    "id": "food_last",
    "label": "food_last",
    "description": "The user asks how long food will keep before spoiling."
  },
  {
    "id": "change_ai_name",
    "label": "change_ai_name",
    "description": "The user wants to change what the assistant is called."
  },
  {
    "id": "bill_due",
    "label": "bill_due",
    "description": "The user asks when a bill must be paid."
  },
  {
    "id": "who_do_you_work_for",
    "label": "who_do_you_work_for",
    "description": "The user asks which organization employs or operates the assistant."
  },
  {
    "id": "share_location",
    "label": "share_location",
    "description": "The user wants to share their location with someone."
  },
  {
    "id": "international_visa",
    "label": "international_visa",
    "description": "The user asks whether travel requires a visa."
  },
  {
    "id": "calendar",
    "label": "calendar",
    "description": "The user asks to view events on a calendar."
  },
  {
    "id": "translate",
    "label": "translate",
    "description": "The user asks to translate text into another language."
  },
  {
    "id": "carry_on",
    "label": "carry_on",
    "description": "The user asks about airline carry-on luggage rules."
  },
  {
    "id": "book_flight",
    "label": "book_flight",
    "description": "The user wants to book an airline flight."
  },
  {
    "id": "insurance_change",
    "label": "insurance_change",
    "description": "The user wants to change an insurance policy."
  },
  {
    "id": "todo_list_update",
    "label": "todo_list_update",
    "description": "The user wants to add, remove, or edit items on a to-do list."
  },
  {
    "id": "timezone",
    "label": "timezone",
    "description": "The user asks about a location's time zone."
  },
  {
    "id": "cancel_reservation",
    "label": "cancel_reservation",
    "description": "The user wants to cancel an existing reservation."
  },
  {
    "id": "transactions",
    "label": "transactions",
    "description": "The user asks to see account transactions."
  },
  {
    "id": "credit_score",
    "label": "credit_score",
    "description": "The user asks for their credit score."
  },
  {
    "id": "report_fraud",
    "label": "report_fraud",
    "description": "The user wants to report suspected financial fraud."
  },
  {
    "id": "spending_history",
    "label": "spending_history",
    "description": "The user asks for a summary or history of their spending."
  },
  {
    "id": "directions",
    "label": "directions",
    "description": "The user asks for navigation directions."
  },
  {
    "id": "spelling",
    "label": "spelling",
    "description": "The user asks how a word is spelled."
  },
  {
    "id": "insurance",
    "label": "insurance",
    "description": "The user asks about insurance coverage or policy information."
  },
  {
    "id": "what_is_your_name",
    "label": "what_is_your_name",
    "description": "The user asks the assistant's name."
  },
  {
    "id": "reminder",
    "label": "reminder",
    "description": "The user wants to create or retrieve a reminder."
  },
  {
    "id": "where_are_you_from",
    "label": "where_are_you_from",
    "description": "The user asks where the assistant comes from."
  },
  {
    "id": "distance",
    "label": "distance",
    "description": "The user asks how far apart locations are."
  },
  {
    "id": "payday",
    "label": "payday",
    "description": "The user asks when wages or salary will be paid."
  },
  {
    "id": "flight_status",
    "label": "flight_status",
    "description": "The user asks whether a flight is on time or otherwise its current status."
  },
  {
    "id": "find_phone",
    "label": "find_phone",
    "description": "The user wants help locating their phone."
  },
  {
    "id": "greeting",
    "label": "greeting",
    "description": "The user greets the assistant."
  },
  {
    "id": "alarm",
    "label": "alarm",
    "description": "The user wants to set or manage an alarm."
  },
  {
    "id": "order_status",
    "label": "order_status",
    "description": "The user asks about progress or delivery status of an order."
  },
  {
    "id": "confirm_reservation",
    "label": "confirm_reservation",
    "description": "The user wants to confirm an existing reservation."
  },
  {
    "id": "cook_time",
    "label": "cook_time",
    "description": "The user asks how long a food should be cooked."
  },
  {
    "id": "damaged_card",
    "label": "damaged_card",
    "description": "The user reports a damaged payment card."
  },
  {
    "id": "reset_settings",
    "label": "reset_settings",
    "description": "The user wants to restore the assistant or device settings to defaults."
  },
  {
    "id": "pin_change",
    "label": "pin_change",
    "description": "The user wants to change a payment card PIN."
  },
  {
    "id": "replacement_card_duration",
    "label": "replacement_card_duration",
    "description": "The user asks how long a replacement card will take to arrive."
  },
  {
    "id": "new_card",
    "label": "new_card",
    "description": "The user wants to obtain a new payment card."
  },
  {
    "id": "roll_dice",
    "label": "roll_dice",
    "description": "The user asks the assistant to simulate rolling dice."
  },
  {
    "id": "income",
    "label": "income",
    "description": "The user asks about their income or earnings."
  },
  {
    "id": "taxes",
    "label": "taxes",
    "description": "The user asks about taxes or tax obligations."
  },
  {
    "id": "date",
    "label": "date",
    "description": "The user asks for the current date or a calendar date."
  },
  {
    "id": "who_made_you",
    "label": "who_made_you",
    "description": "The user asks who created the assistant."
  },
  {
    "id": "pto_request",
    "label": "pto_request",
    "description": "The user wants to request paid time off."
  },
  {
    "id": "tire_pressure",
    "label": "tire_pressure",
    "description": "The user asks about a vehicle's tire pressure."
  },
  {
    "id": "how_old_are_you",
    "label": "how_old_are_you",
    "description": "The user asks the assistant's age."
  },
  {
    "id": "rollover_401k",
    "label": "rollover_401k",
    "description": "The user asks how to transfer or roll over a retirement savings account."
  },
  {
    "id": "pto_request_status",
    "label": "pto_request_status",
    "description": "The user asks whether a paid-time-off request has been approved."
  },
  {
    "id": "how_busy",
    "label": "how_busy",
    "description": "The user asks how busy a place is."
  },
  {
    "id": "application_status",
    "label": "application_status",
    "description": "The user asks about the status of an application."
  },
  {
    "id": "recipe",
    "label": "recipe",
    "description": "The user asks for cooking instructions for a dish."
  },
  {
    "id": "calendar_update",
    "label": "calendar_update",
    "description": "The user wants to add, change, or remove a calendar event."
  },
  {
    "id": "play_music",
    "label": "play_music",
    "description": "The user asks the assistant to play music."
  },
  {
    "id": "yes",
    "label": "yes",
    "description": "The user expresses agreement or an affirmative answer."
  },
  {
    "id": "direct_deposit",
    "label": "direct_deposit",
    "description": "The user asks about setting up or receiving direct deposits."
  },
  {
    "id": "credit_limit_change",
    "label": "credit_limit_change",
    "description": "The user wants to change a credit limit."
  },
  {
    "id": "gas",
    "label": "gas",
    "description": "The user asks about fuel availability, prices, or refueling."
  },
  {
    "id": "pay_bill",
    "label": "pay_bill",
    "description": "The user wants to pay a bill."
  },
  {
    "id": "ingredients_list",
    "label": "ingredients_list",
    "description": "The user asks which ingredients a dish or recipe contains."
  },
  {
    "id": "lost_luggage",
    "label": "lost_luggage",
    "description": "The user seeks help with luggage lost during travel."
  },
  {
    "id": "goodbye",
    "label": "goodbye",
    "description": "The user ends the conversation or says farewell."
  },
  {
    "id": "what_can_i_ask_you",
    "label": "what_can_i_ask_you",
    "description": "The user asks which tasks or questions the assistant can handle."
  },
  {
    "id": "book_hotel",
    "label": "book_hotel",
    "description": "The user wants to reserve a hotel room."
  },
  {
    "id": "are_you_a_bot",
    "label": "are_you_a_bot",
    "description": "The user asks whether the assistant is a bot or a person."
  },
  {
    "id": "next_song",
    "label": "next_song",
    "description": "The user asks to skip to the next song."
  },
  {
    "id": "change_speed",
    "label": "change_speed",
    "description": "The user wants to change how quickly the assistant speaks."
  },
  {
    "id": "plug_type",
    "label": "plug_type",
    "description": "The user asks which electrical plug or adapter a destination uses."
  },
  {
    "id": "maybe",
    "label": "maybe",
    "description": "The user expresses uncertainty or a tentative answer."
  },
  {
    "id": "w2",
    "label": "w2",
    "description": "The user asks about obtaining or using a W-2 wage and tax form."
  },
  {
    "id": "oil_change_when",
    "label": "oil_change_when",
    "description": "The user asks when a vehicle needs an oil change."
  },
  {
    "id": "thank_you",
    "label": "thank_you",
    "description": "The user thanks the assistant."
  },
  {
    "id": "shopping_list_update",
    "label": "shopping_list_update",
    "description": "The user wants to add, remove, or edit items on a shopping list."
  },
  {
    "id": "pto_balance",
    "label": "pto_balance",
    "description": "The user asks how much paid time off remains."
  },
  {
    "id": "order_checks",
    "label": "order_checks",
    "description": "The user wants to order bank checks."
  },
  {
    "id": "travel_alert",
    "label": "travel_alert",
    "description": "The user asks about travel warnings or advisories."
  },
  {
    "id": "fun_fact",
    "label": "fun_fact",
    "description": "The user asks for an interesting fact."
  },
  {
    "id": "sync_device",
    "label": "sync_device",
    "description": "The user wants to synchronize or connect a device."
  },
  {
    "id": "schedule_maintenance",
    "label": "schedule_maintenance",
    "description": "The user wants to arrange vehicle servicing."
  },
  {
    "id": "apr",
    "label": "apr",
    "description": "The user asks about an annual percentage rate for borrowing."
  },
  {
    "id": "transfer",
    "label": "transfer",
    "description": "The user wants to transfer money between accounts or recipients."
  },
  {
    "id": "ingredient_substitution",
    "label": "ingredient_substitution",
    "description": "The user asks what can replace an ingredient in a recipe."
  },
  {
    "id": "calories",
    "label": "calories",
    "description": "The user asks for the calorie content of food."
  },
  {
    "id": "current_location",
    "label": "current_location",
    "description": "The user asks where they currently are."
  },
  {
    "id": "international_fees",
    "label": "international_fees",
    "description": "The user asks about fees for using financial services internationally."
  },
  {
    "id": "calculator",
    "label": "calculator",
    "description": "The user asks the assistant to calculate a mathematical expression."
  },
  {
    "id": "definition",
    "label": "definition",
    "description": "The user asks for the meaning of a word or term."
  },
  {
    "id": "next_holiday",
    "label": "next_holiday",
    "description": "The user asks when the next holiday occurs."
  },
  {
    "id": "update_playlist",
    "label": "update_playlist",
    "description": "The user wants to add, remove, or change songs in a playlist."
  },
  {
    "id": "mpg",
    "label": "mpg",
    "description": "The user asks about a vehicle's fuel economy."
  },
  {
    "id": "min_payment",
    "label": "min_payment",
    "description": "The user asks for the minimum payment due on a credit account."
  },
  {
    "id": "change_user_name",
    "label": "change_user_name",
    "description": "The user wants to change their stored name."
  },
  {
    "id": "restaurant_suggestion",
    "label": "restaurant_suggestion",
    "description": "The user asks for a restaurant recommendation."
  },
  {
    "id": "travel_notification",
    "label": "travel_notification",
    "description": "The user wants to notify a financial provider of upcoming travel."
  },
  {
    "id": "cancel",
    "label": "cancel",
    "description": "The user asks to cancel the current action or request."
  },
  {
    "id": "pto_used",
    "label": "pto_used",
    "description": "The user asks how much paid time off they have already used."
  },
  {
    "id": "travel_suggestion",
    "label": "travel_suggestion",
    "description": "The user asks for travel destinations or trip recommendations."
  },
  {
    "id": "change_volume",
    "label": "change_volume",
    "description": "The user wants to change the assistant's audio volume."
  }
]
```

## Label-quality caveats

The [paper](https://aclanthology.org/D19-1131/) reports that detecting out-of-scope requests remains difficult even when supported-intent accuracy is high. OOS is a heterogeneous collection, not a coherent intent; report OOS recall and false rejection separately from 151-way accuracy. Crowdsourced single-turn requests do not reproduce customer conversation context, and responses such as yes/no/maybe are context-dependent in deployment. Similar labels such as nutrition/calories and restaurant reviews/recommendations require the candidate descriptions during audit; no finer authoritative boundary manual was found in the inspected release. The authored descriptions must not be treated as extra observed gold.
