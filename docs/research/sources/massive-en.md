# MASSIVE, English

## What changed vs. the current plan

Pins locale `en-US` and all 60 intents. Per-locale configurations currently load MASSIVE **1.1**, while the `all` default in the legacy loader refers to 1.0; versions must not be mixed implicitly. Original SLURP IDs, not translated row positions, define groups.

All instructions, paraphrases, candidate descriptions, and proposed conversion rules below are authored for Hunch; they are not quoted upstream prompts. Upstream field names, label names and encodings are cited separately. Retrieval date: **2026-09-19**.

Use the candidate `id` as a stable key; label shuffling must move the target with it. Keep labels and label-derived metadata out of `state`. Natural hard annotations are one-hot **label distributions**, not certainty about an objective event. Preserve native holdouts; derive dev/calibration only from eligible training groups; every paraphrase and related example stays with its base group. These are proposed applications of the record and split rules.

Implementation note: the documented schema uses probability maps; [the current runtime schema](../../../hunch/schema.py) uses arrays aligned to candidate order and integer `hard` indices, Boolean `criteria`, and ordered Score `levels`. The JSON candidate lists below are semantic specifications; serialize them through those existing runtime fields.

## Upstream facts and mapping

Observed upstream revision: [`ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740`](https://huggingface.co/datasets/AmazonScience/massive/tree/ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740), verified through the [repository metadata API](https://huggingface.co/api/datasets/AmazonScience/massive) on 2026-09-19. Pin this revision (or record a deliberate replacement) before ingestion; the research did not download the dataset.

Current engineering compatibility: [the adapter](../../../hunch/data/families.py) loads `mteb/amazon_massive_intent`, config `en`. The [bounded mirror schema](https://datasets-server.huggingface.co/first-rows?dataset=mteb%2Famazon_massive_intent&config=en&split=train) exposes `id`, string `label`, `label_text`, `text`, and `lang`. Observed [mirror revision](https://huggingface.co/datasets/mteb/amazon_massive_intent/tree/940fd47a81eaa7f2cc7b129674d945d618ac38c2): `940fd47a81eaa7f2cc7b129674d945d618ac38c2`. For this mirror, use `text` for utterance and exact string `label` for candidate ID; preserve original `id` for translation grouping. Verify alignment to the original before claiming equivalent splits. The mirror's card license tag differs from the original release; do not treat the adapter's class (now B, following the upstream CC BY 4.0 grant) as upstream clearance—resolve inheritance in W3.

The [card](https://huggingface.co/datasets/AmazonScience/massive) documents `id`, `locale`, `partition`, `scenario`, `intent`, `utt`, `annot_utt`, worker metadata and judgments. The [loader's _INTENTS and configuration declarations](https://huggingface.co/datasets/AmazonScience/massive/blob/main/massive.py) define the exact ordered 60-label encoding below and version behavior. Raw JSON intent values are strings; loader feature values can be integers.

Proposed `state = {"utterance": row.utt, "locale": row.locale}`; require `locale == "en-US"`; Choice K=60, `family = support.intent.massive`. Map integer `intent` using pinned feature names; map raw string labels by exact equality. Do not include `scenario`, `annot_utt`, judgments, or slots in state because these contain annotation information. `source_group = "massive:slurp:" + row.id`, shared across every language and paraphrase. Map original `partition=train/dev/test` to train/dev/test; HF calls dev `validation`. The card reports 11,514/2,033/2,974 English examples.

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

Order matches the pinned loader's integer encoding.

```json
[
  {
    "id": "datetime_query",
    "label": "datetime_query",
    "description": "The user asks for the current date or time."
  },
  {
    "id": "iot_hue_lightchange",
    "label": "iot_hue_lightchange",
    "description": "The user wants to change a connected light's color or lighting mode."
  },
  {
    "id": "transport_ticket",
    "label": "transport_ticket",
    "description": "The user wants to buy or reserve a transport ticket."
  },
  {
    "id": "takeaway_query",
    "label": "takeaway_query",
    "description": "The user asks for information about takeaway food or a takeaway order."
  },
  {
    "id": "qa_stock",
    "label": "qa_stock",
    "description": "The user asks about a stock price or financial market information."
  },
  {
    "id": "general_greet",
    "label": "general_greet",
    "description": "The user greets the assistant."
  },
  {
    "id": "recommendation_events",
    "label": "recommendation_events",
    "description": "The user asks for events or activities to attend."
  },
  {
    "id": "music_dislikeness",
    "label": "music_dislikeness",
    "description": "The user expresses dislike for music or asks to record that preference."
  },
  {
    "id": "iot_wemo_off",
    "label": "iot_wemo_off",
    "description": "The user wants to switch off a connected smart plug or device."
  },
  {
    "id": "cooking_recipe",
    "label": "cooking_recipe",
    "description": "The user asks for a recipe."
  },
  {
    "id": "qa_currency",
    "label": "qa_currency",
    "description": "The user asks for a currency conversion or exchange rate."
  },
  {
    "id": "transport_traffic",
    "label": "transport_traffic",
    "description": "The user asks about traffic conditions."
  },
  {
    "id": "general_quirky",
    "label": "general_quirky",
    "description": "The user makes a conversational or playful request outside the more specific assistant intents."
  },
  {
    "id": "weather_query",
    "label": "weather_query",
    "description": "The user asks about the weather."
  },
  {
    "id": "audio_volume_up",
    "label": "audio_volume_up",
    "description": "The user wants to increase audio volume."
  },
  {
    "id": "email_addcontact",
    "label": "email_addcontact",
    "description": "The user wants to add someone to their contacts."
  },
  {
    "id": "takeaway_order",
    "label": "takeaway_order",
    "description": "The user wants to order takeaway food."
  },
  {
    "id": "email_querycontact",
    "label": "email_querycontact",
    "description": "The user asks for contact details from an address book."
  },
  {
    "id": "iot_hue_lightup",
    "label": "iot_hue_lightup",
    "description": "The user wants to make a connected light brighter."
  },
  {
    "id": "recommendation_locations",
    "label": "recommendation_locations",
    "description": "The user asks for places to visit or nearby venues."
  },
  {
    "id": "play_audiobook",
    "label": "play_audiobook",
    "description": "The user wants to play an audiobook."
  },
  {
    "id": "lists_createoradd",
    "label": "lists_createoradd",
    "description": "The user wants to create a list or add items to it."
  },
  {
    "id": "news_query",
    "label": "news_query",
    "description": "The user asks for news."
  },
  {
    "id": "alarm_query",
    "label": "alarm_query",
    "description": "The user asks about existing alarms."
  },
  {
    "id": "iot_wemo_on",
    "label": "iot_wemo_on",
    "description": "The user wants to switch on a connected smart plug or device."
  },
  {
    "id": "general_joke",
    "label": "general_joke",
    "description": "The user asks for a joke."
  },
  {
    "id": "qa_definition",
    "label": "qa_definition",
    "description": "The user asks for the meaning of a word or expression."
  },
  {
    "id": "social_query",
    "label": "social_query",
    "description": "The user asks to read or check social-media content."
  },
  {
    "id": "music_settings",
    "label": "music_settings",
    "description": "The user wants to change music playback settings such as shuffle or repeat."
  },
  {
    "id": "audio_volume_other",
    "label": "audio_volume_other",
    "description": "The user wants an audio-volume adjustment other than a simple increase, decrease, or mute."
  },
  {
    "id": "calendar_remove",
    "label": "calendar_remove",
    "description": "The user wants to delete a calendar entry."
  },
  {
    "id": "iot_hue_lightdim",
    "label": "iot_hue_lightdim",
    "description": "The user wants to make a connected light dimmer."
  },
  {
    "id": "calendar_query",
    "label": "calendar_query",
    "description": "The user asks about calendar entries."
  },
  {
    "id": "email_sendemail",
    "label": "email_sendemail",
    "description": "The user wants to compose or send an email."
  },
  {
    "id": "iot_cleaning",
    "label": "iot_cleaning",
    "description": "The user wants to control a connected cleaning device."
  },
  {
    "id": "audio_volume_down",
    "label": "audio_volume_down",
    "description": "The user wants to decrease audio volume."
  },
  {
    "id": "play_radio",
    "label": "play_radio",
    "description": "The user wants to play a radio station."
  },
  {
    "id": "cooking_query",
    "label": "cooking_query",
    "description": "The user asks a cooking question other than requesting a complete recipe."
  },
  {
    "id": "datetime_convert",
    "label": "datetime_convert",
    "description": "The user asks to convert a date or time between locations or time zones."
  },
  {
    "id": "qa_maths",
    "label": "qa_maths",
    "description": "The user asks a mathematical question or calculation."
  },
  {
    "id": "iot_hue_lightoff",
    "label": "iot_hue_lightoff",
    "description": "The user wants to switch off connected lights."
  },
  {
    "id": "iot_hue_lighton",
    "label": "iot_hue_lighton",
    "description": "The user wants to switch on connected lights."
  },
  {
    "id": "transport_query",
    "label": "transport_query",
    "description": "The user asks for transport information such as routes or departure times."
  },
  {
    "id": "music_likeness",
    "label": "music_likeness",
    "description": "The user expresses liking for music or asks to record that preference."
  },
  {
    "id": "email_query",
    "label": "email_query",
    "description": "The user asks to check or read email."
  },
  {
    "id": "play_music",
    "label": "play_music",
    "description": "The user wants to play music."
  },
  {
    "id": "audio_volume_mute",
    "label": "audio_volume_mute",
    "description": "The user wants to mute audio."
  },
  {
    "id": "social_post",
    "label": "social_post",
    "description": "The user wants to publish content on social media."
  },
  {
    "id": "alarm_set",
    "label": "alarm_set",
    "description": "The user wants to set an alarm."
  },
  {
    "id": "qa_factoid",
    "label": "qa_factoid",
    "description": "The user asks a factual knowledge question."
  },
  {
    "id": "calendar_set",
    "label": "calendar_set",
    "description": "The user wants to create or schedule a calendar entry."
  },
  {
    "id": "play_game",
    "label": "play_game",
    "description": "The user wants to play a game with the assistant."
  },
  {
    "id": "alarm_remove",
    "label": "alarm_remove",
    "description": "The user wants to delete or cancel an alarm."
  },
  {
    "id": "lists_remove",
    "label": "lists_remove",
    "description": "The user wants to remove a list or items from a list."
  },
  {
    "id": "transport_taxi",
    "label": "transport_taxi",
    "description": "The user wants to arrange a taxi or similar ride."
  },
  {
    "id": "recommendation_movies",
    "label": "recommendation_movies",
    "description": "The user asks for movie recommendations."
  },
  {
    "id": "iot_coffee",
    "label": "iot_coffee",
    "description": "The user wants to control a connected coffee maker."
  },
  {
    "id": "music_query",
    "label": "music_query",
    "description": "The user asks about music, artists, tracks, or what is playing."
  },
  {
    "id": "play_podcasts",
    "label": "play_podcasts",
    "description": "The user wants to play a podcast."
  },
  {
    "id": "lists_query",
    "label": "lists_query",
    "description": "The user asks to retrieve or inspect a list."
  }
]
```

## Label-quality caveats

The [MASSIVE paper](https://arxiv.org/abs/2204.08582) describes localization of SLURP assistant utterances and separate intent/slot judgments. That design motivates grouping all localizations by original ID; translated examples are not independent evidence. Judgment intent scores assess translation quality, not a probability distribution over 60 classes. Report failures on close pairs such as cooking query/recipe or music query/playback; no universally applicable operator policy follows from these assistant labels. English-only sprint results do not establish multilingual performance. The generic quirky intent is a dataset category, not a validated open-world rejection detector.
