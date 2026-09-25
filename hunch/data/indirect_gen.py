"""Pillar-A generator: synthetic indirect-answer interpretation (family `pragmatics.implied_meaning`).

Interface-level transfer test for the held-out Circa family. The label space is the same 8-way candidate set
(indirect yes / no / middle / hedges / other / uninterpretable), the items are synthetic and never derived from
Circa. Labels come from a rule engine over a hand-written topic graph: an answer is about the questioned topic
(direct), about a related topic (contrast: negate / partial / suggests-opposite / orthogonal), or is vague or
evasive. No teacher model, no third-party text, license class A.

    python -m hunch.data.indirect_gen records --n 30000 --seed 7 --out data/synth.jsonl
    python -m hunch.data.indirect_gen prepare --base DIR --out DIR --n 30000 --seed 7
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

from hunch.schema import Candidate, Question, Record, Target, read_jsonl, write_jsonl
from .manifest import manifest_digest

FAMILY = "pragmatics.implied_meaning"
INSTRUCTION = ("In the situation described in `context`, X asks Y the question in `question` and Y replies with "
               "`answer`. What does Y's indirect answer mean?")
CANDS = [
    Candidate("yes", "Yes"),
    Candidate("no", "No"),
    Candidate("middle", "In the middle, neither yes nor no"),
    Candidate("probably_yes", "Probably yes / sometimes yes"),
    Candidate("probably_no", "Probably no"),
    Candidate("conditional", "Yes, subject to some conditions"),
    Candidate("other", "Other"),
    Candidate("unsure", "I am not sure how X will interpret Y’s answer")  # exact label text of the held-out Circa family,
]
LABEL_INDEX = {c.id: i for i, c in enumerate(CANDS)}

# generic hedges: a hedged yes/no does not name the topic, so the same bank applies everywhere
GENERIC = {
    "probably_yes": ["I suppose so.", "More or less.", "I guess so, yes.", "Yeah, kind of.", "Sort of."],
    "probably_no": ["Not really.", "I wouldn't say so.", "Not particularly.", "Rarely, if ever.", "Hardly."],
    "middle": ["Sometimes, it depends.", "On and off.", "Now and then.", "It varies.", "Some days yes, some days no."],
    "conditional": ["Only if it's free.", "Only when someone else suggests it.", "On special occasions, maybe."],
    "other": ["Why do you ask?", "That's a strange question.", "Can we talk about something else?", "I'd rather not say.",
              "You tell me first."],
    "unsure": ["I used to, but things changed.", "Maybe. It's complicated.", "Honestly, I don't know anymore.",
               "I keep going back and forth on that.", "Ask me again next month."],
}

# topic -> questions about it, area for the context line, and first-person answers per level.
# yes / no / middle / conditional are topic-specific; probably_yes / probably_no / other / unsure come from GENERIC.
TOPICS: dict[str, dict] = {
    "coffee": {"area": "habits", "questions": ["Do you drink coffee?", "Are you a coffee drinker?"],
               "yes": ["I have three cups a day.", "I can't get through the morning without it.", "I'm a bit of an addict, honestly."],
               "no": ["Caffeine keeps me up all night.", "I never touch the stuff.", "It gives me headaches, so I stay away."],
               "middle": ["Only when I'm desperate.", "Once in a blue moon.", "I drink it about once a week."],
               "conditional": ["Only if someone else makes it.", "Just when I have to work late.", "Only in winter."]},
    "tea": {"area": "habits", "questions": ["Do you drink tea?", "Are you a tea person?"],
            "yes": ["I have a pot of it every evening.", "I always keep a box in the cupboard.", "It's my favourite thing to wind down with."],
            "no": ["I find it watery and boring.", "I've never liked the taste.", "I only keep it for guests."],
            "middle": ["Occasionally, when I'm ill.", "Maybe once a month.", "Only if there's nothing else."],
            "conditional": ["Only herbal, never black.", "Only when I'm at my parents' place.", "Just with a lot of honey."]},
    "vegetarian": {"area": "eating habits", "questions": ["Are you a vegetarian?", "Do you eat vegetarian food?"],
                   "yes": ["I haven't eaten meat in years.", "I only cook with vegetables and grains.", "Meat isn't part of my diet."],
                   "no": ["I could never give up a good burger.", "I eat meat most days.", "A meal without meat isn't a meal for me."],
                   "middle": ["I eat meat only when I'm a guest.", "Mostly vegetables, with exceptions.", "I try, but not strictly."],
                   "conditional": ["Only when I cook for myself.", "Only at home, never in restaurants."]},
    "meat_eating": {"area": "eating habits", "questions": ["Do you eat meat?", "Is meat part of your diet?"],
                    "yes": ["I had a steak last night.", "I grill most weekends.", "I eat whatever is on the menu."],
                    "no": ["I haven't touched it in years.", "I always order the vegetarian option.", "I cook only plants."],
                    "middle": ["Only when someone else cooks it.", "Rarely, on holidays.", "A few times a month."],
                    "conditional": ["Only if it's ethically sourced.", "Only when I'm travelling."]},
    "fish_eating": {"area": "eating habits", "questions": ["Do you eat fish?", "Do you like seafood?"],
                    "yes": ["I could eat salmon every day.", "We have it twice a week.", "I always order it at restaurants."],
                    "no": ["The smell puts me off.", "I can't stand the texture.", "I skip the seafood section entirely."],
                    "middle": ["I eat it occasionally, mostly in summer.", "Sometimes sushi, nothing else.", "Maybe once a month."],
                    "conditional": ["Only if it's cooked, never raw.", "Only when it's fresh from the market."]},
    "cooking": {"area": "daily routines", "questions": ["Do you cook?", "Do you do your own cooking?"],
                "yes": ["I make dinner from scratch every night.", "My kitchen is my favourite room.", "I plan meals a week ahead."],
                "no": ["Takeout is my whole diet.", "I burn water, honestly.", "I haven't used my stove in months."],
                "middle": ["Simple things only, and rarely.", "I cook when I have the energy.", "A few times a week."],
                "conditional": ["Only when I have guests.", "Only on weekends, never after work."]},
    "spicy_food": {"area": "food preferences", "questions": ["Do you like spicy food?", "Can you handle spicy food?"],
                   "yes": ["I add chilli to almost everything.", "The hotter the better for me.", "I keep three kinds of hot sauce at home."],
                   "no": ["Even mild curry is too much for me.", "My mouth burns at the slightest pepper.", "I always ask for it without spice."],
                   "middle": ["A little heat is fine, nothing extreme.", "Depends on the dish.", "Only in certain cuisines."],
                   "conditional": ["Only when it's balanced with something sweet.", "Only if I'm the one cooking."]},
    "desserts": {"area": "food preferences", "questions": ["Do you like desserts?", "Do you have a sweet tooth?"],
                 "yes": ["I never skip pudding.", "There's always chocolate in my house.", "I could live on cake."],
                 "no": ["I find sweet things cloying.", "I'd rather have cheese.", "I always pass on the sweet menu."],
                 "middle": ["Only at celebrations.", "A bite or two is enough.", "Rarely, and only fruit-based ones."],
                 "conditional": ["Only if it's homemade.", "Only when it comes with coffee."]},
    "running": {"area": "free time", "questions": ["Do you run?", "Are you a runner?"],
                "yes": ["I'm out five mornings a week.", "I did a half marathon last month.", "My trainers live by the door."],
                "no": ["My knees won't allow it.", "I'd rather walk.", "I've never understood the appeal."],
                "middle": ["A few kilometres now and then.", "Mostly in summer.", "On and off for years."],
                "conditional": ["Only when the weather is decent.", "Only with a friend, never alone."]},
    "gym": {"area": "free time", "questions": ["Do you go to the gym?", "Do you work out?"],
            "yes": ["I'm there four times a week.", "I lift before work every day.", "I have a standing session with a trainer."],
            "no": ["I've never had a membership.", "I hate that place.", "Exercise machines bore me to death."],
            "middle": ["I go through phases.", "Once a week, if that.", "I've been meaning to go more."],
            "conditional": ["Only in the quiet hours.", "Only when a class is included."]},
    "cycling": {"area": "free time", "questions": ["Do you cycle?", "Do you ride a bike?"],
                "yes": ["I ride to work every day.", "I do long tours on weekends.", "I sold my second car and bought a better bike."],
                "no": ["Traffic scares me too much.", "I never learned properly.", "I haven't touched a bike in years."],
                "middle": ["Short trips only.", "In summer, occasionally.", "For errands sometimes."],
                "conditional": ["Only if there's a bike lane.", "Only in the park, not on roads."]},
    "swimming": {"area": "free time", "questions": ["Do you swim?", "Do you go swimming?"],
                 "yes": ["I'm in the pool every morning.", "I swim a kilometre before breakfast.", "Open water is my favourite place."],
                 "no": ["I can barely float.", "I don't like getting my hair wet.", "Chlorine makes me ill."],
                 "middle": ["Only on holiday.", "A few times a year.", "I used to, not anymore."],
                 "conditional": ["Only in the sea, never pools.", "Only when it's warm enough."]},
    "football": {"area": "free time", "questions": ["Do you follow football?", "Are you into football?"],
                 "yes": ["I never miss a match.", "I've had a season ticket for years.", "I play five-a-side twice a week."],
                 "no": ["I couldn't name a single player.", "It's just twenty-two people running.", "I switch channels when it comes on."],
                 "middle": ["Only the big tournaments.", "I watch the finals, nothing else.", "Now and then with friends."],
                 "conditional": ["Only if my home team is playing.", "Only in a stadium, never on television."]},
    "music_playing": {"area": "free time", "questions": ["Do you play an instrument?", "Are you musical?"],
                      "yes": ["I've played the piano since I was six.", "There's a guitar in every room.", "I practise most evenings."],
                      "no": ["I'm completely tone deaf.", "I never got past recorder class.", "I can't read a note."],
                      "middle": ["A few chords, badly.", "Only when nobody is listening.", "I gave it up years ago."],
                      "conditional": ["Only when I'm alone in the house.", "Only Christmas carols."]},
    "singing": {"area": "free time", "questions": ["Do you sing?", "Do you like singing?"],
                "yes": ["I'm in a choir.", "I sing in the car constantly.", "I've done it since I was a child."],
                "no": ["Only in the shower, and badly.", "I'd never do it in front of people.", "My voice is a disaster."],
                "middle": ["Along to the radio, nothing more.", "Only when I know the song.", "Occasionally at parties."],
                "conditional": ["Only with a group, never solo.", "Only when I've had a drink."]},
    "reading": {"area": "free time", "questions": ["Do you read?", "Do you read a lot?"],
                "yes": ["I finish a book a week.", "My flat is mostly bookshelves.", "I read before bed every night."],
                "no": ["I haven't finished a book in years.", "I can't sit still that long.", "I only read what work requires."],
                "middle": ["A few books a year.", "On holiday, mostly.", "Only when something is recommended."],
                "conditional": ["Only non-fiction.", "Only on paper, never screens."]},
    "movies": {"area": "free time", "questions": ["Do you watch a lot of films?", "Are you a film person?"],
               "yes": ["I'm at the cinema twice a week.", "I have a list of classics I'm working through.", "I watch something every night."],
               "no": ["I fall asleep in the first twenty minutes.", "I don't have the patience.", "I haven't been to a cinema in years."],
               "middle": ["Mostly on flights.", "A couple a month.", "Only what my partner picks."],
               "conditional": ["Only documentaries.", "Only in the original language."]},
    "gaming": {"area": "free time", "questions": ["Do you play video games?", "Are you a gamer?"],
               "yes": ["I play most evenings.", "I've been gaming since I was a kid.", "I built my own machine for it."],
               "no": ["I've never really tried.", "They make me restless.", "I don't own a console."],
               "middle": ["Only a few mobile games.", "Now and then with friends.", "I used to play a lot, not anymore."],
               "conditional": ["Only with people I know.", "Only on weekends."]},
    "gardening": {"area": "home life", "questions": ["Do you garden?", "Do you keep plants?"],
                  "yes": ["My balcony is full of tomatoes.", "I spend every Sunday in the garden.", "I propagate everything I can get my hands on."],
                  "no": ["I kill everything green.", "I don't have the patience for it.", "My plants are all plastic."],
                  "middle": ["A few herbs on the windowsill.", "I water what survives.", "Only what doesn't need attention."],
                  "conditional": ["Only vegetables, no flowers.", "Only in the summer months."]},
    "dogs": {"area": "home life", "questions": ["Do you have a dog?", "Are you a dog person?"],
             "yes": ["There's a labrador asleep on my sofa.", "I walk him twice a day.", "I've had dogs my whole life."],
             "no": ["I'm allergic to them.", "The mess would drive me mad.", "I prefer animals that ignore me."],
             "middle": ["My family has one, not me.", "I dogsit sometimes.", "Only if it's a small one."],
             "conditional": ["Only if I had a garden.", "Only a rescue, never a puppy."]},
    "cats": {"area": "home life", "questions": ["Do you have a cat?", "Are you a cat person?"],
             "yes": ["Two of them run my household.", "She follows me everywhere.", "I've had cats since I was small."],
             "no": ["They make my eyes itch.", "I can't stand the scratching.", "I'd rather have a dog."],
             "middle": ["My flatmate has one.", "I like other people's cats.", "I'm warming up to them."],
             "conditional": ["Only an outdoor cat.", "Only if it's already house-trained."]},
    "travel": {"area": "plans", "questions": ["Do you travel a lot?", "Do you like travelling?"],
               "yes": ["I'm on a plane every month.", "I've been to forty countries.", "I never stay in one place long."],
               "no": ["Home is where I want to be.", "Flying exhausts me.", "I haven't left the country in years."],
               "middle": ["Once or twice a year.", "Only for work.", "Short trips, nothing far."],
               "conditional": ["Only when I can take the train.", "Only off-season."]},
    "flying": {"area": "plans", "questions": ["Do you like flying?", "Are you comfortable flying?"],
               "yes": ["I love watching the clouds.", "I sleep better on planes than at home.", "It still feels like magic."],
               "no": ["Turbulence terrifies me.", "I take the train whenever I can.", "I grip the armrest the whole way."],
               "middle": ["It's fine, nothing special.", "I don't mind short flights.", "Depends on the airline."],
               "conditional": ["Only if I have a window seat.", "Only on daytime flights."]},
    "driving": {"area": "daily routines", "questions": ["Do you drive?", "Do you have a car?"],
                "yes": ["I drive everywhere.", "I've had my licence since I was eighteen.", "There's a car in front of my house."],
                "no": ["I never passed the test.", "Public transport is enough for me.", "I'd rather not be behind a wheel."],
                "middle": ["Only short distances.", "I share one with my sister.", "A few times a month."],
                "conditional": ["Only in the countryside, never in town.", "Only when it's not rush hour."]},
    "city_living": {"area": "home life", "questions": ["Do you live in a city?", "Are you a city person?"],
                    "yes": ["I love the noise and the crowds.", "Everything I need is within walking distance.", "I could never live in a village."],
                    "no": ["I need silence to sleep.", "I'd take fields over streets any day.", "The countryside is where I belong."],
                    "middle": ["The suburbs suit me.", "I like both, honestly.", "It depends on the week."],
                    "conditional": ["Only a small city.", "Only if I had a garden anyway."]},
    "early_riser": {"area": "daily routines", "questions": ["Are you an early riser?", "Do you get up early?"],
                    "yes": ["I'm up at five every day.", "I get my best work done before sunrise.", "I've never needed an alarm."],
                    "no": ["I come alive after midnight.", "Mornings are my enemy.", "I hit snooze five times."],
                    "middle": ["Depends on the season.", "Somewhere in between.", "Only when I sleep badly."],
                    "conditional": ["Only on workdays.", "Only when I go to bed early."]},
    "employed": {"area": "work", "questions": ["Are you employed?", "Do you have a job?"],
                 "yes": ["I'm a veterinary technician.", "I've been with the same company for eight years.", "I work full time at a hospital."],
                 "no": ["I've been looking for months.", "I'm between jobs right now.", "I left my last position in spring."],
                 "middle": ["Freelance, on and off.", "Part time only.", "I do occasional contracts."],
                 "conditional": ["Only when the project is funded.", "Only remotely."]},
    "student": {"area": "work", "questions": ["Are you a student?", "Do you study?"],
                "yes": ["I'm in my third year of chemistry.", "I'm writing my thesis now.", "I spend all day in the library."],
                "no": ["I finished my studies long ago.", "I've never liked classrooms.", "I went straight into work."],
                "middle": ["I take evening classes.", "Only a single course this term.", "Part time, very slowly."],
                "conditional": ["Only if the course is online.", "Only when work allows."]},
    "remote_work": {"area": "work", "questions": ["Do you work from home?", "Do you have a home office?"],
                    "yes": ["My commute is ten steps.", "I've been fully remote for three years.", "I have a proper desk by the window."],
                    "no": ["I need the office to concentrate.", "My flat is too small for that.", "I go in every day."],
                    "middle": ["Two days a week, maybe.", "It alternates.", "Only when there's too much to do."],
                    "conditional": ["Only on Fridays.", "Only when I don't have meetings."]},
    "languages": {"area": "learning", "questions": ["Are you learning a language?", "Do you speak other languages?"],
                  "yes": ["I practise Spanish every morning.", "I've been taking lessons for a year.", "I'm fluent in three."],
                  "no": ["I only speak one language, badly.", "I've tried and given up twice.", "Languages and I don't get along."],
                  "middle": ["A few phrases, nothing more.", "I can order food, that's it.", "Slowly, very slowly."],
                  "conditional": ["Only when I travel there.", "Only through apps."]},
    "photography": {"area": "free time", "questions": ["Do you do photography?", "Do you take photos?"],
                    "yes": ["I carry a camera everywhere.", "I've been shooting for ten years.", "My weekends are for editing."],
                    "no": ["I never remember to take pictures.", "My phone camera is all I have.", "It doesn't interest me."],
                    "middle": ["On holidays only.", "A few good ones a year.", "Mostly of my cat."],
                    "conditional": ["Only in good light.", "Only landscapes, never people."]},
    "museums": {"area": "free time", "questions": ["Do you like museums?", "Do you visit galleries?"],
                "yes": ["I go to every exhibition in town.", "I could spend a whole day in one room.", "I have a membership card."],
                "no": ["I walk through in ten minutes.", "They make me tired.", "I don't see the point."],
                "middle": ["Only when visitors come.", "Once a year, maybe.", "Only the big ones."],
                "conditional": ["Only if it's free.", "Only with a guide."]},
    "beach": {"area": "holidays", "questions": ["Do you like the beach?", "Do you enjoy the seaside?"],
              "yes": ["I'd move there tomorrow.", "Sand and salt water are my idea of happiness.", "I swim every day when I'm there."],
              "no": ["Sand gets everywhere.", "The sun burns me in minutes.", "I prefer mountains."],
              "middle": ["For an afternoon, not a week.", "It depends on the company.", "I like it in winter."],
              "conditional": ["Only if there's shade.", "Only early in the morning."]},
    "winter": {"area": "preferences", "questions": ["Do you like winter?", "Do you enjoy the cold?"],
               "yes": ["Snow makes me happy.", "I love the long dark evenings.", "I'm at my best in a coat and boots."],
               "no": ["I'm cold from October to April.", "It gets dark far too early.", "I dream of moving somewhere warm."],
               "middle": ["The first snow, then I'm done.", "It has its moments.", "Only December."],
               "conditional": ["Only if I don't have to drive.", "Only with proper heating."]},
    "parties": {"area": "social life", "questions": ["Do you enjoy parties?", "Do you go to parties?"],
                "yes": ["I'm always the last to leave.", "I love a full room.", "I host one every month."],
                "no": ["I leave after twenty minutes.", "Crowds drain me.", "I'd rather stay home."],
                "middle": ["Small ones, not big ones.", "Depends on the guests.", "A few times a year."],
                "conditional": ["Only if I know most people there.", "Only when there's dancing."]},
}

# contrast relations: an answer about the second topic is used for a question about the first.
# negate: yes<->no, probably_yes<->probably_no, middle/conditional unchanged
# suggests_opposite: yes/probably_yes -> probably_no, no/probably_no -> probably_yes, middle -> middle
# partial: anything -> middle (the answer only partly addresses the question)
# orthogonal: anything -> other (the answer is about something unrelated)
CONTRASTS: list[tuple[str, str, str]] = [
    ("vegetarian", "meat_eating", "negate"), ("vegetarian", "fish_eating", "partial"),
    ("coffee", "tea", "negate"), ("tea", "coffee", "negate"),
    ("running", "gym", "partial"), ("gym", "running", "partial"), ("cycling", "running", "partial"),
    ("dogs", "cats", "suggests_opposite"), ("cats", "dogs", "suggests_opposite"),
    ("early_riser", "employed", "orthogonal"), ("reading", "movies", "orthogonal"),
    ("cooking", "spicy_food", "partial"), ("music_playing", "singing", "partial"),
    ("travel", "flying", "partial"), ("student", "employed", "partial"),
    ("city_living", "gardening", "orthogonal"), ("beach", "winter", "orthogonal"),
    ("employed", "student", "partial"), ("remote_work", "employed", "partial"),
    ("languages", "travel", "partial"), ("photography", "museums", "partial"),
    ("football", "running", "partial"),
]

CONTEXTS = [
    "X and Y are getting to know each other.",
    "X wants to know about Y's {area}.",
    "X is curious about Y's {area}.",
    "X is making conversation with Y at a party.",
    "X asks Y a personal question about {area}.",
    "X and Y are chatting during a break.",
    "Y has just moved in next door and X asks about {area}.",
    "X is filling in a profile and asks Y about {area}.",
    "X and Y are on a long train journey together.",
    "X is interviewing Y for a student magazine.",
    "X and Y have just been introduced by a mutual friend.",
    "X is trying to make small talk while they wait.",
    "X is asking around about Y before a trip together.",
    "X and Y are catching up after years apart.",
    "X is curious and asks Y over dinner.",
    "X is comparing notes with Y about {area}.",
]

FLIP = {"yes": "no", "no": "yes", "probably_yes": "probably_no", "probably_no": "probably_yes",
        "middle": "middle", "conditional": "conditional"}
SUGGEST = {"yes": "probably_no", "probably_yes": "probably_no", "no": "probably_yes", "probably_no": "probably_yes",
           "middle": "middle", "conditional": "probably_no"}
PARTIAL = {"yes": "middle", "probably_yes": "middle", "middle": "middle", "conditional": "conditional",
           "no": "probably_no", "probably_no": "probably_no"}


def _answers(topic: str, level: str, specific_only: bool = False) -> list[str]:
    """Answers for a level about `topic`. specific_only keeps phrases that actually name the topic, which is
    required for contrast items: a generic hedge ("Not really.") would read as an answer to the asked question."""
    spec = TOPICS[topic]
    if level in spec:
        return list(spec[level]) if specific_only else list(spec[level]) + GENERIC.get(level, [])
    return [] if specific_only else list(GENERIC[level])


def _label_for(topic: str, answer_topic: str, level: str, relation: str | None) -> str:
    if answer_topic == topic or relation is None:
        return level
    if relation == "negate":
        return FLIP[level]
    if relation == "suggests_opposite":
        return SUGGEST[level]
    if relation == "partial":
        return PARTIAL[level]
    if relation == "orthogonal":
        return "other"
    raise ValueError(relation)


def _record(idx: int, seed: int, topic: str, answer_topic: str, level: str, relation: str | None, question: str,
            answer: str, context: str, split: str) -> Record:
    label = _label_for(topic, answer_topic, level, relation)
    state = {"context": context, "question": question, "answer": answer}
    q = Question(id="meaning", type="choice", instruction=INSTRUCTION, candidates=CANDS)
    hard = LABEL_INDEX[label]
    group = f"gen.implied:{topic}:{answer_topic}:{relation or 'direct'}"
    return Record(id=f"gen.implied:{seed}:{idx}", source="gen.indirect_answer", source_group=group, family=FAMILY,
                  split=split, license_class="A", state=state, question=q,
                  target=Target("label_distribution", "deterministic_truth", [1.0 if i == hard else 0.0 for i in range(8)], hard),
                  meta={"topic": topic, "answer_topic": answer_topic, "level": level, "relation": relation or "direct",
                        "split_role": "synth_dev" if split == "dev" else "train",
                        # "dsf.indirect_gen" is the provenance tag the released checkpoints' data carries; kept so a rebuild is byte-identical
                        "generator": "dsf.indirect_gen", "label_space": "held-out Circa family label space (interface-level)"})


def all_groups() -> list[str]:
    """Every source group the generator can emit (direct per topic, plus each contrast triple)."""
    groups = [f"gen.implied:{topic}:{topic}:direct" for topic in sorted(TOPICS)]
    groups += [f"gen.implied:{a}:{b}:{rel}" for a, b, rel in CONTRASTS]
    return sorted(set(groups))


# a deterministic ~1/14 slice of the groups is held out for the synthetic dev split. Hashing group names against a
# threshold was too coarse here: with only ~57 groups, nothing landed below 0.04 and the dev split came out empty.
DEV_GROUPS = set(all_groups()[::14])


def records(n: int, seed: int, dev_frac: float = 0.04) -> list[Record]:
    """Deterministic sample of n distinct synthetic items; source groups are assigned to train or synth_dev whole."""
    rng = random.Random(seed)
    out: list[Record] = []
    seen: set[tuple[str, str, str]] = set()

    def split_of(group: str) -> str:
        return "dev" if group in DEV_GROUPS else "train"

    attempts = 0
    while len(out) < n and attempts < n * 80:
        attempts += 1
        if rng.random() < 0.5:  # direct answer about the questioned topic
            topic = rng.choice(sorted(TOPICS))
            level = rng.choices(["yes", "no", "middle", "conditional", "probably_yes", "probably_no"],
                                weights=[3, 3, 1.5, 2, 1, 1])[0]
            answer_topic, relation = topic, None
        elif rng.random() < 0.62:  # contrast: answer about a related topic
            topic, answer_topic, relation = rng.choice(CONTRASTS)
            level = rng.choices(["yes", "no", "middle", "conditional"], weights=[3, 3, 1, 2])[0]
        elif rng.random() < 0.5:  # vague / contradictory -> "I am not sure how X will interpret"
            topic = rng.choice(sorted(TOPICS))
            answer_topic, relation, level = topic, None, "unsure"
        else:  # evasive -> "Other"
            topic = rng.choice(sorted(TOPICS))
            answer_topic, relation, level = topic, None, "other"
        question = rng.choice(TOPICS[topic]["questions"])
        specific = relation is not None or level in ("yes", "no", "middle", "conditional")
        answer = rng.choice(_answers(answer_topic, level, specific_only=specific))
        context = rng.choice(CONTEXTS).format(area=TOPICS[topic]["area"])
        key = (question, answer, context)
        if key in seen:
            continue
        seen.add(key)
        group = f"gen.implied:{topic}:{answer_topic}:{relation or 'direct'}"
        out.append(_record(len(out), seed, topic, answer_topic, level, relation, question, answer, context, split_of(group)))
    if len(out) < n:
        raise RuntimeError(f"generator produced only {len(out)} distinct items of {n} requested")
    return out


def prepare(base: Path, out: Path, n: int, seed: int) -> None:
    """Append synthetic train records to a baked base dataset; every other split is copied verbatim."""
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    base_manifest = json.loads((base / "manifest.json").read_text())
    for split in ("train", "dev", "calibration", "heldout"):
        expected = manifest_digest(base_manifest["files"][split])
        if hashlib.sha256((base / f"{split}.jsonl").read_bytes()).hexdigest() != expected:
            raise ValueError(f"{split}: base differs from its manifest")
    synth = records(n, seed)
    train = list(read_jsonl(base / "train.jsonl")) + [r for r in synth if r.split == "train"]
    if len({r.id for r in train}) != len(train):
        raise ValueError("duplicate ids after merge")
    out.mkdir(parents=True)
    write_jsonl(train, out / "train.jsonl")
    # the schema has no "synth_dev" split, so those records carry split="dev" plus meta["split_role"]="synth_dev"
    write_jsonl([r for r in synth if r.meta.get("split_role") == "synth_dev"], out / "synth_dev.jsonl")
    for split in ("dev", "calibration", "heldout"):
        shutil.copyfile(base / f"{split}.jsonl", out / f"{split}.jsonl")
        if hashlib.sha256((base / f"{split}.jsonl").read_bytes()).hexdigest() != hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest():
            raise RuntimeError(f"{split}: copy checksum mismatch")
    receipt = {"base": str(base), "base_manifest_sha256": hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest(),
               "seed": seed, "n_requested": n, "n_generated": len(synth), "n_train": len(train),
               "synthetic_train": sum(1 for r in synth if r.meta.get("split_role") == "train"),
               "synthetic_dev": sum(1 for r in synth if r.meta.get("split_role") == "synth_dev"),
               "label_counts": dict(Counter(CANDS[r.target.hard].id for r in synth)),
               "copied_verbatim": ["dev", "calibration", "heldout"], "sealed_splits_read": [],
               "files": {s: hashlib.sha256((out / f"{s}.jsonl").read_bytes()).hexdigest() for s in ("train", "synth_dev", "dev", "calibration", "heldout")}}
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("label_counts",)}, indent=1), flush=True)
    print("labels:", receipt["label_counts"], flush=True)


def synthetic_only(merged: Path, out: Path) -> None:
    """Write a dataset whose training split is *only* the synthetic family (specialist arm); evaluation splits
    are copied from the merged directory so the specialist is scored on the same items as every other arm."""
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    rows = [r for r in read_jsonl(merged / "train.jsonl") if r.family == FAMILY]
    if not rows:
        raise ValueError("no synthetic training rows found")
    if len({r.id for r in rows}) != len(rows):
        raise ValueError("duplicate synthetic ids")
    out.mkdir(parents=True)
    write_jsonl(rows, out / "train.jsonl")
    for split in ("dev", "calibration", "heldout", "synth_dev"):
        src = merged / f"{split}.jsonl"
        if src.exists():
            shutil.copyfile(src, out / f"{split}.jsonl")
    receipt = {"base": str(merged), "train_count": len(rows), "family": FAMILY,
               "copied": [s for s in ("dev", "calibration", "heldout", "synth_dev") if (out / f"{s}.jsonl").exists()],
               "sealed_splits_read": [], "files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob("*.jsonl"))}}
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=1), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("records")
    r.add_argument("--n", type=int, default=30000)
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("prepare")
    b.add_argument("--base", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--n", type=int, default=30000)
    b.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("synthonly")
    s.add_argument("--base", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.action == "records":
        write_jsonl(records(args.n, args.seed), args.out)
        print(f"wrote {args.n} records to {args.out}")
    elif args.action == "prepare":
        prepare(args.base, args.out, args.n, args.seed)
    else:
        synthetic_only(args.base, args.out)


if __name__ == "__main__":
    main()
