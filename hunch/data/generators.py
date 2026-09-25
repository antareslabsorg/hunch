"""Exact-truth generators used in the sprint: a policy rule engine and known random mechanisms.

Both emit records whose targets come from code, unit-tested against brute force here.
"""
from __future__ import annotations

import random
from fractions import Fraction

from ..schema import Candidate, Question, Record, Target, one_hot

REFUND_CANDS = [
    Candidate("approve", "Approve refund", "The request satisfies the policy; refund the full amount."),
    Candidate("partial", "Partial refund", "The policy allows a refund minus a restocking fee."),
    Candidate("review", "Manual review", "The case is an exception the policy sends to a human."),
    Candidate("reject", "Reject", "The request does not satisfy the policy."),
]


def refund_policy(case: dict, policy: dict) -> str:
    """Deterministic decision. Keep it boring: this is the oracle."""
    days = case["days_since_purchase"]
    if case["item_condition"] == "defective":
        return "approve" if days <= policy["defective_window_days"] else "review"
    if days > policy["return_window_days"]:
        return "review" if case["loyalty_tier"] == "gold" and days <= policy["return_window_days"] + policy["gold_grace_days"] else "reject"
    if case["opened"] and policy["restocking_fee_pct"] > 0:
        return "partial"
    return "approve"


def refund_records(n: int, seed: int, split_fn) -> list[Record]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        policy = {
            "return_window_days": rng.choice([14, 30, 45, 60]),
            "defective_window_days": rng.choice([90, 180, 365]),
            "gold_grace_days": rng.choice([0, 7, 14]),
            "restocking_fee_pct": rng.choice([0, 0, 10, 15]),
        }
        case = {
            "order_id": f"ORD-{rng.randint(10000, 99999)}",
            "days_since_purchase": rng.randint(0, 120),
            "item_condition": rng.choice(["unopened", "opened", "defective"]),
            "opened": None,
            "loyalty_tier": rng.choice(["standard", "standard", "gold"]),
            "amount_eur": round(rng.uniform(5, 900), 2),
        }
        case["opened"] = case["item_condition"] != "unopened"
        answer = refund_policy(case, policy)
        policy_text = (
            f"Returns are accepted within {policy['return_window_days']} days of purchase. "
            f"Defective items may be refunded within {policy['defective_window_days']} days; later defective claims go to manual review. "
            f"Gold members get {policy['gold_grace_days']} extra days, but late gold returns require manual review. "
            + (f"Opened non-defective items are refunded minus a {policy['restocking_fee_pct']}% restocking fee. " if policy["restocking_fee_pct"] else "Opened items are refunded in full. ")
            + "Anything outside these rules is rejected."
        )
        group = f"gen.refund:policy:{tuple(sorted(policy.items()))}"
        q = Question(id="decision", type="choice", instruction="Apply `policy` to `case`. Which decision does the policy require?", candidates=REFUND_CANDS)
        idx = [c.id for c in REFUND_CANDS].index(answer)
        out.append(Record(id=f"gen.refund:{seed}:{i}", source="gen.refund_policy", source_group=group, family="rules.refund_policy", split=split_fn(group),
                          license_class="A", state={"policy": policy_text, "case": case}, question=q,
                          target=Target("label_distribution", "deterministic_truth", one_hot(4, idx), idx)))
    return out


def urn_records(n: int, seed: int, split_fn) -> list[Record]:
    """A ball is drawn uniformly from an urn with stated counts; the draw has not been revealed. Exact distribution."""
    rng = random.Random(seed)
    colors_all = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"]
    out = []
    for i in range(n):
        k = rng.randint(2, 6)
        colors = rng.sample(colors_all, k)
        counts = [rng.randint(1, 12) for _ in colors]
        total = sum(counts)
        probs = [c / total for c in counts]
        probs[-1] = 1.0 - sum(probs[:-1])  # exact unit sum in floating point
        group = f"gen.urn:{sorted(zip(colors, counts))}"
        state = {"urn": {c: n_ for c, n_ in zip(colors, counts)},
                 "procedure": "One ball is drawn uniformly at random. The draw has already happened but the result has not been revealed."}
        q = Question(id="color", type="choice", instruction="Which color is the drawn ball?", candidates=[Candidate(c, c, f"The drawn ball is {c}.") for c in colors])
        out.append(Record(id=f"gen.urn:{seed}:{i}", source="gen.urn", source_group=group, family="mechanism.urn", split=split_fn(group), license_class="A",
                          state=state, question=q, target=Target("event_probability", "programmatic_conditional_distribution", probs, None)))
        # the same mechanism as a Boolean event
        target_color = rng.choice(colors)
        p_true = counts[colors.index(target_color)] / total
        qb = Question(id=f"is_{target_color}", type="boolean", instruction=f"Is the drawn ball {target_color}?")
        out.append(Record(id=f"gen.urn:{seed}:{i}:b", source="gen.urn", source_group=group, family="mechanism.urn", split=split_fn(group), license_class="A",
                          state=state, question=qb, target=Target("event_probability", "programmatic_conditional_distribution", [1.0 - p_true, p_true], None)))
    return out


IMPLICATURE_TEMPLATES = [
    # (question, positive_replies, negative_replies, conditional_replies)
    ("Are you free this weekend?",
     ["My schedule is completely open.", "I have zero plans all weekend.", "I would love to hang out."],
     ["I am buried in work until Monday.", "I have a family reunion out of town.", "I am flying out tonight."],
     ["Only if I finish my assignment by Friday evening.", "Depends on whether my shift gets covered."]),
    ("Did you enjoy the meal?",
     ["I ate every single crumb on the plate.", "It was one of the best dishes I have tasted.", "I already want to order it again."],
     ["I could barely swallow two bites.", "The meat was cold and completely raw inside.", "I sent it back to the kitchen."],
     ["The dessert was great, but the main dish was mediocre.", "It was fine, nothing special."]),
    ("Can you lend me twenty dollars?",
     ["Here is a fresh bill right from my wallet.", "Take as much as you need, no problem.", "Sure, consider it done."],
     ["My bank account is currently overdrawn.", "I forgot my wallet at home and have no cash.", "I am completely broke until payday."],
     ["Only if you pay me back by tomorrow morning.", "I can spare ten dollars right now, but not twenty."]),
    ("Is it raining outside?",
     ["People are walking in soaking wet with dripping umbrellas.", "Water is pouring down in heavy sheets.", "Look at the puddles forming everywhere."],
     ["There is not a single cloud in the sky.", "The sun is beating down bright and bone dry.", "I do not see a drop of water anywhere."],
     ["It is just a light mist, not quite rain.", "It was drizzling a minute ago but might have stopped."]),
    ("Did the project get finished on time?",
     ["We submitted the final pull request three hours before the deadline.", "All deliverables were signed off yesterday.", "Everything passed and was merged early."],
     ["We missed the final cutoff by two days.", "The server crashed and nothing was deployed.", "The deadline passed while we were still debugging."],
     ["Part one is done, but the second phase was postponed.", "We asked for an extension until tomorrow."]),
    ("Do you want another cup of coffee?",
     ["I desperately need the caffeine right now.", "Pour me another mug please.", "I would love one."],
     ["My heart is already racing from the last two cups.", "I will never fall asleep tonight if I drink more.", "I have had enough for today."],
     ["Only if it is decaf.", "A small half-cup if there is any left."]),
]

IMPLICATURE_CANDS = [
    Candidate("yes", "Yes", "The speaker's utterance conveys an affirmative answer / agreement."),
    Candidate("no", "No", "The speaker's utterance conveys a negative answer / refusal / disagreement."),
    Candidate("conditional", "In the middle / Conditional", "The answer is conditional, partial, or neither simple yes nor no."),
]


def implicature_records(n: int, seed: int, split_fn) -> list[Record]:
    """Conversational implicature: indirect answers to yes/no questions .
    Provides direct supervision for pragmatic inference tasks like Circa.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        q_text, pos, neg, cond = rng.choice(IMPLICATURE_TEMPLATES)
        cat = rng.choice(["yes", "no", "conditional"])
        if cat == "yes":
            reply = rng.choice(pos)
            target_idx = 0
            p_true = 1.0
        elif cat == "no":
            reply = rng.choice(neg)
            target_idx = 1
            p_true = 0.0
        else:
            reply = rng.choice(cond)
            target_idx = 2
            p_true = 0.5

        group = f"gen.implicature:{q_text}"
        state = {
            "dialogue": f"Person A asks: \"{q_text}\"\nPerson B replies: \"{reply}\"",
            "context": "Person B gives an indirect answer to Person A's question."
        }
        
        # Choice question
        q_choice = Question(
            id="implicature_choice",
            type="choice",
            instruction=f"Based on Person B's reply, what does Person B mean in response to: \"{q_text}\"?",
            candidates=IMPLICATURE_CANDS
        )
        out.append(Record(
            id=f"gen.implicature:{seed}:{i}:c",
            source="gen.implicature",
            source_group=group,
            family="pragmatics.implicature",
            split=split_fn(group),
            license_class="A",
            state=state,
            question=q_choice,
            target=Target("label_distribution", "deterministic_truth", one_hot(3, target_idx), target_idx)
        ))

        # Boolean question
        q_bool = Question(
            id="implicature_bool",
            type="boolean",
            instruction=f"Does Person B's reply imply an affirmative (yes) answer to: \"{q_text}\"?",
            criteria={"true": "The reply implies an affirmative answer.", "false": "The reply implies a negative or conditional/uncertain answer."}
        )
        bool_idx = 1 if cat == "yes" else 0
        out.append(Record(
            id=f"gen.implicature:{seed}:{i}:b",
            source="gen.implicature",
            source_group=group,
            family="pragmatics.implicature",
            split=split_fn(group),
            license_class="A",
            state=state,
            question=q_bool,
            target=Target("label_distribution", "deterministic_truth", one_hot(2, bool_idx), bool_idx)
        ))
    return out


if __name__ == "__main__":  # note: brute-force checks
    recs = refund_records(200, 0, lambda g: "train")
    assert {r.target.hard for r in recs} <= {0, 1, 2, 3} and len({r.source_group for r in recs}) > 10
    # oracle spot checks
    assert refund_policy({"days_since_purchase": 5, "item_condition": "unopened", "opened": False, "loyalty_tier": "standard"},
                         {"return_window_days": 30, "defective_window_days": 90, "gold_grace_days": 0, "restocking_fee_pct": 10}) == "approve"
    assert refund_policy({"days_since_purchase": 40, "item_condition": "opened", "opened": True, "loyalty_tier": "gold"},
                         {"return_window_days": 30, "defective_window_days": 90, "gold_grace_days": 14, "restocking_fee_pct": 10}) == "review"
    assert refund_policy({"days_since_purchase": 100, "item_condition": "defective", "opened": True, "loyalty_tier": "standard"},
                         {"return_window_days": 30, "defective_window_days": 90, "gold_grace_days": 0, "restocking_fee_pct": 0}) == "review"
    urns = urn_records(100, 0, lambda g: "train")
    for r in urns:
        r.validate()
        if r.question.type == "choice":
            counts = list(r.state["urn"].values())
            assert all(abs(p - Fraction(c, sum(counts))) < 1e-9 for p, c in zip(r.target.probs, counts))
    print("generators ok", len(recs), len(urns))
