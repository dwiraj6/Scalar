from models import Action, Reward
from typing import Optional


def _keyword_score(text: Optional[str], keywords: list[str]) -> float:
    """Fraction of keywords found in text (case-insensitive)."""
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def grade_easy(action: Action, expected: dict) -> Reward:
    cat_correct = action.category == expected["category"]
    pri_correct = action.priority == expected["priority"]

    cat_score = 1.0 if cat_correct else 0.0
    pri_score = 1.0 if pri_correct else 0.0
    total = round((cat_score * 0.6) + (pri_score * 0.4), 4)

    return Reward(
        score=total,
        breakdown={"category": cat_score, "priority": pri_score},
        feedback=(
            "Correct classification."
            if total == 1.0
            else f"Expected category={expected['category']}, priority={expected['priority']}."
        ),
        done=True,
    )


def grade_medium(action: Action, expected: dict) -> Reward:
    cat_correct = action.category == expected["category"]
    pri_correct = action.priority == expected["priority"]

    cat_score = 1.0 if cat_correct else 0.0
    pri_score = 1.0 if pri_correct else 0.0

    reply_score = _keyword_score(action.reply, expected.get("reply_keywords", []))

    summary_score = 0.0
    if action.summary:
        word_count = len(action.summary.split())
        summary_score = 1.0 if word_count >= expected.get("summary_min_words", 8) else 0.5

    # Penalize missing reply for urgent email
    if expected["category"] == "urgent" and not action.reply:
        reply_score = 0.0

    total = round(
        cat_score * 0.25
        + pri_score * 0.25
        + reply_score * 0.35
        + summary_score * 0.15,
        4,
    )

    return Reward(
        score=total,
        breakdown={
            "category": cat_score,
            "priority": pri_score,
            "reply_quality": round(reply_score, 4),
            "summary_quality": summary_score,
        },
        feedback=(
            "Good triage and response."
            if total >= 0.8
            else "Improve reply content and ensure correct classification."
        ),
        done=True,
    )


def grade_hard_step(action: Action, expected: dict) -> dict:
    """Grade a single step in the hard task. Returns a score dict."""
    cat_score = 1.0 if action.category == expected["category"] else 0.0
    pri_score = 1.0 if action.priority == expected["priority"] else 0.0

    reply_score = _keyword_score(action.reply, expected.get("reply_keywords", []))

    forward_score = 0.0
    if "forward_to" in expected:
        if action.forward_to and action.forward_to.strip().lower() == expected["forward_to"].lower():
            forward_score = 1.0
        else:
            forward_score = 0.0

    weights = {"category": 0.3, "priority": 0.2, "reply": 0.3, "forward": 0.2}
    if "forward_to" not in expected:
        # redistribute forward weight to reply
        weights = {"category": 0.35, "priority": 0.25, "reply": 0.4, "forward": 0.0}

    step_score = round(
        cat_score * weights["category"]
        + pri_score * weights["priority"]
        + reply_score * weights["reply"]
        + forward_score * weights["forward"],
        4,
    )

    return {
        "email_id": expected["email_id"],
        "score": step_score,
        "category": cat_score,
        "priority": pri_score,
        "reply_quality": round(reply_score, 4),
        "forward": forward_score,
    }


def grade_hard(step_results: list[dict]) -> Reward:
    if not step_results:
        return Reward(score=0.0, breakdown={}, feedback="No steps completed.", done=True)

    total = round(sum(r["score"] for r in step_results) / len(step_results), 4)
    breakdown = {r["email_id"]: r for r in step_results}

    return Reward(
        score=total,
        breakdown=breakdown,
        feedback=(
            "Excellent inbox triage."
            if total >= 0.8
            else "Some emails were misclassified or replies were insufficient."
        ),
        done=True,
    )
