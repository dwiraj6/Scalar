from models import Action, Reward
from typing import Optional

# Scores must be strictly between 0 and 1 (exclusive)
_MIN = 0.01
_MAX = 0.99


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    clamped = max(_MIN, min(_MAX, score))
    # Round to 4 decimals but ensure we stay within bounds
    rounded = round(clamped, 4)
    # Final safety check to ensure strictly between 0 and 1
    if rounded <= 0.0:
        return _MIN
    elif rounded >= 1.0:
        return _MAX
    return rounded


def _keyword_score(text: Optional[str], keywords: list[str]) -> float:
    """Fraction of keywords found in text (case-insensitive)."""
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def grade_easy(action: Action, expected: dict) -> Reward:
    cat_score = 1.0 if action.category == expected["category"] else 0.0
    pri_score = 1.0 if action.priority == expected["priority"] else 0.0
    total = _clamp((cat_score * 0.6) + (pri_score * 0.4))

    return Reward(
        score=total,
        breakdown={"category": _clamp(cat_score), "priority": _clamp(pri_score)},
        feedback=(
            "Correct classification."
            if total >= 0.95
            else f"Expected category={expected['category']}, priority={expected['priority']}."
        ),
        done=True,
    )


def grade_medium(action: Action, expected: dict) -> Reward:
    cat_score = 1.0 if action.category == expected["category"] else 0.0
    pri_score = 1.0 if action.priority == expected["priority"] else 0.0
    reply_score = _keyword_score(action.reply, expected.get("reply_keywords", []))

    summary_score = 0.0
    if action.summary:
        word_count = len(action.summary.split())
        summary_score = 1.0 if word_count >= expected.get("summary_min_words", 8) else 0.5

    if expected["category"] == "urgent" and not action.reply:
        reply_score = 0.0

    raw = (
        cat_score * 0.25
        + pri_score * 0.25
        + reply_score * 0.35
        + summary_score * 0.15
    )
    total = _clamp(raw)

    return Reward(
        score=total,
        breakdown={
            "category": _clamp(cat_score),
            "priority": _clamp(pri_score),
            "reply_quality": _clamp(reply_score),
            "summary_quality": _clamp(summary_score),
        },
        feedback=(
            "Good triage and response."
            if total >= 0.75
            else "Improve reply content and ensure correct classification."
        ),
        done=True,
    )


def grade_hard_step(action: Action, expected: dict) -> dict:
    cat_score = 1.0 if action.category == expected["category"] else 0.0
    pri_score = 1.0 if action.priority == expected["priority"] else 0.0
    reply_score = _keyword_score(action.reply, expected.get("reply_keywords", []))

    forward_score = 0.0
    if "forward_to" in expected:
        if action.forward_to and action.forward_to.strip().lower() == expected["forward_to"].lower():
            forward_score = 1.0

    if "forward_to" not in expected:
        weights = {"category": 0.35, "priority": 0.25, "reply": 0.4, "forward": 0.0}
    else:
        weights = {"category": 0.3, "priority": 0.2, "reply": 0.3, "forward": 0.2}

    raw = (
        cat_score * weights["category"]
        + pri_score * weights["priority"]
        + reply_score * weights["reply"]
        + forward_score * weights["forward"]
    )
    step_score = _clamp(raw)

    return {
        "email_id": expected["email_id"],
        "score": step_score,
        "category": _clamp(cat_score),
        "priority": _clamp(pri_score),
        "reply_quality": _clamp(reply_score),
        "forward": _clamp(forward_score),
    }


def grade_hard(step_results: list[dict]) -> Reward:
    if not step_results:
        return Reward(score=_MIN, breakdown={}, feedback="No steps completed.", done=True)

    raw = sum(r["score"] for r in step_results) / len(step_results)
    total = _clamp(raw)
    breakdown = {r["email_id"]: r for r in step_results}

    return Reward(
        score=total,
        breakdown=breakdown,
        feedback=(
            "Excellent inbox triage."
            if total >= 0.75
            else "Some emails were misclassified or replies were insufficient."
        ),
        done=True,
    )
