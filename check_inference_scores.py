"""Simulate what inference.py logs without needing a real API key."""
import sys

# Patch: simulate model always returning correct easy action
from environment import EmailTriageEnv
from models import Action

MAX_TOTAL_REWARD = 3.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["easy", "medium", "hard"]

# Simulate best-case actions per task
BEST_ACTIONS = {
    "easy": [Action(category="newsletter", priority="low")],
    "medium": [Action(category="urgent", priority="high",
                      reply="We apologize and will process the invoice payment immediately.",
                      summary="Client following up on overdue invoice 4821 for 3200 dollars.")],
    "hard": [
        Action(category="urgent", priority="high", reply="We are looking into the outage immediately with the team."),
        Action(category="newsletter", priority="low"),
        Action(category="urgent", priority="high", reply="We will investigate the duplicate charge and refund it.", forward_to="billing@company.com"),
    ],
}

for task_id in TASKS:
    env = EmailTriageEnv(task_id=task_id)
    result = env.reset()
    rewards = []
    for action in BEST_ACTIONS[task_id]:
        result = env.step(action)
        rewards.append(result.reward)
    score = sum(rewards) / MAX_TOTAL_REWARD
    score = min(max(score, 0.0), 1.0)
    print(f"[END] task={task_id} rewards={rewards} score={score:.4f} strictly_in_range={0.0 < score < 1.0}")
