from models import Observation, Action, Reward, Email
from tasks import TASKS
from graders import grade_easy, grade_medium, grade_hard_step, grade_hard
from environment import EmailTriageEnv

print("--- Testing imports: OK")

# ---- EASY ----
env = EmailTriageEnv(task_id="easy")
result = env.reset()
assert result.observation.email.id == "e1", "easy reset email id wrong"
assert result.done is False, "easy reset should not be done"
print(f"easy reset()  email={result.observation.email.id}  done={result.done}")

# correct action
action = Action(category="newsletter", priority="low")
result = env.step(action)
assert result.reward == 1.0, f"easy correct action should score 1.0, got {result.reward}"
assert result.done is True
print(f"easy correct  reward={result.reward}  done={result.done}")

# wrong action
env.reset()
bad = Action(category="spam", priority="high")
result = env.step(bad)
assert result.reward == 0.0, f"easy wrong action should score 0.0, got {result.reward}"
print(f"easy wrong    reward={result.reward}  done={result.done}")

# ---- MEDIUM ----
env2 = EmailTriageEnv(task_id="medium")
env2.reset()
action2 = Action(
    category="urgent",
    priority="high",
    reply="We apologize for the delay. We will process the invoice payment immediately.",
    summary="Client is following up on overdue invoice 4821 for 3200 dollars.",
)
result2 = env2.step(action2)
assert result2.done is True
assert result2.reward > 0.5, f"medium good action should score >0.5, got {result2.reward}"
print(f"medium good   reward={result2.reward}  done={result2.done}")
print(f"medium breakdown: {result2.info['breakdown']}")

# medium with no reply (should penalize)
env2.reset()
no_reply = Action(category="urgent", priority="high")
result_nr = env2.step(no_reply)
assert result_nr.reward < result2.reward, "missing reply should score lower"
print(f"medium no-reply penalty  reward={result_nr.reward}")

# ---- HARD ----
env3 = EmailTriageEnv(task_id="hard")
env3.reset()

r1 = env3.step(Action(
    category="urgent", priority="high",
    reply="We are looking into the outage immediately with the team."
))
assert r1.done is False, "hard should not be done after step 1"
print(f"hard step 1   reward={r1.reward}  done={r1.done}")

r2 = env3.step(Action(category="newsletter", priority="low"))
assert r2.done is False, "hard should not be done after step 2"
print(f"hard step 2   reward={r2.reward}  done={r2.done}")

r3 = env3.step(Action(
    category="urgent", priority="high",
    reply="We will investigate the duplicate charge and refund it.",
    forward_to="billing@company.com",
))
assert r3.done is True, "hard should be done after step 3"
assert r3.reward > 0.5, f"hard good actions should score >0.5, got {r3.reward}"
print(f"hard step 3   reward={r3.reward}  done={r3.done}")
print(f"hard breakdown: {r3.info['breakdown']}")

# ---- STATE ----
s = env3.state()
assert s["task_id"] == "hard"
assert s["done"] is True
assert s["step_number"] == 3
print(f"state()  task={s['task_id']}  steps={s['step_number']}  done={s['done']}")

# ---- REWARD BOUNDS ----
for reward_val in [r1.reward, r2.reward, r3.reward, result2.reward]:
    assert 0.0 <= reward_val <= 1.0, f"reward out of bounds: {reward_val}"
print("reward bounds 0.0-1.0: OK")

# ---- RESET CLEARS STATE ----
env3.reset()
s2 = env3.state()
assert s2["step_number"] == 0
assert s2["done"] is False
print("reset() clears state: OK")

print()
print("=" * 40)
print("ALL TESTS PASSED")
print("=" * 40)
