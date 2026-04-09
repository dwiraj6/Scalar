"""Simulate the exact output that inference.py would produce."""
from environment import EmailTriageEnv
from models import Action

MAX_TOTAL_REWARD = 3.0

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

print("Simulating inference.py output format:")
print("="*60)

for task_id in ["easy", "medium", "hard"]:
    print(f"\n[START] task={task_id} env=EmailTriageEnv model=test")
    
    env = EmailTriageEnv(task_id=task_id)
    env.reset()
    
    rewards = []
    step = 0
    
    for action in BEST_ACTIONS[task_id]:
        step += 1
        result = env.step(action)
        reward = result.reward
        rewards.append(reward)
        
        action_str = action.model_dump_json()
        print(f"[STEP]  step={step} action={action_str} reward={reward:.4f} done={result.done} error=None")
        
        # Check if this reward is valid
        if not (0 < reward < 1):
            print(f"  ⚠️  WARNING: Step reward {reward} is NOT strictly between 0 and 1!")
    
    # Calculate final score
    score = sum(rewards) / MAX_TOTAL_REWARD
    score = min(max(score, 0.0), 1.0)
    success = score >= 0.5
    
    print(f"[END]   success={success} steps={step} score={score:.4f} rewards={rewards}")
    
    # Check if final score is valid
    if not (0 < score < 1):
        print(f"  ⚠️  WARNING: Final score {score} is NOT strictly between 0 and 1!")
    
    # Check each reward in the list
    for i, r in enumerate(rewards):
        if not (0 < r < 1):
            print(f"  ⚠️  WARNING: rewards[{i}] = {r} is NOT strictly between 0 and 1!")

print("\n" + "="*60)
print("Checking all values...")

all_valid = True
for task_id in ["easy", "medium", "hard"]:
    env = EmailTriageEnv(task_id=task_id)
    env.reset()
    rewards = []
    for action in BEST_ACTIONS[task_id]:
        result = env.step(action)
        rewards.append(result.reward)
    
    score = sum(rewards) / MAX_TOTAL_REWARD
    score = min(max(score, 0.0), 1.0)
    
    # Check each reward
    for r in rewards:
        if not (0 < r < 1):
            print(f"❌ Task {task_id}: reward {r} is OUT OF RANGE")
            all_valid = False
    
    # Check final score
    if not (0 < score < 1):
        print(f"❌ Task {task_id}: final score {score} is OUT OF RANGE")
        all_valid = False

if all_valid:
    print("✅ All task scores and rewards are strictly between 0 and 1")
else:
    print("❌ Some scores are out of range!")
