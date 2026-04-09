"""Debug all possible score outputs to find edge cases."""
from environment import EmailTriageEnv
from models import Action

MAX_TOTAL_REWARD = 3.0

def test_task_with_actions(task_id, actions_list, description):
    """Test a task with given actions and check if scores are valid."""
    env = EmailTriageEnv(task_id=task_id)
    env.reset()
    
    rewards = []
    for action in actions_list:
        result = env.step(action)
        rewards.append(result.reward)
        print(f"  Step reward: {result.reward:.4f} (valid: {0 < result.reward < 1})")
    
    # Calculate final score like inference.py does
    final_score = sum(rewards) / MAX_TOTAL_REWARD
    final_score = min(max(final_score, 0.0), 1.0)
    
    is_valid = 0 < final_score < 1
    status = "✓" if is_valid else "❌"
    
    print(f"{status} {description}")
    print(f"  Task: {task_id}")
    print(f"  Rewards: {rewards}")
    print(f"  Sum: {sum(rewards):.4f}")
    print(f"  Final score: {final_score:.6f} (valid: {is_valid})")
    print()
    
    return is_valid, final_score

print("="*60)
print("TESTING ALL EDGE CASES")
print("="*60)
print()

all_valid = True

# Easy task - perfect
valid, score = test_task_with_actions(
    "easy",
    [Action(category="newsletter", priority="low")],
    "Easy - Perfect match"
)
all_valid = all_valid and valid

# Easy task - complete failure
valid, score = test_task_with_actions(
    "easy",
    [Action(category="urgent", priority="high")],
    "Easy - Complete failure"
)
all_valid = all_valid and valid

# Medium task - perfect
valid, score = test_task_with_actions(
    "medium",
    [Action(
        category="urgent", 
        priority="high",
        reply="We apologize and will process the invoice payment immediately.",
        summary="Client following up on overdue invoice 4821 for 3200 dollars."
    )],
    "Medium - Perfect match"
)
all_valid = all_valid and valid

# Medium task - complete failure
valid, score = test_task_with_actions(
    "medium",
    [Action(category="newsletter", priority="low")],
    "Medium - Complete failure"
)
all_valid = all_valid and valid

# Hard task - perfect
valid, score = test_task_with_actions(
    "hard",
    [
        Action(category="urgent", priority="high", 
               reply="We are looking into the outage immediately with the team."),
        Action(category="newsletter", priority="low"),
        Action(category="urgent", priority="high", 
               reply="We will investigate the duplicate charge and refund it.", 
               forward_to="billing@company.com"),
    ],
    "Hard - Perfect match"
)
all_valid = all_valid and valid

# Hard task - complete failure
valid, score = test_task_with_actions(
    "hard",
    [
        Action(category="newsletter", priority="low"),
        Action(category="urgent", priority="high"),
        Action(category="newsletter", priority="low"),
    ],
    "Hard - Complete failure"
)
all_valid = all_valid and valid

# Hard task - mixed results
valid, score = test_task_with_actions(
    "hard",
    [
        Action(category="urgent", priority="high"),  # partial match
        Action(category="newsletter", priority="low"),  # perfect match
        Action(category="urgent", priority="low"),  # partial match
    ],
    "Hard - Mixed results"
)
all_valid = all_valid and valid

print("="*60)
if all_valid:
    print("✅ ALL SCORES ARE VALID (strictly between 0 and 1)")
else:
    print("❌ SOME SCORES ARE INVALID (equal to 0.0 or 1.0)")
print("="*60)
