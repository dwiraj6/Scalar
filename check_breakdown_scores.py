"""Check if breakdown scores contain 0.0 or 1.0 values."""
from environment import EmailTriageEnv
from models import Action

# Test actions
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

print("Checking breakdown scores for 0.0 or 1.0 values:")
print("="*60)

found_invalid = []

for task_id in ["easy", "medium", "hard"]:
    print(f"\nTask: {task_id}")
    env = EmailTriageEnv(task_id=task_id)
    env.reset()
    
    step = 0
    for action in BEST_ACTIONS[task_id]:
        step += 1
        result = env.step(action)
        
        print(f"  Step {step}:")
        print(f"    Main score: {result.reward:.4f} (valid: {0 < result.reward < 1})")
        print(f"    Breakdown: {result.info.get('breakdown', {})}")
        
        # Check breakdown for 0.0 or 1.0 values
        breakdown = result.info.get('breakdown', {})
        
        def check_dict_values(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, dict):
                    check_dict_values(value, f"{prefix}{key}.")
                elif isinstance(value, (int, float)):
                    if value == 0.0 or value == 1.0:
                        msg = f"{prefix}{key} = {value}"
                        print(f"      ⚠️  {msg} (EXACTLY 0.0 or 1.0!)")
                        found_invalid.append(msg)
        
        check_dict_values(breakdown)

print("\n" + "="*60)
if found_invalid:
    print("❌ Found breakdown scores that are exactly 0.0 or 1.0!")
    print("   The validator might be checking these values too!")
else:
    print("✅ No breakdown scores are exactly 0.0 or 1.0")
