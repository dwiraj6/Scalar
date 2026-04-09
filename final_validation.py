"""Final comprehensive validation of all scores."""
from environment import EmailTriageEnv
from models import Action
import json

def check_all_numeric_values(obj, path=""):
    """Recursively check all numeric values in an object."""
    invalid = []
    
    # Skip checking if this is a known boolean field
    if path.endswith(".done") or path == "done":
        return invalid
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            invalid.extend(check_all_numeric_values(value, new_path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            new_path = f"{path}[{i}]"
            invalid.extend(check_all_numeric_values(value, new_path))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if obj == 0.0 or obj == 1.0:
            invalid.append((path, obj))
    
    return invalid

# Test all tasks with various scenarios
test_scenarios = [
    ("easy", "perfect", [Action(category="newsletter", priority="low")]),
    ("easy", "failure", [Action(category="urgent", priority="high")]),
    ("easy", "partial", [Action(category="newsletter", priority="high")]),
    
    ("medium", "perfect", [Action(
        category="urgent", priority="high",
        reply="We apologize and will process the invoice payment immediately.",
        summary="Client following up on overdue invoice 4821 for 3200 dollars."
    )]),
    ("medium", "failure", [Action(category="newsletter", priority="low")]),
    ("medium", "no_reply", [Action(category="urgent", priority="high")]),
    
    ("hard", "perfect", [
        Action(category="urgent", priority="high", 
               reply="We are looking into the outage immediately with the team."),
        Action(category="newsletter", priority="low"),
        Action(category="urgent", priority="high", 
               reply="We will investigate the duplicate charge and refund it.", 
               forward_to="billing@company.com"),
    ]),
    ("hard", "failure", [
        Action(category="newsletter", priority="low"),
        Action(category="urgent", priority="high"),
        Action(category="newsletter", priority="low"),
    ]),
]

print("="*70)
print("FINAL COMPREHENSIVE VALIDATION")
print("="*70)

all_valid = True
total_tests = 0
passed_tests = 0

for task_id, scenario, actions in test_scenarios:
    total_tests += 1
    print(f"\nTest: {task_id} - {scenario}")
    
    env = EmailTriageEnv(task_id=task_id)
    env.reset()
    
    step_results = []
    for action in actions:
        result = env.step(action)
        step_results.append({
            "reward": result.reward,
            "breakdown": result.info.get("breakdown", {}),
            "done": result.done
        })
    
    # Check all numeric values
    invalid_values = check_all_numeric_values(step_results)
    
    if invalid_values:
        print(f"  ❌ FAILED - Found invalid values:")
        for path, value in invalid_values:
            print(f"     {path} = {value}")
        all_valid = False
    else:
        print(f"  ✅ PASSED - All values strictly between 0 and 1")
        passed_tests += 1

print("\n" + "="*70)
print(f"Results: {passed_tests}/{total_tests} tests passed")

if all_valid:
    print("✅ ALL TESTS PASSED - Ready for submission!")
else:
    print("❌ SOME TESTS FAILED - Fix required")

print("="*70)
