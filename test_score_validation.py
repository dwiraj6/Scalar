"""Test that all grader scores are strictly between 0 and 1."""
from environment import EmailTriageEnv
from models import Action

def test_all_edge_cases():
    """Test various action combinations to ensure scores are valid."""
    test_cases = [
        # Easy task - all correct
        ("easy", [Action(category="newsletter", priority="low")]),
        # Easy task - all wrong
        ("easy", [Action(category="urgent", priority="high")]),
        # Easy task - partial correct
        ("easy", [Action(category="newsletter", priority="high")]),
        
        # Medium task - all correct
        ("medium", [Action(
            category="urgent", 
            priority="high",
            reply="We apologize and will process the invoice payment immediately.",
            summary="Client following up on overdue invoice 4821 for 3200 dollars."
        )]),
        # Medium task - all wrong
        ("medium", [Action(category="newsletter", priority="low")]),
        
        # Hard task - all correct
        ("hard", [
            Action(category="urgent", priority="high", 
                   reply="We are looking into the outage immediately with the team."),
            Action(category="newsletter", priority="low"),
            Action(category="urgent", priority="high", 
                   reply="We will investigate the duplicate charge and refund it.", 
                   forward_to="billing@company.com"),
        ]),
        # Hard task - all wrong
        ("hard", [
            Action(category="newsletter", priority="low"),
            Action(category="urgent", priority="high"),
            Action(category="newsletter", priority="low"),
        ]),
    ]
    
    all_valid = True
    for task_id, actions in test_cases:
        env = EmailTriageEnv(task_id=task_id)
        env.reset()
        
        for action in actions:
            result = env.step(action)
            score = result.reward
            
            # Check if score is strictly between 0 and 1
            if not (0.0 < score < 1.0):
                print(f"❌ INVALID: task={task_id}, score={score} (must be strictly between 0 and 1)")
                all_valid = False
            else:
                print(f"✓ Valid: task={task_id}, score={score:.4f}")
    
    if all_valid:
        print("\n✅ All scores are strictly between 0 and 1!")
    else:
        print("\n❌ Some scores are out of range!")
    
    return all_valid

if __name__ == "__main__":
    test_all_edge_cases()
