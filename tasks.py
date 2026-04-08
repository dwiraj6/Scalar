from models import Email

TASKS = {
    "easy": {
        "id": "easy",
        "description": (
            "Classify a single email by category and priority. "
            "No reply required."
        ),
        "objective": "Correctly categorize and prioritize the email.",
        "max_steps": 1,
        "emails": [
            Email(
                id="e1",
                subject="50% OFF - Limited Time Offer!",
                body=(
                    "Dear customer, enjoy our biggest sale of the year. "
                    "Click here to shop now. Unsubscribe at any time."
                ),
                sender="deals@shopnow.com",
                timestamp="2026-04-08T09:00:00Z",
            )
        ],
        "expected": {
            "category": "newsletter",
            "priority": "low",
        },
    },
    "medium": {
        "id": "medium",
        "description": (
            "Classify an email, write a short professional reply, "
            "and provide a one-sentence summary."
        ),
        "objective": (
            "Correctly classify, reply appropriately, and summarize the email."
        ),
        "max_steps": 1,
        "emails": [
            Email(
                id="m1",
                subject="Invoice #4821 overdue",
                body=(
                    "Hi, I noticed invoice #4821 for $3,200 is now 15 days overdue. "
                    "Could you please confirm when payment will be processed? "
                    "Let me know if there are any issues. Thanks, Alex."
                ),
                sender="alex@clientcorp.com",
                timestamp="2026-04-08T10:30:00Z",
            )
        ],
        "expected": {
            "category": "urgent",
            "priority": "high",
            "reply_keywords": ["invoice", "payment", "apologize", "process"],
            "summary_min_words": 8,
        },
    },
    "hard": {
        "id": "hard",
        "description": (
            "Process a batch of 3 emails: classify each, write replies for urgent ones, "
            "forward the appropriate one to billing@company.com, and summarize all."
        ),
        "objective": (
            "Handle a realistic inbox triage scenario with multiple emails correctly."
        ),
        "max_steps": 3,
        "emails": [
            Email(
                id="h1",
                subject="Server down - production outage",
                body=(
                    "URGENT: Our production server has been down for 20 minutes. "
                    "Customers cannot access the platform. Need immediate assistance."
                ),
                sender="ops@techclient.com",
                timestamp="2026-04-08T08:00:00Z",
            ),
            Email(
                id="h2",
                subject="Monthly newsletter - April edition",
                body=(
                    "Welcome to our April newsletter! This month we cover "
                    "industry trends, upcoming webinars, and team spotlights."
                ),
                sender="news@industry.org",
                timestamp="2026-04-08T08:05:00Z",
            ),
            Email(
                id="h3",
                subject="Payment dispute for order #9921",
                body=(
                    "Hello, I was charged twice for order #9921. "
                    "Please refund the duplicate charge of $89.99. "
                    "My account: user@example.com"
                ),
                sender="user@example.com",
                timestamp="2026-04-08T08:10:00Z",
            ),
        ],
        "expected": [
            {
                "email_id": "h1",
                "category": "urgent",
                "priority": "high",
                "reply_keywords": ["outage", "team", "immediately", "looking"],
            },
            {
                "email_id": "h2",
                "category": "newsletter",
                "priority": "low",
            },
            {
                "email_id": "h3",
                "category": "urgent",
                "priority": "high",
                "forward_to": "billing@company.com",
                "reply_keywords": ["refund", "charge", "investigate"],
            },
        ],
    },
}
