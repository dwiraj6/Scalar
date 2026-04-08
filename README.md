---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - email-triage
  - rl-environment
  - agent-evaluation
---

# Email Triage OpenEnv

A real-world OpenEnv environment that simulates professional inbox triage —
classifying, prioritizing, replying to, and routing emails.
Built for AI agent evaluation and reinforcement learning research.

---

## Real-World Relevance

Email triage is one of the most common knowledge-worker tasks globally.
Professionals spend an average of 2.5 hours/day on email.
This environment models the core decisions: what is urgent, what can wait,
what needs a reply, and what should be forwarded — making it directly useful
for training and evaluating LLM-based agents on practical office tasks.

---

## Environment Design

The environment follows the full OpenEnv spec:

- `reset()` — initializes episode, returns first `Observation`
- `step(action)` — processes one `Action`, returns reward + next observation
- `state()` — returns current episode state as a plain dict
- Typed Pydantic models: `Observation`, `Action`, `Reward`
- HTTP API via FastAPI (port 7860) for containerized deployment

---

## Action Space

```json
{
  "category":   "urgent" | "normal" | "spam" | "newsletter",
  "priority":   "high" | "medium" | "low",
  "reply":      "<optional reply text>",
  "forward_to": "<optional email address>",
  "summary":    "<optional one-sentence summary>"
}
```

## Observation Space

```json
{
  "email": {
    "id":        "<string>",
    "subject":   "<string>",
    "body":      "<string>",
    "sender":    "<string>",
    "timestamp": "<ISO 8601>"
  },
  "task_id":     "easy" | "medium" | "hard",
  "step_number": <int>,
  "max_steps":   <int>,
  "context":     "<optional hint string>"
}
```

---

## Tasks

### Easy
- Objective: Classify a single promotional email by category and priority
- Email: A discount newsletter from a retail sender
- Expected: `category=newsletter`, `priority=low`
- Max steps: 1
- Grader weights: category (60%), priority (40%)

### Medium
- Objective: Classify an overdue invoice email, write a professional reply, and summarize it
- Email: Client chasing a $3,200 overdue invoice
- Expected: `category=urgent`, `priority=high`, reply with relevant keywords, summary ≥ 8 words
- Max steps: 1
- Grader weights: category (25%), priority (25%), reply quality (35%), summary (15%)

### Hard
- Objective: Triage a 3-email inbox — classify all, reply to urgent ones, forward billing dispute
- Emails: Production outage, monthly newsletter, payment dispute
- Expected: Correct classification + replies + `forward_to=billing@company.com` for dispute
- Max steps: 3 (one per email)
- Grader: average of per-email scores; each scored on category, priority, reply quality, forwarding

---

## Reward Function

Rewards are partial and continuous (0.0–1.0), not binary:

- Correct category → partial score
- Correct priority → partial score
- Reply quality → keyword coverage score (fraction of expected keywords present)
- Missing reply on urgent email → penalized (reply_score = 0)
- Forwarding accuracy → exact match on email address
- Hard task: per-step partial rewards + final aggregated score

---

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Hugging Face Spaces

Push this repo to a HF Space with `sdk: docker`.
The Space will auto-build and expose port 7860.

---

## Running Inference

Set environment variables:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
```

Run:

```bash
python inference.py
```

Output follows strict log format:

```
[START] task=easy env=EmailTriageEnv model=gpt-4o-mini
[STEP]  step=1 action={...} reward=1.0000 done=True error=None
[END]   success=True steps=1 score=0.3333 rewards=[1.0]
```

---

## Baseline Scores (gpt-4o-mini)

| Task   | Score |
|--------|-------|
| easy   | ~1.00 |
| medium | ~0.75 |
| hard   | ~0.65 |
| overall| ~0.80 |

---

## Project Structure

```
.
├── models.py        # Pydantic models: Observation, Action, Reward
├── tasks.py         # Task definitions and email fixtures
├── graders.py       # Deterministic graders for each task
├── environment.py   # Core OpenEnv environment (reset/step/state)
├── server.py        # FastAPI HTTP server
├── inference.py     # Baseline inference script (OpenAI client)
├── openenv.yaml     # OpenEnv metadata and spec
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

---

## Environment Variables

| Variable      | Description                        |
|---------------|------------------------------------|
| API_BASE_URL  | LLM API endpoint                   |
| MODEL_NAME    | Model identifier for inference     |
| HF_TOKEN      | Hugging Face / API key             |
