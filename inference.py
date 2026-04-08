"""
Baseline inference script for Email Triage OpenEnv.
Uses OpenAI client against API_BASE_URL / MODEL_NAME / HF_TOKEN.

Log format (strict):
  [START] task=<name> env=EmailTriageEnv model=<model>
  [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<None|msg>
  [END]   success=<bool> steps=<n> score=<float> rewards=<list>
"""

import asyncio
import json
import os
import sys
from typing import Optional

from openai import OpenAI

from environment import EmailTriageEnv
from models import Action

# ------------------------------------------------------------------
# Config from environment variables
# ------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")

BENCHMARK = "EmailTriageEnv"
MAX_STEPS = 3
MAX_TOTAL_REWARD = 3.0   # max cumulative reward across all steps (3 steps * max 1.0 each)
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS_TO_RUN = ["easy", "medium", "hard"]

# ------------------------------------------------------------------
# Strict log helpers
# ------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP]  step={step} action={action} reward={reward:.4f} done={done} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list):
    print(
        f"[END]   success={success} steps={steps} score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ------------------------------------------------------------------
# LLM call
# ------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    observation_dict: dict,
    task_id: str,
    step: int,
    last_reward: float,
    history: list[str],
) -> Action:
    """Ask the LLM to produce a structured Action for the current observation."""

    system_prompt = (
        "You are an expert email triage assistant. "
        "Given an email, respond with a JSON object matching this schema:\n"
        "{\n"
        '  "category": "urgent" | "normal" | "spam" | "newsletter",\n'
        '  "priority": "high" | "medium" | "low",\n'
        '  "reply": "<optional reply text>",\n'
        '  "forward_to": "<optional email address>",\n'
        '  "summary": "<optional one-sentence summary>"\n'
        "}\n"
        "Return ONLY valid JSON. No markdown, no explanation."
    )

    email = observation_dict.get("email", {})
    context = observation_dict.get("context", "")

    user_prompt = (
        f"Task: {task_id} | Step: {step} | Last reward: {last_reward:.4f}\n"
        f"Context: {context}\n\n"
        f"Email:\n"
        f"  From: {email.get('sender', '')}\n"
        f"  Subject: {email.get('subject', '')}\n"
        f"  Body: {email.get('body', '')}\n\n"
        f"History: {'; '.join(history[-3:]) if history else 'none'}\n\n"
        "Respond with JSON action:"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    return Action(**data)


# ------------------------------------------------------------------
# Run one task episode
# ------------------------------------------------------------------

async def run_task(client: OpenAI, task_id: str) -> float:
    TASK_NAME = task_id
    env = EmailTriageEnv(task_id=task_id)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: list[str] = []

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs = result.observation
            obs_dict = obs.model_dump() if obs else {}

            error = None
            action_str = ""

            try:
                action = get_model_action(
                    client, obs_dict, task_id, step, last_reward, history
                )
                action_str = action.model_dump_json()
                result = env.step(action)
                reward = result.reward
            except Exception as e:
                error = str(e)
                reward = 0.0
                result.done = True  # type: ignore

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=result.done, error=error)
            history.append(f"Step {step}: {action_str!r} -> reward {reward:+.4f}")

            if result.done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        pass  # No async close needed for in-process env

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def main():
    api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)

    all_scores: dict[str, float] = {}

    for task_id in TASKS_TO_RUN:
        print(f"\n{'='*50}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*50}", flush=True)
        score = await run_task(client, task_id)
        all_scores[task_id] = score

    print("\n[SUMMARY]", flush=True)
    for tid, sc in all_scores.items():
        print(f"  {tid}: {sc:.4f}", flush=True)
    overall = sum(all_scores.values()) / len(all_scores)
    print(f"  overall: {overall:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
