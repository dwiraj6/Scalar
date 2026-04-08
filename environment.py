"""
Email Triage OpenEnv Environment
"""
import copy
from typing import Optional
from models import Observation, Action, Reward, Email
from tasks import TASKS
from graders import grade_easy, grade_medium, grade_hard_step, grade_hard


class StepResult:
    def __init__(self, observation, reward: float, done: bool, info: dict):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info


class EmailTriageEnv:
    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")
        self._task_id = task_id
        self._task = copy.deepcopy(TASKS[task_id])
        self._step_number = 0
        self._done = False
        self._hard_step_results = []
        self._current_email_index = 0

    def reset(self) -> StepResult:
        self._task = copy.deepcopy(TASKS[self._task_id])
        self._step_number = 0
        self._done = False
        self._hard_step_results = []
        self._current_email_index = 0
        obs = self._make_observation()
        return StepResult(observation=obs, reward=0.0, done=False, info={"task": self._task_id})

    def step(self, action: Action) -> StepResult:
        if self._done:
            obs = self._make_observation()
            return StepResult(observation=obs, reward=0.0, done=True, info={"warning": "Episode already done."})
        self._step_number += 1
        if self._task_id == "easy":
            reward_obj = grade_easy(action, self._task["expected"])
            self._done = True
        elif self._task_id == "medium":
            reward_obj = grade_medium(action, self._task["expected"])
            self._done = True
        elif self._task_id == "hard":
            expected_list = self._task["expected"]
            idx = self._current_email_index
            step_result = grade_hard_step(action, expected_list[idx])
            self._hard_step_results.append(step_result)
            self._current_email_index += 1
            if self._current_email_index >= len(self._task["emails"]):
                reward_obj = grade_hard(self._hard_step_results)
                self._done = True
            else:
                reward_obj = Reward(
                    score=round(step_result["score"], 4),
                    breakdown=step_result,
                    feedback=f"Email {step_result['email_id']} processed.",
                    done=False,
                )
        else:
            raise RuntimeError(f"Unknown task_id: {self._task_id}")
        obs = self._make_observation() if not self._done else None
        return StepResult(
            observation=obs,
            reward=reward_obj.score,
            done=self._done,
            info={
                "breakdown": reward_obj.breakdown,
                "feedback": reward_obj.feedback,
                "step": self._step_number,
            },
        )

    def state(self) -> dict:
        return {
            "task_id": self._task_id,
            "step_number": self._step_number,
            "done": self._done,
            "current_email_index": self._current_email_index,
            "max_steps": self._task["max_steps"],
            "hard_step_results": self._hard_step_results,
        }

    def _make_observation(self) -> Observation:
        emails = self._task["emails"]
        idx = min(self._current_email_index, len(emails) - 1)
        email = emails[idx]
        context = None
        if self._task_id == "hard":
            remaining = len(emails) - self._current_email_index
            context = f"Inbox triage: {remaining} email(s) remaining. Process email {self._current_email_index + 1} of {len(emails)}."
        elif self._task_id == "medium":
            context = "Reply professionally and provide a one-sentence summary."
        return Observation(
            email=email,
            task_id=self._task_id,
            step_number=self._step_number,
            max_steps=self._task["max_steps"],
            context=context,
        )
