"""
FastAPI server exposing the EmailTriageEnv as an HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from models import Action, Observation, Reward
from environment import EmailTriageEnv, StepResult

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")

# One env instance per task, keyed by task_id
_envs: dict[str, EmailTriageEnv] = {}


def _get_env(task_id: str) -> EmailTriageEnv:
    if task_id not in _envs:
        _envs[task_id] = EmailTriageEnv(task_id=task_id)
    return _envs[task_id]


class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Action


class StepResponse(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: dict


@app.post("/reset", response_model=StepResponse)
def reset(req: Optional[ResetRequest] = None):
    task_id = req.task_id if req else "easy"
    env = _get_env(task_id)
    result: StepResult = env.reset()
    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task_id)
    result: StepResult = env.step(req.action)
    return StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
def state(task_id: str = "easy"):
    env = _get_env(task_id)
    return env.state()


@app.get("/tasks")
def list_tasks():
    from tasks import TASKS
    return {
        tid: {
            "id": t["id"],
            "description": t["description"],
            "objective": t["objective"],
            "max_steps": t["max_steps"],
        }
        for tid, t in TASKS.items()
    }


@app.get("/")
def root():
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
