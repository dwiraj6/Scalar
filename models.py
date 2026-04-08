from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class Email(BaseModel):
    id: str
    subject: str
    body: str
    sender: str
    timestamp: str


class Observation(BaseModel):
    email: Email
    task_id: str
    step_number: int
    max_steps: int
    context: Optional[str] = None


class Action(BaseModel):
    category: Literal["urgent", "normal", "spam", "newsletter"]
    priority: Literal["high", "medium", "low"]
    reply: Optional[str] = None
    forward_to: Optional[str] = None
    summary: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    breakdown: dict
    feedback: str
    done: bool
