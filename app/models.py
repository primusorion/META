from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Team = Literal["billing", "infra", "product", "access", "security"]
Priority = Literal["P1", "P2", "P3"]
Status = Literal["open", "in_progress", "resolved"]
Difficulty = Literal["easy", "medium", "hard"]


class Action(BaseModel):
    """Agent action used by the step(action) API."""

    action_type: Literal[
        "select_ticket",
        "set_priority",
        "assign_team",
        "request_info",
        "escalate",
        "resolve_ticket",
        "noop",
    ]
    ticket_id: str | None = None
    value: str | None = None
    note: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TicketPublicView(BaseModel):
    ticket_id: str
    customer_tier: Literal["free", "pro", "enterprise"]
    issue_type: Literal["billing", "infra", "product", "access", "security"]
    urgency: Literal["low", "medium", "high", "critical"]
    sentiment: Literal["calm", "frustrated", "angry"]
    sla_minutes_remaining: int
    status: Status
    assigned_team: Team | None = None
    assigned_priority: Priority | None = None
    escalated: bool = False


class TicketState(BaseModel):
    ticket_id: str
    customer_tier: Literal["free", "pro", "enterprise"]
    issue_type: Literal["billing", "infra", "product", "access", "security"]
    urgency: Literal["low", "medium", "high", "critical"]
    sentiment: Literal["calm", "frustrated", "angry"]
    sla_minutes_remaining: int
    status: Status = "open"
    assigned_team: Team | None = None
    assigned_priority: Priority | None = None
    escalated: bool = False
    info_requested: bool = False
    breached: bool = False
    resolution_note: str | None = None

    # Hidden labels used by graders.
    required_team: Team
    required_priority: Priority
    should_escalate: bool


class Reward(BaseModel):
    """Structured reward output with explainable components."""

    value: float
    cumulative: float
    components: dict[str, float]
    rationale: str


class Observation(BaseModel):
    task_id: str
    difficulty: Difficulty
    step_count: int
    max_steps: int
    active_ticket_id: str | None
    queue: list[TicketPublicView]
    completed_tickets: int
    breached_tickets: int
    score_estimate: float = Field(default=0.0, ge=0.0, le=1.0)
    last_feedback: str = ""


class EnvironmentState(BaseModel):
    task_id: str
    difficulty: Difficulty
    step_count: int
    max_steps: int
    active_ticket_id: str | None
    tickets: list[TicketState]
    action_history: list[Action]
    cumulative_reward: float
    done: bool
    final_score: float | None = None


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, str | float | int | bool]


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int = 0


class TaskSummary(BaseModel):
    task_id: str
    title: str
    difficulty: Difficulty
    max_steps: int
    objective: str
