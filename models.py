from __future__ import annotations

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SupportOpsTriageAction(Action):
    """OpenEnv action schema for the Support Ops triage environment."""

    action_type: str = Field(..., description="Action type")
    ticket_id: str | None = Field(default=None, description="Target ticket id")
    value: str | None = Field(default=None, description="Optional value for action")
    note: str | None = Field(default=None, description="Optional note")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SupportOpsTriageObservation(Observation):
    """OpenEnv observation schema for Support Ops triage."""

    task_id: str = Field(default="")
    step_count: int = Field(default=0)
    max_steps: int = Field(default=0)
    active_ticket_id: str | None = Field(default=None)
    completed_tickets: int = Field(default=0)
    breached_tickets: int = Field(default=0)
    score_estimate: float = Field(default=0.0, ge=0.0, le=1.0)
    last_feedback: str = Field(default="")
