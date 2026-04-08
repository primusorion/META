from __future__ import annotations

from dataclasses import dataclass

from .models import Difficulty


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    difficulty: Difficulty
    max_steps: int
    objective: str
    tickets: tuple[dict[str, object], ...]


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="easy_priority_rescue",
        title="Save the enterprise outage first",
        difficulty="easy",
        max_steps=12,
        objective=(
            "Prioritize and resolve the critical enterprise outage before SLA breach, "
            "while still handling low-risk tickets."
        ),
        tickets=(
            {
                "ticket_id": "E-1001",
                "customer_tier": "enterprise",
                "issue_type": "infra",
                "urgency": "critical",
                "sentiment": "angry",
                "sla_minutes_remaining": 45,
                "required_team": "infra",
                "required_priority": "P1",
                "should_escalate": True,
            },
            {
                "ticket_id": "E-1002",
                "customer_tier": "pro",
                "issue_type": "billing",
                "urgency": "medium",
                "sentiment": "frustrated",
                "sla_minutes_remaining": 180,
                "required_team": "billing",
                "required_priority": "P2",
                "should_escalate": False,
            },
            {
                "ticket_id": "E-1003",
                "customer_tier": "free",
                "issue_type": "access",
                "urgency": "high",
                "sentiment": "frustrated",
                "sla_minutes_remaining": 120,
                "required_team": "access",
                "required_priority": "P2",
                "should_escalate": False,
            },
        ),
    ),
    TaskSpec(
        task_id="medium_sla_balancing",
        title="Balance SLA pressure across teams",
        difficulty="medium",
        max_steps=20,
        objective=(
            "Handle a mixed queue with conflicting urgency while avoiding unnecessary "
            "escalations and minimizing total SLA breaches."
        ),
        tickets=(
            {
                "ticket_id": "M-2001",
                "customer_tier": "pro",
                "issue_type": "security",
                "urgency": "high",
                "sentiment": "angry",
                "sla_minutes_remaining": 60,
                "required_team": "security",
                "required_priority": "P1",
                "should_escalate": True,
            },
            {
                "ticket_id": "M-2002",
                "customer_tier": "enterprise",
                "issue_type": "access",
                "urgency": "critical",
                "sentiment": "angry",
                "sla_minutes_remaining": 35,
                "required_team": "access",
                "required_priority": "P1",
                "should_escalate": True,
            },
            {
                "ticket_id": "M-2003",
                "customer_tier": "pro",
                "issue_type": "product",
                "urgency": "high",
                "sentiment": "frustrated",
                "sla_minutes_remaining": 95,
                "required_team": "product",
                "required_priority": "P2",
                "should_escalate": False,
            },
            {
                "ticket_id": "M-2004",
                "customer_tier": "free",
                "issue_type": "billing",
                "urgency": "low",
                "sentiment": "calm",
                "sla_minutes_remaining": 220,
                "required_team": "billing",
                "required_priority": "P3",
                "should_escalate": False,
            },
            {
                "ticket_id": "M-2005",
                "customer_tier": "pro",
                "issue_type": "infra",
                "urgency": "medium",
                "sentiment": "frustrated",
                "sla_minutes_remaining": 120,
                "required_team": "infra",
                "required_priority": "P2",
                "should_escalate": False,
            },
        ),
    ),
    TaskSpec(
        task_id="hard_multi_incident_shift",
        title="Run a full on-call support shift",
        difficulty="hard",
        max_steps=30,
        objective=(
            "Control a high-volume queue with multiple critical incidents where poor "
            "prioritization can cause cascading SLA failures."
        ),
        tickets=(
            {
                "ticket_id": "H-3001",
                "customer_tier": "enterprise",
                "issue_type": "security",
                "urgency": "critical",
                "sentiment": "angry",
                "sla_minutes_remaining": 30,
                "required_team": "security",
                "required_priority": "P1",
                "should_escalate": True,
            },
            {
                "ticket_id": "H-3002",
                "customer_tier": "enterprise",
                "issue_type": "infra",
                "urgency": "critical",
                "sentiment": "angry",
                "sla_minutes_remaining": 25,
                "required_team": "infra",
                "required_priority": "P1",
                "should_escalate": True,
            },
            {
                "ticket_id": "H-3003",
                "customer_tier": "pro",
                "issue_type": "access",
                "urgency": "high",
                "sentiment": "frustrated",
                "sla_minutes_remaining": 75,
                "required_team": "access",
                "required_priority": "P1",
                "should_escalate": True,
            },
            {
                "ticket_id": "H-3004",
                "customer_tier": "free",
                "issue_type": "billing",
                "urgency": "medium",
                "sentiment": "calm",
                "sla_minutes_remaining": 150,
                "required_team": "billing",
                "required_priority": "P3",
                "should_escalate": False,
            },
            {
                "ticket_id": "H-3005",
                "customer_tier": "pro",
                "issue_type": "infra",
                "urgency": "high",
                "sentiment": "frustrated",
                "sla_minutes_remaining": 70,
                "required_team": "infra",
                "required_priority": "P2",
                "should_escalate": False,
            },
            {
                "ticket_id": "H-3006",
                "customer_tier": "enterprise",
                "issue_type": "product",
                "urgency": "high",
                "sentiment": "frustrated",
                "sla_minutes_remaining": 80,
                "required_team": "product",
                "required_priority": "P2",
                "should_escalate": False,
            },
            {
                "ticket_id": "H-3007",
                "customer_tier": "pro",
                "issue_type": "security",
                "urgency": "medium",
                "sentiment": "frustrated",
                "sla_minutes_remaining": 110,
                "required_team": "security",
                "required_priority": "P2",
                "should_escalate": True,
            },
            {
                "ticket_id": "H-3008",
                "customer_tier": "free",
                "issue_type": "access",
                "urgency": "low",
                "sentiment": "calm",
                "sla_minutes_remaining": 180,
                "required_team": "access",
                "required_priority": "P3",
                "should_escalate": False,
            },
        ),
    ),
)


def list_task_specs() -> list[TaskSpec]:
    return list(TASK_SPECS)


def get_task_spec(task_id: str) -> TaskSpec:
    for task in TASK_SPECS:
        if task.task_id == task_id:
            return task
    valid = ", ".join(t.task_id for t in TASK_SPECS)
    raise KeyError(f"Unknown task_id '{task_id}'. Valid values: {valid}")


def default_task_id() -> str:
    return TASK_SPECS[0].task_id
