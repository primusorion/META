from __future__ import annotations

from collections import Counter

from .models import Action, TicketState
from .tasks import TaskSpec


SCORE_EPSILON = 1e-3


def _ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _to_open_unit_interval(value: float) -> float:
    """Map [0, 1] to (0, 1) so scores are never exactly 0.0 or 1.0."""
    clipped = _clip01(value)
    return SCORE_EPSILON + (1.0 - 2.0 * SCORE_EPSILON) * clipped


def grade_episode(
    task: TaskSpec,
    tickets: list[TicketState],
    action_history: list[Action],
    steps_taken: int,
) -> tuple[float, dict[str, float]]:
    """Return a deterministic score in the 0.0-1.0 range with detailed metrics."""

    total = len(tickets)
    resolved = sum(1 for t in tickets if t.status == "resolved")
    breached = sum(1 for t in tickets if t.breached and t.status != "resolved")

    critical_tickets = [t for t in tickets if t.urgency == "critical"]
    critical_resolved = sum(1 for t in critical_tickets if t.status == "resolved")

    correct_priority = sum(1 for t in tickets if t.assigned_priority == t.required_priority)
    correct_team = sum(1 for t in tickets if t.assigned_team == t.required_team)
    correct_escalation = sum(1 for t in tickets if t.escalated == t.should_escalate)

    # Penalize unproductive behavior such as repeated no-ops.
    action_counts = Counter(a.action_type for a in action_history)
    noops = action_counts.get("noop", 0)
    excessive_noop_penalty = _clip01(noops / max(1, len(action_history)))

    resolution_score = _ratio(resolved, total)
    critical_score = _ratio(critical_resolved, len(critical_tickets)) if critical_tickets else 1.0
    routing_score = 0.5 * (_ratio(correct_priority, total) + _ratio(correct_team, total))
    escalation_score = _ratio(correct_escalation, total)
    sla_score = _clip01(1.0 - _ratio(breached, total))
    efficiency_score = _clip01(1.0 - _ratio(steps_taken, task.max_steps))
    behavior_score = _clip01(1.0 - excessive_noop_penalty)

    if task.difficulty == "easy":
        weights = {
            "resolution": 0.25,
            "critical": 0.30,
            "routing": 0.20,
            "escalation": 0.10,
            "sla": 0.10,
            "behavior": 0.05,
        }
    elif task.difficulty == "medium":
        weights = {
            "resolution": 0.25,
            "critical": 0.25,
            "routing": 0.20,
            "escalation": 0.15,
            "sla": 0.10,
            "behavior": 0.05,
        }
    else:
        weights = {
            "resolution": 0.20,
            "critical": 0.25,
            "routing": 0.20,
            "escalation": 0.15,
            "sla": 0.10,
            "efficiency": 0.05,
            "behavior": 0.05,
        }

    metrics = {
        "resolution": resolution_score,
        "critical": critical_score,
        "routing": routing_score,
        "escalation": escalation_score,
        "sla": sla_score,
        "efficiency": efficiency_score,
        "behavior": behavior_score,
    }

    score = 0.0
    for key, weight in weights.items():
        score += weight * metrics[key]

    breakdown = {
        "score": _to_open_unit_interval(score),
        "resolution_score": resolution_score,
        "critical_score": critical_score,
        "routing_score": routing_score,
        "escalation_score": escalation_score,
        "sla_score": sla_score,
        "efficiency_score": efficiency_score,
        "behavior_score": behavior_score,
    }

    return breakdown["score"], breakdown
