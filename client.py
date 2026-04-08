from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SupportOpsTriageAction, SupportOpsTriageObservation


class SupportOpsTriageEnv(
    EnvClient[SupportOpsTriageAction, SupportOpsTriageObservation, State]
):
    """OpenEnv client wrapper for the Support Ops triage environment."""

    def _step_payload(self, action: SupportOpsTriageAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SupportOpsTriageObservation]:
        obs_data = payload.get("observation", payload)
        observation = SupportOpsTriageObservation(
            task_id=obs_data.get("task_id", ""),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 0),
            active_ticket_id=obs_data.get("active_ticket_id"),
            completed_tickets=obs_data.get("completed_tickets", 0),
            breached_tickets=obs_data.get("breached_tickets", 0),
            score_estimate=obs_data.get("score_estimate", 0.0),
            last_feedback=obs_data.get("last_feedback", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", {}).get("value") if isinstance(payload.get("reward"), dict) else payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("task_id"),
            step_count=payload.get("step_count", 0),
        )
