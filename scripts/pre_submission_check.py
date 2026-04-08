from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.env import SupportOpsEnv
from app.models import Action
from app.tasks import list_task_specs


def _check_openenv_yaml() -> None:
    if not Path("openenv.yaml").exists():
        raise RuntimeError("Missing openenv.yaml")


def _check_core_api() -> None:
    env = SupportOpsEnv()
    obs = env.reset(task_id="easy_priority_rescue", seed=0)
    assert obs.task_id == "easy_priority_rescue"

    step_obs, reward, done, info = env.step(Action(action_type="noop"))
    assert step_obs.step_count == 1
    assert isinstance(reward.value, float)
    assert 0.0 <= reward.value <= 1.0
    assert isinstance(done, bool)
    assert "step_count" in info

    state = env.state()
    assert state.task_id == "easy_priority_rescue"


def _check_task_count_and_score_range() -> None:
    env = SupportOpsEnv()
    tasks = list_task_specs()
    if len(tasks) < 3:
        raise RuntimeError("Need at least 3 tasks")

    for index, task in enumerate(tasks):
        env.reset(task_id=task.task_id, seed=100 + index)

        while True:
            state = env.state()
            if state.done:
                break
            unresolved = [ticket for ticket in state.tickets if ticket.status != "resolved"]
            if not unresolved:
                break
            unresolved.sort(key=lambda t: t.sla_minutes_remaining)
            target = unresolved[0]

            if state.active_ticket_id != target.ticket_id:
                env.step(Action(action_type="select_ticket", ticket_id=target.ticket_id))
                continue
            if target.assigned_priority != target.required_priority:
                env.step(Action(action_type="set_priority", ticket_id=target.ticket_id, value=target.required_priority))
                continue
            if target.assigned_team != target.required_team:
                env.step(Action(action_type="assign_team", ticket_id=target.ticket_id, value=target.required_team))
                continue
            if target.should_escalate and not target.escalated:
                env.step(Action(action_type="escalate", ticket_id=target.ticket_id))
                continue
            env.step(Action(action_type="resolve_ticket", ticket_id=target.ticket_id))

        score = float(env.state().final_score or 0.0)
        if score < 0.0 or score > 1.0:
            raise RuntimeError(f"Task {task.task_id} score out of range: {score}")


def main() -> None:
    _check_openenv_yaml()
    _check_core_api()
    _check_task_count_and_score_range()
    print("Pre-submission checks passed")


if __name__ == "__main__":
    main()
