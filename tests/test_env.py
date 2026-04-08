from app.env import SupportOpsEnv
from app.models import Action
from app.tasks import list_task_specs


def _oracle_episode(env: SupportOpsEnv, task_id: str) -> float:
    env.reset(task_id=task_id, seed=123)

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
            env.step(
                Action(
                    action_type="set_priority",
                    ticket_id=target.ticket_id,
                    value=target.required_priority,
                )
            )
            continue

        if target.assigned_team != target.required_team:
            env.step(
                Action(
                    action_type="assign_team",
                    ticket_id=target.ticket_id,
                    value=target.required_team,
                )
            )
            continue

        if target.should_escalate and not target.escalated:
            env.step(Action(action_type="escalate", ticket_id=target.ticket_id))
            continue

        env.step(Action(action_type="resolve_ticket", ticket_id=target.ticket_id))

    final_state = env.state()
    return float(final_state.final_score or 0.0)


def test_reset_returns_clean_state() -> None:
    env = SupportOpsEnv()
    observation = env.reset(task_id="easy_priority_rescue", seed=0)
    state = env.state()

    assert observation.step_count == 0
    assert state.step_count == 0
    assert state.done is False
    assert len(state.tickets) == 3


def test_step_contract_returns_observation_reward_done_info() -> None:
    env = SupportOpsEnv()
    env.reset(task_id="easy_priority_rescue", seed=0)

    observation, reward, done, info = env.step(Action(action_type="noop"))

    assert observation.task_id == "easy_priority_rescue"
    assert isinstance(reward.value, float)
    assert 0.0 <= reward.value <= 1.0
    assert isinstance(done, bool)
    assert "step_count" in info


def test_all_task_graders_return_score_between_zero_and_one() -> None:
    env = SupportOpsEnv()
    for task in list_task_specs():
        score = _oracle_episode(env, task.task_id)
        assert 0.0 < score < 1.0
