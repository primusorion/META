from __future__ import annotations

import random
from typing import Iterable

from .graders import grade_episode
from .models import Action, EnvironmentState, Observation, Reward, TicketPublicView, TicketState
from .tasks import TaskSpec, default_task_id, get_task_spec


class SupportOpsEnv:
    """OpenEnv-style environment for customer support triage operations."""

    def __init__(self, task_id: str | None = None):
        self._rng = random.Random(0)
        self._task: TaskSpec = get_task_spec(task_id or default_task_id())
        self._state = EnvironmentState(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            step_count=0,
            max_steps=self._task.max_steps,
            active_ticket_id=None,
            tickets=[],
            action_history=[],
            cumulative_reward=0.0,
            done=False,
            final_score=None,
        )
        self._last_feedback = ""
        self.reset(task_id=task_id, seed=0)

    def reset(self, task_id: str | None = None, seed: int = 0) -> Observation:
        if task_id is not None:
            self._task = get_task_spec(task_id)

        self._rng = random.Random(seed)
        ticket_rows = [dict(row) for row in self._task.tickets]
        if seed:
            self._rng.shuffle(ticket_rows)

        tickets = [TicketState(**row) for row in ticket_rows]
        self._state = EnvironmentState(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            step_count=0,
            max_steps=self._task.max_steps,
            active_ticket_id=self._pick_next_ticket_id(tickets),
            tickets=tickets,
            action_history=[],
            cumulative_reward=0.0,
            done=False,
            final_score=None,
        )
        self._last_feedback = "Environment reset. Start by selecting the highest-risk ticket."
        return self._build_observation(score_estimate=0.0, last_feedback=self._last_feedback)

    def state(self) -> EnvironmentState:
        return self._state.model_copy(deep=True)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, str | float | int | bool]]:
        if self._state.done:
            observation = self._build_observation(
                score_estimate=self._state.final_score or 0.0,
                last_feedback="Episode already complete. Call reset() to start another run.",
            )
            reward = Reward(
                value=0.0,
                cumulative=self._state.cumulative_reward,
                components={"action": 0.0, "clock": 0.0, "loop": 0.0, "terminal": 0.0},
                rationale="No-op because the episode is already done.",
            )
            return observation, reward, True, {"episode_complete": True}

        self._state.step_count += 1

        action_reward, action_msg, action_error = self._apply_action(action)
        loop_penalty = self._loop_penalty(action)
        self._state.action_history.append(action)

        clock_penalty, new_breaches = self._advance_sla_clock()
        self._refresh_active_ticket()

        score_estimate, _ = grade_episode(
            self._task,
            self._state.tickets,
            self._state.action_history,
            self._state.step_count,
        )

        done = self._all_resolved(self._state.tickets) or self._state.step_count >= self._state.max_steps
        terminal_bonus = 0.0
        final_score = None
        if done:
            final_score, _ = grade_episode(
                self._task,
                self._state.tickets,
                self._state.action_history,
                self._state.step_count,
            )
            self._state.final_score = final_score
            terminal_bonus = 0.2 * (final_score - 0.5)
            score_estimate = final_score

        raw_reward = action_reward + clock_penalty + loop_penalty + terminal_bonus
        reward_value = self._clip01(0.5 + raw_reward)

        # Keep cumulative_reward in 0-1 as a running average for easier cross-task comparison.
        prev_total = self._state.cumulative_reward * (self._state.step_count - 1)
        self._state.cumulative_reward = (prev_total + reward_value) / self._state.step_count
        self._state.done = done

        completed = sum(1 for t in self._state.tickets if t.status == "resolved")
        self._last_feedback = (
            f"{action_msg} Resolved {completed}/{len(self._state.tickets)} tickets. "
            f"New breaches this step: {new_breaches}."
        )

        observation = self._build_observation(score_estimate=score_estimate, last_feedback=self._last_feedback)
        reward = Reward(
            value=reward_value,
            cumulative=self._state.cumulative_reward,
            components={
                "action": action_reward,
                "clock": clock_penalty,
                "loop": loop_penalty,
                "terminal": terminal_bonus,
                "raw": raw_reward,
            },
            rationale=self._last_feedback,
        )

        info: dict[str, str | float | int | bool] = {
            "task_id": self._state.task_id,
            "difficulty": self._state.difficulty,
            "step_count": self._state.step_count,
            "new_breaches": new_breaches,
            "completed_tickets": completed,
        }
        if action_error is not None:
            info["last_action_error"] = action_error
        if final_score is not None:
            info["final_score"] = final_score

        return observation, reward, done, info

    def _apply_action(self, action: Action) -> tuple[float, str, str | None]:
        if action.action_type == "noop":
            return -0.03, "No-op consumed a step without improving queue health.", None

        if action.action_type == "select_ticket":
            if not action.ticket_id:
                message = "select_ticket requires ticket_id."
                return -0.04, message, message
            ticket = self._find_ticket(action.ticket_id)
            if ticket is None:
                message = f"Ticket {action.ticket_id} does not exist."
                return -0.04, message, message
            if ticket.status == "resolved":
                return -0.02, f"Ticket {ticket.ticket_id} is already resolved.", None
            self._state.active_ticket_id = ticket.ticket_id
            return 0.02, f"Selected ticket {ticket.ticket_id}.", None

        ticket = self._resolve_ticket_target(action)
        if ticket is None:
            message = "Action requires a valid active or explicit ticket_id."
            return -0.05, message, message

        if action.action_type == "set_priority":
            if action.value not in {"P1", "P2", "P3"}:
                message = "set_priority requires value in {P1, P2, P3}."
                return -0.04, message, message
            old = ticket.assigned_priority
            ticket.assigned_priority = action.value
            if action.value == ticket.required_priority and old != ticket.required_priority:
                return 0.07, f"Priority for {ticket.ticket_id} set correctly to {action.value}.", None
            if action.value != ticket.required_priority:
                return -0.03, f"Priority {action.value} is suboptimal for {ticket.ticket_id}.", None
            return 0.01, f"Priority for {ticket.ticket_id} remains correct.", None

        if action.action_type == "assign_team":
            if action.value not in {"billing", "infra", "product", "access", "security"}:
                message = "assign_team requires a valid team value."
                return -0.04, message, message
            old = ticket.assigned_team
            ticket.assigned_team = action.value
            if action.value == ticket.required_team and old != ticket.required_team:
                return 0.07, f"Routing for {ticket.ticket_id} set correctly to {action.value}.", None
            if action.value != ticket.required_team:
                return -0.03, f"Team {action.value} is incorrect for {ticket.ticket_id}.", None
            return 0.01, f"Routing for {ticket.ticket_id} remains correct.", None

        if action.action_type == "request_info":
            if ticket.status == "resolved":
                return -0.02, f"Ticket {ticket.ticket_id} is already resolved.", None
            ticket.info_requested = True
            empathy_bonus = 0.02 if ticket.sentiment in {"frustrated", "angry"} else 0.01
            confidence_bonus = 0.01 * action.confidence
            return (
                empathy_bonus + confidence_bonus,
                f"Requested clarifying information for {ticket.ticket_id}.",
                None,
            )

        if action.action_type == "escalate":
            if ticket.escalated:
                return -0.01, f"Ticket {ticket.ticket_id} was already escalated.", None
            ticket.escalated = True
            if ticket.should_escalate:
                return 0.06, f"Correctly escalated {ticket.ticket_id}.", None
            return -0.05, f"Unnecessary escalation for {ticket.ticket_id}.", None

        if action.action_type == "resolve_ticket":
            if ticket.status == "resolved":
                return -0.02, f"Ticket {ticket.ticket_id} is already resolved.", None

            prerequisites_met = (
                ticket.assigned_priority == ticket.required_priority
                and ticket.assigned_team == ticket.required_team
                and (not ticket.should_escalate or ticket.escalated)
            )

            if not prerequisites_met:
                return -0.09, f"Attempted to resolve {ticket.ticket_id} without complete triage.", None

            ticket.status = "resolved"
            ticket.resolution_note = action.note or "Resolved by automated support policy."

            reward = 0.12
            if ticket.sla_minutes_remaining > 0:
                reward += 0.04
            else:
                reward -= 0.03
            if ticket.customer_tier == "enterprise":
                reward += 0.02
            if ticket.urgency == "critical":
                reward += 0.03
            return reward, f"Resolved ticket {ticket.ticket_id}.", None

        message = f"Unsupported action_type: {action.action_type}."
        return -0.05, message, message

    def _advance_sla_clock(self) -> tuple[float, int]:
        new_breaches = 0
        pressure_count = 0
        for ticket in self._state.tickets:
            if ticket.status == "resolved":
                continue

            if ticket.urgency == "critical":
                decay = 15
            elif ticket.urgency == "high":
                decay = 12
            elif ticket.urgency == "medium":
                decay = 9
            else:
                decay = 7

            ticket.sla_minutes_remaining -= decay

            if ticket.sla_minutes_remaining <= 0 and not ticket.breached:
                ticket.breached = True
                new_breaches += 1

            if ticket.sla_minutes_remaining <= 25:
                pressure_count += 1

        clock_penalty = (-0.08 * new_breaches) + (-0.01 * pressure_count)
        return clock_penalty, new_breaches

    def _build_observation(self, score_estimate: float, last_feedback: str) -> Observation:
        queue = sorted(
            [self._to_public_ticket(t) for t in self._state.tickets],
            key=lambda t: (t.status == "resolved", t.sla_minutes_remaining),
        )
        completed = sum(1 for t in self._state.tickets if t.status == "resolved")
        breached = sum(1 for t in self._state.tickets if t.breached and t.status != "resolved")
        return Observation(
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            step_count=self._state.step_count,
            max_steps=self._state.max_steps,
            active_ticket_id=self._state.active_ticket_id,
            queue=queue,
            completed_tickets=completed,
            breached_tickets=breached,
            score_estimate=score_estimate,
            last_feedback=last_feedback,
        )

    @staticmethod
    def _to_public_ticket(ticket: TicketState) -> TicketPublicView:
        return TicketPublicView(
            ticket_id=ticket.ticket_id,
            customer_tier=ticket.customer_tier,
            issue_type=ticket.issue_type,
            urgency=ticket.urgency,
            sentiment=ticket.sentiment,
            sla_minutes_remaining=ticket.sla_minutes_remaining,
            status=ticket.status,
            assigned_team=ticket.assigned_team,
            assigned_priority=ticket.assigned_priority,
            escalated=ticket.escalated,
        )

    def _refresh_active_ticket(self) -> None:
        active = self._find_ticket(self._state.active_ticket_id) if self._state.active_ticket_id else None
        if active is not None and active.status != "resolved":
            return
        self._state.active_ticket_id = self._pick_next_ticket_id(self._state.tickets)

    @staticmethod
    def _all_resolved(tickets: Iterable[TicketState]) -> bool:
        return all(ticket.status == "resolved" for ticket in tickets)

    @staticmethod
    def _pick_next_ticket_id(tickets: Iterable[TicketState]) -> str | None:
        unresolved = [t for t in tickets if t.status != "resolved"]
        if not unresolved:
            return None
        unresolved.sort(key=lambda t: (t.sla_minutes_remaining, t.customer_tier != "enterprise"))
        return unresolved[0].ticket_id

    def _find_ticket(self, ticket_id: str | None) -> TicketState | None:
        if ticket_id is None:
            return None
        for ticket in self._state.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _resolve_ticket_target(self, action: Action) -> TicketState | None:
        target_id = action.ticket_id or self._state.active_ticket_id
        return self._find_ticket(target_id)

    def _loop_penalty(self, action: Action) -> float:
        if len(self._state.action_history) < 2:
            return 0.0

        last = self._state.action_history[-1]
        prev = self._state.action_history[-2]
        same_action = last.action_type == prev.action_type == action.action_type
        same_target = (last.ticket_id or self._state.active_ticket_id) == (
            prev.ticket_id or self._state.active_ticket_id
        ) == (action.ticket_id or self._state.active_ticket_id)

        if same_action and same_target:
            return -0.04
        return 0.0

    @staticmethod
    def _clip01(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
