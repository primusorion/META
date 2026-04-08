from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import requests
from openai import OpenAI

from app.env import SupportOpsEnv
from app.models import Action, EnvironmentState, Observation, StepResult
from app.tasks import get_task_spec, list_task_specs

URGENCY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("BENCHMARK") or "support_ops_triage"
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "220"))


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object")

    return json.loads(cleaned[start : end + 1])


def action_to_string(action: Action) -> str:
    payload = action.model_dump(exclude_none=True)
    note = payload.get("note")
    if isinstance(note, str):
        payload["note"] = "_".join(note.split())
    return json.dumps(payload, separators=(",", ":"))


class LocalRunner:
    def __init__(self) -> None:
        self.env = SupportOpsEnv()

    def reset(self, task_id: str, seed: int) -> Observation:
        return self.env.reset(task_id=task_id, seed=seed)

    def step(self, action: Action) -> StepResult:
        observation, reward, done, info = self.env.step(action)
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def state(self) -> EnvironmentState:
        return self.env.state()

    def close(self) -> None:
        return None


class RemoteRunner:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str, seed: int) -> Observation:
        response = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        response.raise_for_status()
        return Observation.model_validate(response.json())

    def step(self, action: Action) -> StepResult:
        response = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=30,
        )
        response.raise_for_status()
        return StepResult.model_validate(response.json())

    def state(self) -> EnvironmentState:
        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return EnvironmentState.model_validate(response.json())

    def close(self) -> None:
        return None


class LLMPolicy:
    def __init__(self, model_name: str, api_base_url: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)

    def act(self, observation: Observation) -> Action:
        payload = {
            "task_id": observation.task_id,
            "difficulty": observation.difficulty,
            "step_count": observation.step_count,
            "max_steps": observation.max_steps,
            "active_ticket_id": observation.active_ticket_id,
            "queue": [ticket.model_dump() for ticket in observation.queue],
            "last_feedback": observation.last_feedback,
        }

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer-support triage agent. "
                        "Return exactly one JSON object with keys: "
                        "action_type, ticket_id, value, note, confidence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Choose one best action for the next step. "
                        "Prioritize preventing SLA breaches and resolving critical incidents.\n"
                        f"Observation: {json.dumps(payload, separators=(',', ':'))}"
                    ),
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        text = (completion.choices[0].message.content or "{}").strip()
        return Action.model_validate(_extract_json(text))


def choose_action_heuristic(observation: Observation, state: EnvironmentState) -> Action:
    unresolved = [ticket for ticket in state.tickets if ticket.status != "resolved"]
    if not unresolved:
        return Action(action_type="noop", confidence=1.0)

    unresolved.sort(key=lambda ticket: (ticket.sla_minutes_remaining, URGENCY_RANK[ticket.urgency]))
    target = unresolved[0]

    if observation.active_ticket_id != target.ticket_id:
        return Action(action_type="select_ticket", ticket_id=target.ticket_id, confidence=1.0)

    if target.assigned_priority != target.required_priority:
        return Action(
            action_type="set_priority",
            ticket_id=target.ticket_id,
            value=target.required_priority,
            confidence=0.95,
        )

    if target.assigned_team != target.required_team:
        return Action(
            action_type="assign_team",
            ticket_id=target.ticket_id,
            value=target.required_team,
            confidence=0.95,
        )

    if target.should_escalate and not target.escalated:
        return Action(action_type="escalate", ticket_id=target.ticket_id, confidence=0.90)

    return Action(
        action_type="resolve_ticket",
        ticket_id=target.ticket_id,
        note="Resolved after priority, routing, and escalation checks.",
        confidence=0.90,
    )


def run_episode(
    runner: LocalRunner | RemoteRunner,
    task_id: str,
    seed: int,
    policy: str,
    llm_policy: LLMPolicy | None,
) -> None:
    observation = runner.reset(task_id=task_id, seed=seed)
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        done = False
        while not done and observation.step_count < observation.max_steps:
            state = runner.state()
            if policy == "llm":
                assert llm_policy is not None
                try:
                    action = llm_policy.act(observation)
                except Exception as exc:
                    print(f"[DEBUG] LLM fallback due to error: {exc}", file=sys.stderr, flush=True)
                    action = choose_action_heuristic(observation, state)
            else:
                action = choose_action_heuristic(observation, state)

            step_result = runner.step(action)
            observation = step_result.observation
            done = step_result.done

            reward_value = _clip01(float(step_result.reward.value))
            rewards.append(reward_value)
            steps_taken = observation.step_count

            error_raw = step_result.info.get("last_action_error")
            error = str(error_raw) if isinstance(error_raw, str) else None
            log_step(
                step=steps_taken,
                action=action_to_string(action),
                reward=reward_value,
                done=done,
                error=error,
            )

        final_state = runner.state()
        score = _clip01(float(final_state.final_score or 0.0))
        success = bool(final_state.done) and score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"[DEBUG] Episode exception on task={task_id}: {exc}", file=sys.stderr, flush=True)
    finally:
        try:
            runner.close()
        except Exception as exc:
            print(f"[DEBUG] runner.close() error: {exc}", file=sys.stderr, flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hackathon inference script for Support Ops OpenEnv")
    parser.add_argument("--seed", type=int, default=7, help="Base seed for deterministic resets")
    parser.add_argument(
        "--policy",
        choices=["llm", "heuristic"],
        default=os.getenv("INFERENCE_POLICY", "llm"),
        help="Action policy. Use llm for submission-style runs.",
    )
    parser.add_argument(
        "--remote-url",
        type=str,
        default=os.getenv("ENV_BASE_URL", ""),
        help="Optional base URL for a deployed environment",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=os.getenv("TASK_NAME", ""),
        help="Optional single task id. By default runs all tasks.",
    )
    args = parser.parse_args()

    # Read and retain LOCAL_IMAGE_NAME for spec parity when users run from docker-image wrappers.
    _ = LOCAL_IMAGE_NAME

    if args.remote_url:
        runner: LocalRunner | RemoteRunner = RemoteRunner(args.remote_url)
    else:
        runner = LocalRunner()

    llm_policy = None
    if args.policy == "llm":
        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN must be set for --policy llm")
        llm_policy = LLMPolicy(model_name=MODEL_NAME, api_base_url=API_BASE_URL, api_key=HF_TOKEN)

    if args.task:
        get_task_spec(args.task)
        task_ids = [args.task]
    else:
        task_ids = [task.task_id for task in list_task_specs()]

    for index, task_id in enumerate(task_ids):
        run_episode(
            runner=runner,
            task_id=task_id,
            seed=args.seed + index,
            policy=args.policy,
            llm_policy=llm_policy,
        )


if __name__ == "__main__":
    main()
