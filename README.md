---
title: Support Ops Triage OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Support Ops Triage OpenEnv

A real-world OpenEnv environment where an agent acts as a support operations specialist.
The agent must triage customer tickets under SLA pressure and choose concrete operational actions:
priority assignment, team routing, escalation, and resolution.

This project is designed for OpenEnv-style evaluation and includes:
- Typed `Observation`, `Action`, and `Reward` Pydantic models
- Full `step(action)`, `reset()`, and `state()` interface
- Three graded tasks with deterministic score outputs in `0.0-1.0`
- Meaningful reward shaping with partial progress and anti-loop penalties
- Reproducible baseline inference script (`inference.py`)
- Containerized deployment assets for Hugging Face Spaces

## Why this is real-world

Support triage is a production problem in SaaS companies. Incorrect routing, delayed escalation,
or poor prioritization creates SLA breaches, customer churn risk, and operational cost.
This environment models those tradeoffs directly.

## Environment API

The FastAPI server exposes OpenEnv endpoints:
- `POST /reset` -> returns initial `Observation`
- `POST /step` -> accepts `Action`, returns `StepResult` (`observation`, `reward`, `done`, `info`)
- `GET /state` -> returns full internal `EnvironmentState`
- `GET /tasks` -> returns available tasks
- `GET /health` -> liveness

Start server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

## Action space

`Action.action_type` supports:
- `select_ticket`
- `set_priority`
- `assign_team`
- `request_info`
- `escalate`
- `resolve_ticket`
- `noop`

Action fields:
- `ticket_id`: target ticket
- `value`: used by `set_priority` or `assign_team`
- `note`: optional resolution note
- `confidence`: `0.0-1.0`

## Observation space

`Observation` includes:
- Current task metadata (`task_id`, `difficulty`)
- Episode progress (`step_count`, `max_steps`)
- Queue snapshot (`queue` with ticket public fields)
- Active ticket pointer (`active_ticket_id`)
- Partial performance signals (`completed_tickets`, `breached_tickets`, `score_estimate`, `last_feedback`)

## Task suite and graders

Three deterministic tasks are provided (`easy -> medium -> hard`):

1. `easy_priority_rescue`
2. `medium_sla_balancing`
3. `hard_multi_incident_shift`

All tasks use the same grading contract and return final score in `0.0-1.0`.
Graders evaluate:
- Resolution coverage
- Critical incident handling
- Routing correctness
- Escalation correctness
- SLA outcomes
- Efficiency and behavior quality (difficulty-dependent)

## Reward shaping

Per-step reward is composed of:
- Action quality signal (correct triage actions vs harmful actions)
- SLA clock penalty (new breaches and queue pressure)
- Loop penalty (repeating unproductive actions)
- Terminal bonus aligned with final grader score

The raw shaping signal is normalized into the `0.0-1.0` range before it is returned
in `Reward.value`, so both grader scores and API reward outputs are validator-friendly.

## Local setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

Run environment API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Run baseline inference (reproducible deterministic baseline):

```bash
python inference.py --policy heuristic --seed 7
```

Expected deterministic scores with `--seed 7`:

| Task | Difficulty | Score |
|---|---|---|
| easy_priority_rescue | easy | 1.0000 |
| medium_sla_balancing | medium | 1.0000 |
| hard_multi_incident_shift | hard | 0.9533 |
| Average | - | 0.9844 |

Run submission-style LLM mode (OpenAI client, mandatory env vars):

```bash
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set HF_TOKEN=your_token
set LOCAL_IMAGE_NAME=your-local-image-name
python inference.py --policy llm --seed 7
```

## Structured stdout format

`inference.py` emits only these line types:
- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

Formatting guarantees:
- `reward` and each value in `rewards` are formatted to 2 decimals.
- `done` and `success` are lowercase booleans.
- `error` is raw `last_action_error` when present, otherwise `null`.
- No extra stdout line types are emitted.

## Pre-submission validation

```bash
python scripts/pre_submission_check.py
```

Additional end-to-end validator script:

```bash
bash scripts/validate-submission.sh https://your-space.hf.space
```

Official metadata validator:

```bash
openenv validate
```

## Tests

```bash
pytest -q
```

## Docker and Hugging Face Space

Build image:

```bash
docker build -t support-ops-openenv .
```

Run locally:

```bash
docker run --rm -p 7860:7860 support-ops-openenv
```

For Hugging Face Spaces:
1. Create a new Space with SDK type `Docker`.
2. Push this repository.
3. Ensure Space starts on port `7860`.
4. Verify `GET /health`, `POST /reset`, `POST /step`, and `GET /state`.

## Project structure

```text
app/
  __init__.py
  env.py
  graders.py
  main.py
  models.py
  tasks.py
scripts/
  pre_submission_check.py
  validate-submission.sh
server/
  app.py
tests/
  test_env.py
inference.py
openenv.yaml
pyproject.toml
uv.lock
Dockerfile
requirements.txt
README.md
```
