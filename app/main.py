from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from .env import SupportOpsEnv
from .models import Action, EnvironmentState, Observation, ResetRequest, StepResult, TaskSummary
from .tasks import get_task_spec, list_task_specs

app = FastAPI(
    title="Support Ops Triage OpenEnv",
    version="1.0.0",
    description="Real-world OpenEnv for customer support queue triage and SLA rescue.",
)

_env = SupportOpsEnv()
_env_lock = Lock()


def _api_manifest() -> dict[str, str]:
    return {
        "name": "support-ops-triage-openenv",
        "health": "/health",
        "tasks": "/tasks",
        "reset": "POST /reset",
        "step": "POST /step",
        "state": "/state",
    }


def _web_panel_html() -> str:
    return """
<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Support Ops Triage OpenEnv</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

            :root {
                --bg: #0c111b;
                --panel: #121a2a;
                --panel-soft: #18243b;
                --line: #2c3b58;
                --text: #e8eefb;
                --muted: #9eb1d8;
                --accent: #2dd4bf;
                --accent-2: #60a5fa;
            }

            * { box-sizing: border-box; }
            body {
                margin: 0;
                min-height: 100vh;
                font-family: 'Space Grotesk', Segoe UI, Arial, sans-serif;
                color: var(--text);
                background:
                    radial-gradient(circle at 15% 15%, #1d2b4a 0%, transparent 35%),
                    radial-gradient(circle at 85% 20%, #1d3a4a 0%, transparent 40%),
                    linear-gradient(135deg, #0a0f19 0%, #0c111b 45%, #0f1728 100%);
                padding: 28px;
            }

            .layout {
                max-width: 980px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 1fr;
                gap: 16px;
            }

            .hero {
                background: linear-gradient(160deg, rgba(45, 212, 191, 0.10), rgba(96, 165, 250, 0.12));
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 22px;
            }

            .hero h1 {
                margin: 0 0 10px;
                font-size: 30px;
                letter-spacing: 0.2px;
            }

            .hero p {
                margin: 0;
                color: var(--muted);
                max-width: 68ch;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 12px;
            }

            .card {
                background: linear-gradient(160deg, var(--panel) 0%, var(--panel-soft) 100%);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 14px;
            }

            .label {
                display: inline-block;
                border: 1px solid rgba(96, 165, 250, 0.45);
                color: var(--accent-2);
                border-radius: 999px;
                padding: 4px 10px;
                font-size: 12px;
                margin-bottom: 8px;
            }

            .card h2 {
                margin: 0 0 10px;
                font-size: 17px;
            }

            .list {
                margin: 0;
                padding-left: 18px;
                color: var(--muted);
            }

            .list li { margin: 6px 0; }

            code {
                background: rgba(15, 23, 42, 0.65);
                border: 1px solid #344766;
                color: #d6e3ff;
                border-radius: 8px;
                padding: 2px 6px;
                font-family: Consolas, "Courier New", monospace;
            }

            a {
                color: var(--accent);
                text-decoration: none;
            }

            a:hover { text-decoration: underline; }

            .footer {
                color: #85a0cf;
                font-size: 13px;
                text-align: center;
                margin-top: 6px;
            }
        </style>
    </head>
    <body>
        <div class="layout">
            <section class="hero">
                <h1>Support Ops Triage OpenEnv</h1>
                <p>
                    This Space hosts a real-world, API-first environment for evaluating triage agents under SLA pressure.
                    Use the endpoints below or open the interactive API docs.
                </p>
            </section>

            <section class="grid">
                <article class="card">
                    <span class="label">Core API</span>
                    <h2>Environment Endpoints</h2>
                    <ul class="list">
                        <li><a href="/health">GET /health</a></li>
                        <li><a href="/tasks">GET /tasks</a></li>
                        <li><a href="/state">GET /state</a></li>
                        <li><code>POST /reset</code></li>
                        <li><code>POST /step</code></li>
                    </ul>
                </article>

                <article class="card">
                    <span class="label">Explore</span>
                    <h2>Interactive Docs</h2>
                    <p style="margin: 0 0 8px; color: var(--muted);">
                        Test requests directly in the generated Swagger interface.
                    </p>
                    <p style="margin: 0;">
                        <a href="/docs">Open /docs</a>
                    </p>
                </article>

                <article class="card">
                    <span class="label">Machine Readable</span>
                    <h2>API Manifest</h2>
                    <p style="margin: 0 0 8px; color: var(--muted);">
                        Use <code>/api</code> for the JSON endpoint manifest.
                    </p>
                    <p style="margin: 0;">
                        <a href="/api">Open /api</a>
                    </p>
                </article>
            </section>

            <p class="footer">
                API-first service is healthy when <code>/health</code> returns <code>{"status":"ok"}</code>.
            </p>
        </div>
    </body>
</html>
""".strip()


@app.get("/", response_model=None)
def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=_web_panel_html())
    return _api_manifest()


@app.get("/api")
def api_manifest() -> dict[str, str]:
    return _api_manifest()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/web", include_in_schema=False, response_class=HTMLResponse)
def web_root() -> str:
    # HF Space App panel may request /web, so serve a friendly landing page.
    return _web_panel_html()


@app.get("/web/{subpath:path}", include_in_schema=False, response_class=HTMLResponse)
def web_any(subpath: str) -> str:
    # Catch additional HF panel routes under /web/* and avoid 404 noise in UI.
    return _web_panel_html()


@app.get("/tasks", response_model=list[TaskSummary])
def tasks() -> list[TaskSummary]:
    return [
        TaskSummary(
            task_id=t.task_id,
            title=t.title,
            difficulty=t.difficulty,
            max_steps=t.max_steps,
            objective=t.objective,
        )
        for t in list_task_specs()
    ]


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest) -> Observation:
    with _env_lock:
        if request.task_id is not None:
            try:
                get_task_spec(request.task_id)
            except KeyError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _env.reset(task_id=request.task_id, seed=request.seed)


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    with _env_lock:
        observation, reward, done, info = _env.step(action)
        return StepResult(observation=observation, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    with _env_lock:
        return _env.state()
