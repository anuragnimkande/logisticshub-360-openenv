"""
LogisticsHub-360: Interactive Gradio Web Application
Hugging Face Spaces-compatible demo with two modes:
  1. AI Agent Mode  — watch an LLM solve logistics tasks in real-time
  2. Human Mode     — play as the agent and compete against the AI score
"""
from __future__ import annotations

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import gradio as gr

from env.environment import LogisticsHub360Env
from typing import Any, Dict, Generator, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import LogisticsHub360Env
from env.models import Action, ToolName
from env.tasks import TASK_BUILDERS, TASK_ORDER
from env.utils import observation_to_prompt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_URL = os.environ.get("LH360_BASE_URL", "https://router.huggingface.co/v1")
MODEL_ID = os.environ.get("LH360_MODEL", "Qwen/Qwen2.5-72B-Instruct")

TASK_LABELS = {
    "order_tracking":     "🟢  Task 1 — Order Tracking (Easy)",
    "shipment_rerouting": "🟡  Task 2 — Shipment Rerouting (Medium)",
    "stockout_crisis":    "🔴  Task 3 — Stockout Crisis Resolution (Hard)",
}
TOOL_LABELS = [t.value for t in ToolName]

SYSTEM_PROMPT = """You are an expert AI logistics operations agent inside LogisticsHub-360.
Respond ONLY with a single JSON object. No prose outside the JSON.
{"tool": "<tool_name>", "parameters": {"<key>": "<value>"}}

Available tools: get_tracking, check_inventory, find_warehouse, reroute_order, issue_refund, update_crm
Rules:
1. Call get_tracking FIRST.
2. Call check_inventory before rerouting or refunding.
3. Use find_warehouse before reroute_order.
4. If fully out of stock, issue_refund — do NOT reroute.
5. Always end with update_crm.
"""

# ---------------------------------------------------------------------------
# Shared state store (Gradio State)
# ---------------------------------------------------------------------------

def fresh_session() -> Dict[str, Any]:
    return {
        "env": None,
        "obs": None,
        "done": False,
        "cumulative_reward": 0.0,
        "step_logs": [],
        "task_id": None,
        "messages": [],          # LLM message history
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_obs(obs: Any) -> str:
    """Render observation as clean markdown for display."""
    if obs is None:
        return "_No task loaded yet. Click **Start Task** to begin._"

    o = obs
    lines = [
        f"## Task: `{o.task_id}` — Difficulty: **{o.difficulty.upper()}**",
        f"**Step:** {o.step_count} / {o.max_steps} &nbsp;|&nbsp; "
        f"**Sentiment:** {o.customer_sentiment:.2f} &nbsp;|&nbsp; "
        f"**Last Reward:** `{o.last_reward if o.last_reward is not None else 'N/A'}`",
        "",
        "### 📋 Task Description",
        o.task_description,
        "",
    ]

    if o.order_status:
        ost = o.order_status
        lines += [
            "### 📦 Order Status",
            f"| Field | Value |",
            f"|---|---|",
            f"| Order ID | `{ost.order_id}` |",
            f"| Status | **{ost.status}** |",
            f"| Product | {ost.product_name} |",
            f"| Carrier | {ost.carrier} |",
            f"| Destination | {ost.destination} |",
            f"| Est. Delivery | {ost.estimated_delivery} |",
            f"| Last Location | {ost.last_known_location or 'Unknown'} |",
        ]
        if ost.delay_reason:
            lines.append(f"| Delay Reason | ⚠️ {ost.delay_reason} |")
        lines.append("")

    if o.inventory_state:
        inv = o.inventory_state
        lvl_icon = {"high": "🟢", "low": "🟡", "out_of_stock": "🔴"}.get(inv.level, "⚪")
        lines += [
            "### 🏭 Inventory State",
            f"**{inv.product_name}** — Level: {lvl_icon} `{inv.level}` ({inv.quantity} units)",
            "",
        ]

    if o.system_logs:
        lines += ["### 🖥️ System Logs _(last 5)_"]
        for log in o.system_logs[-5:]:
            lines.append(f"> {log}")
        lines.append("")

    if o.hints:
        lines += ["### 💡 Hints"]
        for h in o.hints:
            lines.append(f"- {h}")
        lines.append("")

    return "\n".join(lines)


def _render_history(step_logs: List[Dict[str, Any]]) -> str:
    if not step_logs:
        return "_No actions taken yet._"
    rows = ["| Step | Tool | Reward | Result |", "|---|---|---|---|"]
    for e in step_logs:
        r = e["reward"]
        sign = "🟢" if r > 0 else ("🔴" if r < 0 else "⚪")
        rows.append(
            f"| {e['step']} | `{e['tool']}` | {sign} `{r:+.3f}` | {e.get('result_summary', '')[:60]} |"
        )
    return "\n".join(rows)


def _reward_chart_data(step_logs: List[Dict[str, Any]]) -> List[List]:
    """Return data for reward bar chart."""
    if not step_logs:
        return []
    cumulative = 0.0
    result = []
    for e in step_logs:
        cumulative += e["reward"]
        result.append([f"Step {e['step']}: {e['tool']}", e["reward"], round(cumulative, 3)])
    return result


def _grade_bar(grade: float) -> str:
    n = int(grade * 20)
    bar = "█" * n + "░" * (20 - n)
    pct = int(grade * 100)
    color = "🟢" if grade >= 0.8 else ("🟡" if grade >= 0.5 else "🔴")
    return f"{color} [{bar}] {pct}%"


# ---------------------------------------------------------------------------
# Reset / Start Task
# ---------------------------------------------------------------------------

def start_task(task_id: str, session: Dict) -> Tuple[str, str, str, str, Dict]:
    """Initialize the environment for the selected task."""
    env = LogisticsHub360Env(task_id=task_id)
    obs = env.reset()
    session = fresh_session()
    session["env"] = env
    session["obs"] = obs
    session["task_id"] = task_id
    session["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]

    obs_md = _render_obs(obs)
    history_md = _render_history([])
    status_md = f"**Task loaded:** {TASK_LABELS.get(task_id, task_id)}\n\nClick **Run AI Step** or select a tool manually below."
    grade_md = ""
    return obs_md, history_md, status_md, grade_md, session


# ---------------------------------------------------------------------------
# AI Step
# ---------------------------------------------------------------------------

def ai_step(session: Dict) -> Tuple[str, str, str, str, Dict]:
    """Run one step with the LLM agent."""
    if session.get("env") is None:
        return "_No task loaded._", "", "Please start a task first.", "", session
    if session.get("done"):
        return _render_obs(session["obs"]), _render_history(session["step_logs"]), \
               "**Episode complete.** Restart to play again.", _compute_grade_md(session), session

    if not HF_TOKEN:
        return (
            _render_obs(session["obs"]),
            _render_history(session["step_logs"]),
            "**ERROR:** `HF_TOKEN` not set. Add it to Space secrets or your environment.",
            "",
            session,
        )

    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=HF_TOKEN)

    obs = session["obs"]
    messages = session["messages"]
    messages.append({"role": "user", "content": observation_to_prompt(obs)})

    # Call LLM
    raw = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.2,
                max_tokens=512,
            )
            raw = (resp.choices[0].message.content or "").strip()
            break
        except Exception as exc:
            time.sleep(2.0 * (attempt + 1))
            if attempt == 2:
                return (
                    _render_obs(obs),
                    _render_history(session["step_logs"]),
                    f"**API Error:** {exc}",
                    "",
                    session,
                )

    # Parse
    action = _parse_action(raw or "")
    if action is None:
        messages.append({"role": "assistant", "content": raw or ""})
        return (
            _render_obs(obs),
            _render_history(session["step_logs"]),
            f"**AI response could not be parsed.**\n\n```\n{raw[:200]}\n```",
            "",
            session,
        )

    messages.append({"role": "assistant", "content": raw or ""})

    # Execute
    new_obs, reward, done, info = session["env"].step(action)
    session["obs"] = new_obs
    session["done"] = done
    session["cumulative_reward"] = round(session["cumulative_reward"] + reward, 4)
    session["messages"] = messages

    step_log = {
        "step": new_obs.step_count,
        "tool": action.tool,
        "parameters": action.parameters,
        "reward": reward,
        "result_summary": info.get("tool_result", {}).get("state_change", "")[:60],
    }
    session["step_logs"].append(step_log)

    status = (
        f"**AI Action:** `{action.tool}` with params `{json.dumps(action.parameters)}`\n\n"
        f"**Step Reward:** `{reward:+.4f}` | **Cumulative:** `{session['cumulative_reward']:+.4f}`\n\n"
        f"**Explanation:** {info.get('explanation', 'N/A')}"
    )

    grade_md = _compute_grade_md(session) if done else ""
    return _render_obs(new_obs), _render_history(session["step_logs"]), status, grade_md, session


# ---------------------------------------------------------------------------
# Human Step
# ---------------------------------------------------------------------------

def human_step(
    tool: str,
    order_id_in: str,
    product_id_in: str,
    warehouse_id_in: str,
    location_in: str,
    message_in: str,
    session: Dict,
) -> Tuple[str, str, str, str, Dict]:
    """Execute a manually chosen tool action."""
    if session.get("env") is None:
        return "_No task loaded._", "", "Please start a task first.", "", session
    if session.get("done"):
        return _render_obs(session["obs"]), _render_history(session["step_logs"]), \
               "**Episode complete.**", _compute_grade_md(session), session

    # Build params from inputs
    params: Dict[str, Any] = {}
    if tool == "get_tracking":
        params = {"order_id": order_id_in.strip()}
    elif tool == "check_inventory":
        params = {"product_id": product_id_in.strip()}
    elif tool == "find_warehouse":
        params = {"location": location_in.strip()}
    elif tool == "reroute_order":
        params = {"order_id": order_id_in.strip(), "warehouse_id": warehouse_id_in.strip()}
    elif tool == "issue_refund":
        params = {"order_id": order_id_in.strip()}
    elif tool == "update_crm":
        params = {"order_id": order_id_in.strip(), "message": message_in.strip()}

    try:
        action = Action(tool=ToolName(tool), parameters=params)
    except Exception as exc:
        return (
            _render_obs(session["obs"]),
            _render_history(session["step_logs"]),
            f"**Invalid action:** {exc}",
            "",
            session,
        )

    new_obs, reward, done, info = session["env"].step(action)
    session["obs"] = new_obs
    session["done"] = done
    session["cumulative_reward"] = round(session["cumulative_reward"] + reward, 4)

    step_log = {
        "step": new_obs.step_count,
        "tool": tool,
        "parameters": params,
        "reward": reward,
        "result_summary": info.get("tool_result", {}).get("state_change", "")[:60],
    }
    session["step_logs"].append(step_log)

    status = (
        f"**Your Action:** `{tool}` | **Reward:** `{reward:+.4f}` | **Cumulative:** `{session['cumulative_reward']:+.4f}`\n\n"
        f"**Result:** {info.get('tool_result', {}).get('log', 'N/A')}"
    )
    grade_md = _compute_grade_md(session) if done else ""
    return _render_obs(new_obs), _render_history(session["step_logs"]), status, grade_md, session


# ---------------------------------------------------------------------------
# Run Full AI Episode
# ---------------------------------------------------------------------------

def run_full_ai(task_id: str, session: Dict) -> Generator:
    """Generator: runs the entire AI episode yielding updates per step."""
    obs_md, hist_md, status, grade_md, session = start_task(task_id, session)
    yield obs_md, hist_md, status, grade_md, session

    for _ in range(30):  # safety cap
        if session.get("done"):
            break
        obs_md, hist_md, status, grade_md, session = ai_step(session)
        yield obs_md, hist_md, status, grade_md, session
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> Optional[Action]:
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
        tool_raw = data.get("tool", "")
        valid = {t.value for t in ToolName}
        if tool_raw not in valid:
            return None
        return Action(tool=ToolName(tool_raw), parameters=data.get("parameters", {}))
    except Exception:
        return None


def _compute_grade_md(session: Dict) -> str:
    env = session.get("env")
    if env is None:
        return ""
    try:
        g = env.grade_episode()
        bar = _grade_bar(g)
        steps = session["obs"].step_count if session.get("obs") else "?"
        reward = session.get("cumulative_reward", 0.0)
        sentiment = session["obs"].customer_sentiment if session.get("obs") else 0.0
        return (
            f"## Final Grade\n\n"
            f"{bar}\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Grade | **{g:.4f} / 1.0** |\n"
            f"| Cumulative Reward | `{reward:+.4f}` |\n"
            f"| Steps Used | `{steps}` |\n"
            f"| Customer Sentiment | `{sentiment:.2f}` |"
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Gradio UI Layout
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
body, .gradio-container {
    font-family: 'Inter', sans-serif;
}
.task-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    text-align: center;
}
.reward-positive { color: #4ade80; font-weight: bold; }
.reward-negative { color: #f87171; font-weight: bold; }
footer { display: none !important; }
"""

BANNER_MD = """
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
     border-radius: 14px; padding: 28px 32px; margin-bottom: 8px; text-align: center;">
  <h1 style="color: #e2e8f0; margin: 0; font-size: 2rem; letter-spacing: -0.5px;">
    🚚 LogisticsHub-360
  </h1>
  <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 1.05rem;">
    Intelligent E-Commerce Operations · AI Evaluation Environment
  </p>
  <p style="color: #64748b; margin: 4px 0 0 0; font-size: 0.85rem;">
    OpenEnv-Compliant · Dense Reward Signals · 3 Difficulty Levels
  </p>
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="LogisticsHub-360",
    ) as app:

        session_state = gr.State(fresh_session())

        gr.HTML(BANNER_MD)

        with gr.Tabs():

            # ── Tab 1: AI Agent Mode ────────────────────────────────────────
            with gr.Tab("🤖 AI Agent Mode"):
                gr.Markdown(
                    "_Watch an LLM agent (Qwen2.5-72B) solve logistics tasks step-by-step "
                    "using the available tools. Each action earns a dense reward signal. "
                    "Requires `HF_TOKEN` environment variable._"
                )
                with gr.Row():
                    ai_task_dd = gr.Dropdown(
                        choices=list(TASK_LABELS.values()),
                        value=list(TASK_LABELS.values())[0],
                        label="Select Task",
                        scale=3,
                    )
                    with gr.Column(scale=1):
                        ai_start_btn = gr.Button("▶ Start Task", variant="primary")
                        ai_run_btn = gr.Button("⚡ Run Full Episode")

                with gr.Row():
                    with gr.Column(scale=2):
                        ai_obs_md = gr.Markdown(label="Observation", value="_Select a task and click Start._")
                    with gr.Column(scale=1):
                        ai_step_btn = gr.Button("→ Run One AI Step", variant="secondary")
                        ai_status_md = gr.Markdown(label="Step Result")
                        ai_grade_md = gr.Markdown(label="Final Grade")

                ai_hist_md = gr.Markdown(label="Action History", value="_No actions yet._")

                # Resolve task_id from label
                def _task_id_from_label(label: str) -> str:
                    return {v: k for k, v in TASK_LABELS.items()}.get(label, TASK_ORDER[0])

                def ai_start(label, sess):
                    return start_task(_task_id_from_label(label), sess)

                def ai_one_step(sess):
                    return ai_step(sess)

                ai_start_btn.click(
                    ai_start,
                    inputs=[ai_task_dd, session_state],
                    outputs=[ai_obs_md, ai_hist_md, ai_status_md, ai_grade_md, session_state],
                )
                ai_step_btn.click(
                    ai_one_step,
                    inputs=[session_state],
                    outputs=[ai_obs_md, ai_hist_md, ai_status_md, ai_grade_md, session_state],
                )

                # Full episode streaming
                def run_full(label, sess):
                    task_id = _task_id_from_label(label)
                    o, h, s, g, sess = start_task(task_id, sess)
                    yield o, h, s, g, sess
                    for _ in range(25):
                        if sess.get("done"):
                            break
                        o, h, s, g, sess = ai_step(sess)
                        yield o, h, s, g, sess
                        time.sleep(0.3)

                ai_run_btn.click(
                    run_full,
                    inputs=[ai_task_dd, session_state],
                    outputs=[ai_obs_md, ai_hist_md, ai_status_md, ai_grade_md, session_state],
                )

            # ── Tab 2: Human Play Mode ──────────────────────────────────────
            with gr.Tab("🎮 Human Play Mode"):
                gr.Markdown(
                    "_You are the agent. Select tools manually and see how your score "
                    "compares to the AI. Can you match or beat it?_"
                )
                with gr.Row():
                    hu_task_dd = gr.Dropdown(
                        choices=list(TASK_LABELS.values()),
                        value=list(TASK_LABELS.values())[0],
                        label="Select Task",
                        scale=3,
                    )
                    hu_start_btn = gr.Button("▶ Start Task", variant="primary", scale=1)

                with gr.Row():
                    with gr.Column(scale=2):
                        hu_obs_md = gr.Markdown(value="_Select a task to begin._")
                    with gr.Column(scale=1):
                        gr.Markdown("### 🛠️ Choose Your Action")
                        hu_tool_dd = gr.Dropdown(
                            choices=TOOL_LABELS,
                            value=TOOL_LABELS[0],
                            label="Tool",
                        )
                        hu_order_id = gr.Textbox(label="order_id", placeholder="e.g. ORD-88421")
                        hu_product_id = gr.Textbox(label="product_id", placeholder="e.g. PROD-WH-7723")
                        hu_warehouse_id = gr.Textbox(label="warehouse_id", placeholder="e.g. WH-WEST-08")
                        hu_location = gr.Textbox(label="location", placeholder="e.g. Los Angeles, CA")
                        hu_message = gr.Textbox(
                            label="message (for update_crm)",
                            placeholder="Your customer note...",
                            lines=3,
                        )
                        hu_action_btn = gr.Button("Execute Action", variant="primary")
                        hu_status_md = gr.Markdown()
                        hu_grade_md = gr.Markdown()

                hu_hist_md = gr.Markdown(value="_No actions yet._")

                hu_session = gr.State(fresh_session())

                def hu_start(label, sess):
                    return start_task(_task_id_from_label(label), sess)

                hu_start_btn.click(
                    hu_start,
                    inputs=[hu_task_dd, hu_session],
                    outputs=[hu_obs_md, hu_hist_md, hu_status_md, hu_grade_md, hu_session],
                )
                hu_action_btn.click(
                    human_step,
                    inputs=[
                        hu_tool_dd, hu_order_id, hu_product_id,
                        hu_warehouse_id, hu_location, hu_message, hu_session,
                    ],
                    outputs=[hu_obs_md, hu_hist_md, hu_status_md, hu_grade_md, hu_session],
                )

            # ── Tab 3: Environment Info ─────────────────────────────────────
            with gr.Tab("📖 Environment Info"):
                gr.Markdown("""
## LogisticsHub-360 — Environment Reference

### Available Tools

| Tool | Required Parameters | Description |
|---|---|---|
| `get_tracking` | `order_id` | Retrieve real-time shipment tracking |
| `check_inventory` | `product_id` | Query stock levels and warehouses |
| `find_warehouse` | `location` | Locate optimal alternate warehouse |
| `reroute_order` | `order_id`, `warehouse_id` | Reroute shipment to warehouse |
| `issue_refund` | `order_id` | Issue full refund when no stock exists |
| `update_crm` | `order_id`, `message` | Log customer communication |

### Reward System

| Event | Signal |
|---|---|
| Valid tool executed | **+0.30** |
| Correct sequence step | **+0.20** |
| Out-of-order correct tool | **+0.10** |
| Task completion | **+1.00** |
| Efficiency bonus (steps saved) | up to **+0.30** |
| Invalid action | **−0.10** |
| Repeated failure | **−0.30** |
| Wrong strategic decision | **−0.50** |
| Destructive action (OOS reroute) | **−1.00** |

### Grading Weights
- **Correctness:** 40%
- **Efficiency:** 30%
- **Decision Quality:** 30%

### Task Order IDs
- `order_tracking` — Easy (8 steps max)
- `shipment_rerouting` — Medium (15 steps max)
- `stockout_crisis` — Hard (20 steps max)

### Benchmarks (Qwen2.5-72B-Instruct)

| Task | Observed Grade |
|---|---|
| Order Tracking | 0.925 |
| Shipment Rerouting | ~0.85 |
| Stockout Crisis | ~0.70 |
""")

    return app


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fastapi import FastAPI, Request
    import uvicorn

    fastapi_app = FastAPI()
    _api_env = None

    @fastapi_app.post("/reset")
    async def api_reset(request: Request):
        global _api_env
        try:
            data = await request.json()
        except:
            data = {}
        task_id = data.get("task_id", "order_tracking")
        _api_env = LogisticsHub360Env(task_id=task_id)
        obs = _api_env.reset()
        return obs.model_dump()

    @fastapi_app.post("/step")
    async def api_step(action: Action):
        global _api_env
        if not _api_env:
            _api_env = LogisticsHub360Env(task_id="order_tracking")
            _api_env.reset()
        obs, reward, done, info = _api_env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info
        }

    @fastapi_app.get("/state")
    async def api_state():
        if _api_env:
            return _api_env.state()
        return {}

    @fastapi_app.post("/state")
    async def api_state_post():
        if _api_env:
            return _api_env.state()
        return {}

    # Mount Gradio UI
    app = build_app()
    gr.mount_gradio_app(fastapi_app, app, path="/")

    # Run server (IMPORTANT FIXED PORT)
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)