#!/usr/bin/env python3
"""
LogisticsHub-360: Baseline Inference Script

Runs an AI agent (via OpenAI-compatible API) through all three evaluation tasks.
Logs step-by-step actions, rewards, and final grades.

Environment Variables:
    HF_TOKEN               Hugging Face inference API token (required)
    LH360_BASE_URL         API base URL (default: https://api-inference.huggingface.co/v1)
    LH360_MODEL            Model ID (default: mistralai/Mistral-7B-Instruct-v0.3)
    LH360_MAX_RETRIES      Max retries per API call (default: 3)
    LH360_TEMPERATURE      Sampling temperature (default: 0.2)

Usage:
    HF_TOKEN=<token> python scripts/run_inference.py
    HF_TOKEN=<token> python scripts/run_inference.py --task order_tracking
    HF_TOKEN=<token> python scripts/run_inference.py --task all --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from env.environment import LogisticsHub360Env
from env.models import Action, ToolName
from env.tasks import TASK_ORDER
from env.utils import configure_logging, observation_to_prompt

configure_logging()
logger = logging.getLogger("logisticshub360.inference")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_URL = os.environ.get(
    "LH360_BASE_URL",
    "https://router.huggingface.co/v1",
)
MODEL_ID = os.environ.get(
    "LH360_MODEL",
    "Qwen/Qwen2.5-72B-Instruct",
)
MAX_RETRIES = int(os.environ.get("LH360_MAX_RETRIES", "3"))
TEMPERATURE = float(os.environ.get("LH360_TEMPERATURE", "0.2"))
MAX_TOKENS = 512


# ---------------------------------------------------------------------------
# Agent System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI logistics operations agent working inside LogisticsHub-360.

Your sole job is to resolve logistics issues by calling the available tools in the correct order.

RESPONSE FORMAT (STRICT):
You must ALWAYS respond with a single valid JSON object. No prose, no explanation outside the JSON.

{
  "tool": "<tool_name>",
  "parameters": {
    "<param_key>": "<param_value>"
  }
}

Available tools:
- get_tracking: {"order_id": "<id>"}
- check_inventory: {"product_id": "<id>"}
- find_warehouse: {"location": "<destination_city>"}
- reroute_order: {"order_id": "<id>", "warehouse_id": "<wh_id>"}
- issue_refund: {"order_id": "<id>"}
- update_crm: {"order_id": "<id>", "message": "<detailed_message>"}

Rules:
1. Always call get_tracking FIRST to understand order status.
2. Call check_inventory before making rerouting or refund decisions.
3. Use find_warehouse before reroute_order.
4. If product is completely out of stock, issue a refund — do NOT reroute.
5. Always end with update_crm to notify the customer.
6. Extract exact IDs (order_id, product_id) from the task description.
7. Be efficient — use the minimum number of steps to resolve each task.
"""


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

def create_client() -> OpenAI:
    """Create and return an OpenAI-compatible client pointing at HF inference."""
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Export it before running this script."
        )
    return OpenAI(base_url=BASE_URL, api_key=HF_TOKEN)


def call_llm(
    client: OpenAI,
    messages: List[Dict[str, str]],
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    """
    Call the LLM API with retry logic.

    Args:
        client: OpenAI-compatible client.
        messages: Chat history for this step.
        retries: Number of retry attempts.

    Returns:
        Raw response string or None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,  # type: ignore[arg-type]
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as exc:
            err_str = str(exc)
            # Rate limited — wait longer
            wait = 5.0 * attempt if "402" in err_str or "429" in err_str else 2.0 * attempt
            logger.warning(
                f"[LLM] API call failed (attempt {attempt}/{retries}): {exc}. "
                f"Retrying in {wait:.0f}s..."
            )
            if attempt < retries:
                time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Action Parser
# ---------------------------------------------------------------------------

def parse_action(raw_response: str) -> Optional[Action]:
    """
    Parse an LLM response string into a validated Action model.

    Handles common LLM formatting quirks (markdown code fences, trailing text).

    Args:
        raw_response: Raw string from the LLM.

    Returns:
        Action instance or None if parsing fails.
    """
    # Strip markdown fences
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()

    # Extract first JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning(f"[PARSE] No JSON object found in: {text[:100]}")
        return None

    json_str = text[start:end]
    try:
        data = json.loads(json_str)
        tool_raw = data.get("tool", "")
        params = data.get("parameters", {})

        # Validate tool name
        valid_tools = {t.value for t in ToolName}
        if tool_raw not in valid_tools:
            logger.warning(f"[PARSE] Unknown tool '{tool_raw}'. Skipping.")
            return None

        return Action(tool=ToolName(tool_raw), parameters=params)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning(f"[PARSE] Failed to parse action: {exc}. Raw: {json_str[:200]}")
        return None


# ---------------------------------------------------------------------------
# Single-Task Runner
# ---------------------------------------------------------------------------


def run_task(
    client: OpenAI,
    task_id: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single evaluation task with the LLM agent.

    Args:
        client: OpenAI-compatible client.
        task_id: Task identifier string.
        verbose: Whether to print step-by-step output.

    Returns:
        Dict of task results including grade and metrics.
    """
    env = LogisticsHub360Env(task_id=task_id)
    obs = env.reset()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    step_logs: List[Dict[str, Any]] = []
    cumulative_reward = 0.0
    done = False

    if verbose:
        print(f"\n{'='*70}")
        print(f"  TASK: {task_id.upper()} | Difficulty: {obs.difficulty}")
        print(f"{'='*70}")
        print(f"  Description: {obs.task_description[:200]}...")
        print()

    while not done:
        # Build prompt for this step
        user_prompt = observation_to_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        if verbose:
            print(f"  [Step {obs.step_count + 1}/{obs.max_steps}]", end=" ")

        # Get LLM action
        raw_response = call_llm(client, messages)
        if raw_response is None:
            logger.error("[RUN] LLM returned no response. Aborting task.")
            break

        # Parse action
        action = parse_action(raw_response)
        if action is None:
            logger.warning("[RUN] Could not parse valid action. Skipping step.")
            # Add failed response to history
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({
                "role": "user",
                "content": "ERROR: Your response was not valid JSON. Respond with a JSON action only.",
            })
            continue

        if verbose:
            print(f"Tool: {action.tool} | Params: {action.parameters}")

        # Execute action
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward

        # Record for history
        messages.append({"role": "assistant", "content": raw_response})

        step_log = {
            "step": obs.step_count,
            "tool": action.tool,
            "parameters": action.parameters,
            "reward": reward,
            "done": done,
            "explanation": info.get("explanation", ""),
        }
        step_logs.append(step_log)

        if verbose:
            explanation = info.get("explanation", "")
            print(
                f"         → Reward: {reward:+.3f} | Cumulative: {cumulative_reward:+.3f}"
            )
            if explanation:
                print(f"         → {explanation[:100]}")

        if done:
            break

    # Final Grade
    final_grade = grade_episode(env, task_id)

    if verbose:
        print()
        print(f"  [DONE] Task '{task_id}' complete.")
        print(f"     Cumulative Reward : {cumulative_reward:.4f}")
        print(f"     Final Grade       : {final_grade:.4f} / 1.0")
        print(f"     Steps Used        : {obs.step_count} / {obs.max_steps}")
        print(f"     Customer Sentiment: {obs.customer_sentiment:.2f}")

    return {
        "task_id": task_id,
        "difficulty": obs.difficulty,
        "final_grade": final_grade,
        "cumulative_reward": round(cumulative_reward, 4),
        "steps_used": obs.step_count,
        "max_steps": obs.max_steps,
        "customer_sentiment_final": obs.customer_sentiment,
        "step_logs": step_logs,
    }


def grade_episode(env: LogisticsHub360Env, task_id: str) -> float:
    """Safely grade the completed episode."""
    try:
        return env.grade_episode()
    except Exception as exc:
        logger.warning(f"[GRADE] Could not grade episode for {task_id}: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Multi-Task Runner
# ---------------------------------------------------------------------------


def run_all_tasks(
    client: OpenAI,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run all three tasks sequentially and return aggregated results."""
    all_results: List[Dict[str, Any]] = []

    for task_id in TASK_ORDER:
        result = run_task(client, task_id, verbose=verbose)
        all_results.append(result)
        if task_id != TASK_ORDER[-1]:
            inter_task_sleep = 8.0  # Respect HF free-tier rate limits between tasks
            logger.info(f"[RUN] Sleeping {inter_task_sleep}s between tasks to respect rate limits...")
            time.sleep(inter_task_sleep)

    return all_results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table of all task results."""
    print("\n" + "=" * 70)
    print("  LOGISTICSHUB-360 — EVALUATION SUMMARY")
    print("=" * 70)
    print(
        f"  {'Task':<28} {'Grade':>7} {'Reward':>9} {'Steps':>7} {'Sentiment':>10}"
    )
    print("  " + "-" * 60)

    total_grade = 0.0
    for r in results:
        total_grade += r["final_grade"]
        print(
            f"  {r['task_id']:<28} "
            f"{r['final_grade']:>7.4f} "
            f"{r['cumulative_reward']:>9.4f} "
            f"{r['steps_used']:>4}/{r['max_steps']:<2} "
            f"{r['customer_sentiment_final']:>9.2f}"
        )

    avg_grade = total_grade / max(len(results), 1)
    print("  " + "-" * 60)
    print(f"  {'AVERAGE GRADE':<28} {avg_grade:>7.4f}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LogisticsHub-360 — Baseline Inference Script"
    )
    parser.add_argument(
        "--task",
        choices=TASK_ORDER + ["all"],
        default="all",
        help="Which task to run (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose step-by-step output.",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    try:
        client = create_client()
    except EnvironmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.task == "all":
        results = run_all_tasks(client, verbose=verbose)
    else:
        results = [run_task(client, args.task, verbose=verbose)]

    if verbose:
        print_summary(results)

    # Save results
    if args.output:
        out_path = args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {out_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
