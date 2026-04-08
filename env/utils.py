"""
LogisticsHub-360: Utility Functions
Shared helpers for logging, loop detection, metrics tracking, and serialization.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from env.models import InternalState, ActionHistoryEntry

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger for structured output."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger prefixed with logisticshub360."""
    return logging.getLogger(f"logisticshub360.{name}")


# ---------------------------------------------------------------------------
# Loop Detection
# ---------------------------------------------------------------------------

LOOP_THRESHOLD = 3  # Same action repeated ≥ 3 times = loop


def detect_loop(
    action_history: List["ActionHistoryEntry"],
    window: int = 6,
) -> bool:
    """
    Detect whether the agent is stuck in a repetitive loop.

    Checks if any single tool call appears ≥ LOOP_THRESHOLD times
    within the last `window` steps.

    Args:
        action_history: Full list of past ActionHistoryEntry objects.
        window: Number of recent steps to examine.

    Returns:
        True if a loop is detected, False otherwise.
    """
    if len(action_history) < LOOP_THRESHOLD:
        return False

    recent = action_history[-window:]
    tool_counts = Counter(entry.tool for entry in recent)
    for tool, count in tool_counts.items():
        if count >= LOOP_THRESHOLD:
            return True
    return False


def is_action_repeated(
    tool_name: str,
    params: Dict[str, Any],
    action_history: List["ActionHistoryEntry"],
) -> bool:
    """
    Check if identical tool+params combination was already executed.

    Args:
        tool_name: Tool being invoked.
        params: Tool parameters dict.
        action_history: Full history of past actions.

    Returns:
        True if identical action already occurred.
    """
    params_str = json.dumps(params, sort_keys=True)
    for entry in action_history:
        if entry.tool == tool_name:
            try:
                existing_params = json.dumps(
                    entry.parameters, sort_keys=True
                )
                if existing_params == params_str:
                    return True
            except (TypeError, ValueError):
                pass
    return False


# ---------------------------------------------------------------------------
# Metrics Tracking
# ---------------------------------------------------------------------------


class EpisodeMetrics:
    """Tracks cumulative metrics across an entire episode."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.rewards: List[float] = []
        self.tool_calls: List[str] = []
        self.successful_tools: int = 0
        self.failed_tools: int = 0
        self.loop_detected: bool = False
        self.final_grade: Optional[float] = None
        self.final_sentiment: Optional[float] = None

    def record_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def record_tool(self, tool_name: str, success: bool) -> None:
        self.tool_calls.append(tool_name)
        if success:
            self.successful_tools += 1
        else:
            self.failed_tools += 1

    def finalize(
        self,
        grade: float,
        sentiment: float,
        loop: bool = False,
    ) -> None:
        self.end_time = time.time()
        self.final_grade = grade
        self.final_sentiment = sentiment
        self.loop_detected = loop

    @property
    def total_reward(self) -> float:
        return round(sum(self.rewards), 4)

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 2)

    @property
    def tool_success_rate(self) -> float:
        total = self.successful_tools + self.failed_tools
        if total == 0:
            return 0.0
        return round(self.successful_tools / total, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_reward": self.total_reward,
            "step_count": len(self.tool_calls),
            "tool_calls": self.tool_calls,
            "successful_tools": self.successful_tools,
            "failed_tools": self.failed_tools,
            "tool_success_rate": self.tool_success_rate,
            "loop_detected": self.loop_detected,
            "final_grade": self.final_grade,
            "final_sentiment": self.final_sentiment,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def summary(self) -> str:
        return (
            f"Task={self.task_id} | Steps={len(self.tool_calls)} | "
            f"TotalReward={self.total_reward:.3f} | Grade={self.final_grade:.4f} | "
            f"Sentiment={self.final_sentiment:.2f} | "
            f"SuccessRate={self.tool_success_rate:.2f} | "
            f"Elapsed={self.elapsed_seconds}s"
        )


# ---------------------------------------------------------------------------
# Serialization Helpers
# ---------------------------------------------------------------------------


def state_to_dict(state: "InternalState") -> Dict[str, Any]:
    """
    Serialize an InternalState to a JSON-compatible dict.

    Drops non-serializable types and converts enums to their string values.
    """
    return state.model_dump(mode="json")


def observation_to_prompt(obs: Any) -> str:
    """
    Convert an Observation to a structured prompt string suitable for LLM input.

    Args:
        obs: An Observation pydantic model instance.

    Returns:
        Formatted string prompt.
    """
    parts: List[str] = [
        "=== LogisticsHub-360 Observation ===",
        f"Task: {obs.task_id} (Difficulty: {obs.difficulty})",
        f"Step: {obs.step_count}/{obs.max_steps}",
        "",
        "--- Task Description ---",
        obs.task_description,
        "",
        "--- System Logs (Recent) ---",
    ]
    for log in obs.system_logs[-5:]:
        parts.append(f"  {log}")

    if obs.order_status:
        parts += [
            "",
            "--- Order Status ---",
            f"  Order ID: {obs.order_status.order_id}",
            f"  Status: {obs.order_status.status}",
            f"  Product: {obs.order_status.product_name}",
            f"  Carrier: {obs.order_status.carrier}",
            f"  Destination: {obs.order_status.destination}",
            f"  Est. Delivery: {obs.order_status.estimated_delivery}",
            f"  Last Location: {obs.order_status.last_known_location or 'N/A'}",
            f"  Delay Reason: {obs.order_status.delay_reason or 'None'}",
        ]

    if obs.inventory_state:
        parts += [
            "",
            "--- Inventory State ---",
            f"  Product: {obs.inventory_state.product_name}",
            f"  Level: {obs.inventory_state.level}",
            f"  Quantity: {obs.inventory_state.quantity}",
        ]

    parts += [
        "",
        f"Customer Sentiment: {obs.customer_sentiment:.2f} (0.0=hostile, 1.0=satisfied)",
        f"Last Reward: {obs.last_reward}",
        "",
        "--- Hints ---",
    ]
    for h in obs.hints:
        parts.append(f"  • {h}")

    parts += [
        "",
        "--- Constraints ---",
    ]
    for c in obs.constraints:
        parts.append(f"  ⚠️  {c}")

    parts += [
        "",
        "--- Available Tools ---",
    ]
    for t in obs.available_tools:
        parts.append(f"  - {t}")

    parts += [
        "",
        "--- Action History ---",
    ]
    if obs.action_history:
        for entry in obs.action_history[-5:]:
            parts.append(
                f"  Step {entry.step}: {entry.tool}({entry.parameters}) "
                f"→ reward={entry.reward:.3f} | {entry.result_summary}"
            )
    else:
        parts.append("  (No actions taken yet)")

    parts.append("")
    parts.append(
        "Respond with a JSON action object: "
        '{"tool": "<tool_name>", "parameters": {<key>: <value>, ...}}'
    )

    return "\n".join(parts)
