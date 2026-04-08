"""
LogisticsHub-360: Core Environment
Implements the OpenEnv-compliant interface:
    reset() → Observation
    step(action) → (Observation, float, bool, dict)
    state() → dict
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from env.models import (
    Action,
    ActionHistoryEntry,
    InternalState,
    Observation,
    OrderInfo,
    InventoryInfo,
    TaskDifficulty,
    ToolName,
)
from env.tasks import TASK_BUILDERS, TASK_ORDER
from env.tools import TOOL_REGISTRY, ToolResult
from env.graders import compute_step_reward, compute_completion_reward, grade
from env.utils import (
    configure_logging,
    detect_loop,
    is_action_repeated,
    EpisodeMetrics,
    state_to_dict,
)

configure_logging()
logger = logging.getLogger("logisticshub360.environment")


# ---------------------------------------------------------------------------
# Available Tools List (shown in observations)
# ---------------------------------------------------------------------------

_ALL_TOOLS = [t.value for t in ToolName]


# ---------------------------------------------------------------------------
# LogisticsHub360Env — Main Environment Class
# ---------------------------------------------------------------------------


class LogisticsHub360Env:
    """
    OpenEnv-compliant environment for evaluating AI agents on
    e-commerce logistics tasks.

    Supports three canonical tasks:
        1. order_tracking    (Easy)
        2. shipment_rerouting (Medium)
        3. stockout_crisis   (Hard)

    Usage:
        env = LogisticsHub360Env(task_id="order_tracking")
        obs = env.reset()
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        score = env.grade()
    """

    def __init__(
        self,
        task_id: str = "order_tracking",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialise the environment with a specific task.

        Args:
            task_id: One of 'order_tracking', 'shipment_rerouting', 'stockout_crisis'.
            config: Optional config overrides (e.g., max_steps, difficulty).

        Raises:
            ValueError: If task_id is unrecognised.
        """
        if task_id not in TASK_BUILDERS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid options: {list(TASK_BUILDERS.keys())}"
            )
        self._task_id = task_id
        self._config = config or {}
        self._internal_state: Optional[InternalState] = None
        self._metrics: Optional[EpisodeMetrics] = None
        self._episode_done: bool = False

        logger.info(f"[ENV] LogisticsHub360Env initialised for task='{task_id}'.")

    # -----------------------------------------------------------------------
    # OpenEnv Interface
    # -----------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment and return the initial observation.

        Builds a fresh internal state from the task definition,
        resets all flags, and returns a fully populated Observation.

        Returns:
            Observation — the initial state visible to the agent.
        """
        task_cfg = TASK_BUILDERS[self._task_id]
        builder_fn = task_cfg["builder"]

        self._internal_state = builder_fn()
        self._metrics = EpisodeMetrics(self._task_id)
        self._episode_done = False

        # Apply config overrides
        if "max_steps" in self._config:
            self._internal_state.max_steps = int(self._config["max_steps"])

        obs = self._build_observation()
        logger.info(
            f"[ENV] reset() → task='{self._task_id}', "
            f"max_steps={self._internal_state.max_steps}."
        )
        return obs

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Execute one agent action and advance the environment.

        Pipeline:
          1. Validate environment state (reset must have been called).
          2. Detect loops and repeated actions.
          3. Dispatch tool call.
          4. Compute per-step reward.
          5. Check terminal conditions.
          6. Return (observation, reward, done, info).

        Args:
            action: An Action pydantic model specifying tool + parameters.

        Returns:
            Tuple of (Observation, float reward, bool done, dict info).

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._internal_state is None or self._metrics is None:
            raise RuntimeError("Call reset() before step().")

        if self._episode_done:
            logger.warning("[ENV] step() called after episode is done. Returning terminal obs.")
            obs = self._build_observation()
            return obs, 0.0, True, {"warning": "episode_already_done"}

        state = self._internal_state
        state.step_count += 1

        # --- Loop Detection ---
        if detect_loop(state.action_history):
            state.loop_detected = True
            state.failed = True
            self._episode_done = True
            loop_penalty = -1.0
            final_score = grade(self._task_id, state)
            self._metrics.loop_detected = True
            self._metrics.finalize(
                final_score, state.customer_sentiment, loop=True
            )
            obs = self._build_observation(is_done=True)
            info = {
                "reason": "loop_detected",
                "final_grade": final_score,
                "metrics": self._metrics.to_dict(),
            }
            logger.error(
                f"[ENV] Loop detected at step {state.step_count}. "
                f"Episode terminated. Grade={final_score:.4f}."
            )
            return obs, loop_penalty, True, info

        # --- Step Limit Check ---
        if state.step_count > state.max_steps:
            state.failed = True
            self._episode_done = True
            final_score = grade(self._task_id, state)
            self._metrics.finalize(final_score, state.customer_sentiment)
            obs = self._build_observation(is_done=True)
            info = {
                "reason": "max_steps_exceeded",
                "final_grade": final_score,
                "metrics": self._metrics.to_dict(),
            }
            logger.warning(
                f"[ENV] Max steps ({state.max_steps}) exceeded. "
                f"Grade={final_score:.4f}."
            )
            return obs, -0.10, True, info

        # --- Repeated Action Detection ---
        action_was_repeated = is_action_repeated(
            action.tool, action.parameters, state.action_history
        )

        # --- Tool Dispatch ---
        tool_fn = TOOL_REGISTRY.get(action.tool)
        if tool_fn is None:
            # Unknown tool — penalise
            penalty = -0.10
            state.system_logs.append(
                f"[ERROR] Unknown tool '{action.tool}' at step {state.step_count}."
            )
            obs = self._build_observation(last_reward=penalty)
            return obs, penalty, False, {"error": f"unknown_tool: {action.tool}"}

        # Route tool parameters
        tool_params = dict(action.parameters)
        tool_result: ToolResult = _dispatch_tool(action.tool, tool_params, state)

        # --- Reward Computation ---
        reward_signal = compute_step_reward(
            tool_name=action.tool,
            tool_result=tool_result,
            state=state,
            action_was_repeated=action_was_repeated,
        )
        step_reward = reward_signal.scalar

        # Record action history
        history_entry = ActionHistoryEntry(
            step=state.step_count,
            tool=action.tool,
            parameters=action.parameters,
            result_summary=(
                tool_result.state_change_summary[:120]
                if tool_result.success
                else f"FAILED: {tool_result.data.get('error', 'unknown')}"
            ),
            reward=step_reward,
        )
        state.action_history.append(history_entry)
        state.cumulative_reward += step_reward

        # Track metrics
        self._metrics.record_reward(step_reward)
        self._metrics.record_tool(action.tool, tool_result.success)

        # --- Terminal Condition Detection ---
        done, terminal_reward, reason = self._check_terminal(state)

        if done:
            self._episode_done = True
            step_reward += terminal_reward
            state.cumulative_reward += terminal_reward
            final_score = grade(self._task_id, state)
            self._metrics.record_reward(terminal_reward)
            self._metrics.finalize(final_score, state.customer_sentiment)

            obs = self._build_observation(is_done=True, last_reward=step_reward)
            info = {
                "reason": reason,
                "final_grade": final_score,
                "reward_breakdown": reward_signal.breakdown.model_dump(),
                "explanation": reward_signal.explanation,
                "metrics": self._metrics.to_dict(),
                "tool_result": tool_result.to_dict(),
            }
            logger.info(
                f"[ENV] Episode complete. reason={reason}, "
                f"grade={final_score:.4f}, reward={step_reward:.4f}."
            )
            return obs, step_reward, True, info

        # --- Continuing ---
        obs = self._build_observation(last_reward=step_reward)
        info = {
            "step": state.step_count,
            "reward_breakdown": reward_signal.breakdown.model_dump(),
            "explanation": reward_signal.explanation,
            "tool_result": tool_result.to_dict(),
        }
        logger.debug(
            f"[ENV] step={state.step_count}, tool={action.tool}, "
            f"reward={step_reward:.4f}."
        )
        return obs, step_reward, False, info

    def state(self) -> Dict[str, Any]:
        """
        Return the full internal state as a dict (debug only — not exposed to agent).

        Returns:
            JSON-serializable dict of the complete InternalState.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._internal_state is None:
            raise RuntimeError("Call reset() before accessing state().")
        return state_to_dict(self._internal_state)

    def grade_episode(self) -> float:
        """
        Compute and return the final deterministic grade for the current episode.

        Returns:
            Grade in [0.0, 1.0].

        Raises:
            RuntimeError: If episode has not completed.
        """
        if self._internal_state is None:
            raise RuntimeError("Call reset() before grade_episode().")
        return grade(self._task_id, self._internal_state)

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _check_terminal(
        self, state: InternalState
    ) -> Tuple[bool, float, str]:
        """
        Determine if the episode has reached a terminal state.

        Returns:
            (is_done, terminal_reward, reason_string)
        """
        task_id = self._task_id

        # Task 1: tracking + CRM = resolved
        if task_id == "order_tracking":
            if state.tracking_checked and state.crm_updated:
                state.resolved = True
                completion_signal = compute_completion_reward(state)
                return True, completion_signal.scalar, "task_completed"

        # Task 2: full sequence complete
        elif task_id == "shipment_rerouting":
            if (
                state.tracking_checked
                and state.inventory_checked
                and state.warehouse_found
                and state.rerouted
                and state.crm_updated
            ):
                state.resolved = True
                completion_signal = compute_completion_reward(state)
                return True, completion_signal.scalar, "task_completed"

        # Task 3: (refund OR reroute) AND crm_updated AND inventory_checked
        elif task_id == "stockout_crisis":
            resolution_made = state.refund_issued or state.rerouted
            if (
                state.tracking_checked
                and state.inventory_checked
                and resolution_made
                and state.crm_updated
            ):
                state.resolved = True
                completion_signal = compute_completion_reward(state)
                return True, completion_signal.scalar, "task_completed"

        return False, 0.0, "continuing"

    def _build_observation(
        self,
        is_done: bool = False,
        last_reward: Optional[float] = None,
    ) -> Observation:
        """Construct the agent-visible Observation from internal state."""
        state = self._internal_state
        task_cfg = TASK_BUILDERS[self._task_id]

        return Observation(
            task_id=state.task_id,
            task_description=task_cfg["description"],
            difficulty=task_cfg["difficulty"],
            system_logs=list(state.system_logs[-10:]),  # Last 10 log lines
            order_status=state.order,
            inventory_state=state.inventory,
            customer_sentiment=state.customer_sentiment,
            available_tools=_ALL_TOOLS,
            action_history=list(state.action_history),
            hints=task_cfg["hints"],
            constraints=task_cfg["constraints"],
            step_count=state.step_count,
            max_steps=state.max_steps,
            is_done=is_done,
            last_reward=last_reward,
        )


# ---------------------------------------------------------------------------
# Tool Parameter Router
# ---------------------------------------------------------------------------


def _dispatch_tool(
    tool_name: str,
    params: Dict[str, Any],
    state: InternalState,
) -> ToolResult:
    """
    Route tool name and parameters to the correct tool function.

    Args:
        tool_name: One of the registered tool names.
        params: Dict of tool-specific parameters.
        state: Current internal state (mutated by tool).

    Returns:
        ToolResult from the tool execution.
    """
    fn = TOOL_REGISTRY[tool_name]

    try:
        if tool_name == "get_tracking":
            return fn(order_id=params.get("order_id", ""), state=state)
        elif tool_name == "check_inventory":
            return fn(product_id=params.get("product_id", ""), state=state)
        elif tool_name == "find_warehouse":
            return fn(location=params.get("location", ""), state=state)
        elif tool_name == "reroute_order":
            return fn(
                order_id=params.get("order_id", ""),
                warehouse_id=params.get("warehouse_id", ""),
                state=state,
            )
        elif tool_name == "issue_refund":
            return fn(order_id=params.get("order_id", ""), state=state)
        elif tool_name == "update_crm":
            return fn(
                order_id=params.get("order_id", ""),
                message=params.get("message", ""),
                state=state,
            )
        else:
            from env.tools import ToolResult as TR
            return TR(
                success=False,
                data={"error": f"unrouted_tool: {tool_name}"},
                log_message=f"Unrouted tool: {tool_name}",
                state_change_summary="No state change.",
            )
    except Exception as exc:
        logger.error(f"[ENV] Tool '{tool_name}' raised exception: {exc}", exc_info=True)
        from env.tools import ToolResult as TR
        return TR(
            success=False,
            data={"error": f"tool_exception: {str(exc)}"},
            log_message=f"Exception in tool '{tool_name}': {exc}",
            state_change_summary="No state change — exception.",
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_env(task_id: str = "order_tracking", **config_kwargs: Any) -> LogisticsHub360Env:
    """
    Factory function to create and return an environment instance.

    Args:
        task_id: Task to evaluate.
        **config_kwargs: Config overrides passed to the environment.

    Returns:
        Configured LogisticsHub360Env instance.
    """
    return LogisticsHub360Env(task_id=task_id, config=config_kwargs)
