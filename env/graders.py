"""
LogisticsHub-360: Dense Reward Engine
Computes per-step rewards and final task grades based on correctness,
efficiency, decision quality, and sequence adherence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from env.models import RewardBreakdown, RewardSignal

if TYPE_CHECKING:
    from env.models import InternalState
    from env.tools import ToolResult

logger = logging.getLogger("logisticshub360.graders")


# ---------------------------------------------------------------------------
# Reward Constants
# ---------------------------------------------------------------------------

REWARD_CORRECT_TOOL = 0.30
REWARD_CORRECT_SEQUENCE = 0.20
REWARD_PARTIAL_PROGRESS = 0.10
REWARD_COMPLETION = 1.00

PENALTY_INVALID_ACTION = -0.10
PENALTY_REPEATED_MISTAKE = -0.30
PENALTY_WRONG_DECISION = -0.50
PENALTY_DESTRUCTIVE = -1.00

# ---------------------------------------------------------------------------
# Per-Step Reward Computation
# ---------------------------------------------------------------------------


def compute_step_reward(
    tool_name: str,
    tool_result: "ToolResult",
    state: "InternalState",
    action_was_repeated: bool,
) -> RewardSignal:
    """
    Compute the dense reward signal after each tool call.

    Rewards tool correctness, sequence adherence, and partial progress.
    Penalizes invalid actions, repeated mistakes, and wrong decisions.

    Args:
        tool_name: The name of the tool that was invoked.
        tool_result: Result object from the tool execution.
        state: Post-execution internal state.
        action_was_repeated: Whether this exact action was already attempted.

    Returns:
        RewardSignal with scalar value and annotated breakdown.
    """
    breakdown = RewardBreakdown()
    explanation_parts: list[str] = []

    # --- Destructive action check (Task 3 specific: rerouting with full stockout) ---
    # Applies regardless of tool success: successfully rerouting to an OOS warehouse
    # is the WORST possible outcome — it should carry the maximum penalty.
    if (
        tool_name == "reroute_order"
        and state.inventory.level == "out_of_stock"
    ):
        breakdown.penalty += PENALTY_DESTRUCTIVE
        explanation_parts.append(
            f"CRITICAL: Reroute attempted/succeeded during complete stockout. "
            f"Penalty {PENALTY_DESTRUCTIVE}."
        )
        return RewardSignal(
            scalar=breakdown.total,
            breakdown=breakdown,
            explanation=" | ".join(explanation_parts),
            step=state.step_count,
        )

    # --- Invalid action ---
    if not tool_result.success:
        if action_was_repeated:
            breakdown.penalty += PENALTY_REPEATED_MISTAKE
            explanation_parts.append(
                f"Repeated failed action '{tool_name}'. "
                f"Penalty {PENALTY_REPEATED_MISTAKE}."
            )
        else:
            breakdown.penalty += PENALTY_INVALID_ACTION
            explanation_parts.append(
                f"Invalid action '{tool_name}': {tool_result.data.get('error', 'unknown')}. "
                f"Penalty {PENALTY_INVALID_ACTION}."
            )
        return RewardSignal(
            scalar=breakdown.total,
            breakdown=breakdown,
            explanation=" | ".join(explanation_parts),
            step=state.step_count,
        )

    # --- Correct tool usage ---
    breakdown.tool_use += REWARD_CORRECT_TOOL
    explanation_parts.append(
        f"Valid tool '{tool_name}' executed successfully (+{REWARD_CORRECT_TOOL})."
    )

    # --- Sequence adherence bonus ---
    expected = state.expected_sequence
    completed = state.completed_sequence_steps
    next_expected_idx = len(completed)

    if next_expected_idx < len(expected) and expected[next_expected_idx] == tool_name:
        breakdown.sequence_bonus += REWARD_CORRECT_SEQUENCE
        explanation_parts.append(
            f"Correct sequence step #{next_expected_idx + 1} '{tool_name}' "
            f"(+{REWARD_CORRECT_SEQUENCE})."
        )
        state.completed_sequence_steps.append(tool_name)
    elif tool_name in expected and tool_name not in completed:
        # Correct tool but out of order — partial credit
        breakdown.progress += REWARD_PARTIAL_PROGRESS
        explanation_parts.append(
            f"Tool '{tool_name}' is correct but out of sequence "
            f"(+{REWARD_PARTIAL_PROGRESS} partial)."
        )

    # --- Wrong decision: rerouting when refund was the right call ---
    if (
        tool_name == "reroute_order"
        and state.task_id == "stockout_crisis"
        and state.inventory_checked
        and state.inventory.level == "out_of_stock"
    ):
        breakdown.penalty += PENALTY_WRONG_DECISION
        explanation_parts.append(
            f"Wrong decision: rerouted during confirmed stockout. "
            f"Penalty {PENALTY_WRONG_DECISION}."
        )

    # --- Sentiment improvement bonus ---
    if state.customer_sentiment >= 0.75:
        breakdown.progress += 0.05
        explanation_parts.append("Sentiment recovered above 0.75 (+0.05).")

    return RewardSignal(
        scalar=breakdown.total,
        breakdown=breakdown,
        explanation=" | ".join(explanation_parts),
        step=state.step_count,
    )


def compute_completion_reward(state: "InternalState") -> RewardSignal:
    """
    Compute the terminal completion reward upon task resolution.

    Adds bonus for successful completion and tracks completion quality.

    Args:
        state: Final internal state after resolution.

    Returns:
        RewardSignal for the completion event.
    """
    breakdown = RewardBreakdown()
    explanation_parts: list[str] = []

    if state.resolved and not state.failed:
        breakdown.completion_bonus += REWARD_COMPLETION
        explanation_parts.append(f"Task resolved successfully (+{REWARD_COMPLETION}).")

        # Efficiency bonus: fewer steps = higher bonus
        steps_used = state.step_count
        steps_max = state.max_steps
        efficiency_ratio = max(0.0, 1.0 - (steps_used / steps_max))
        efficiency_bonus = round(efficiency_ratio * 0.30, 4)
        breakdown.sequence_bonus += efficiency_bonus
        explanation_parts.append(
            f"Efficiency bonus: {steps_used}/{steps_max} steps used "
            f"(+{efficiency_bonus})."
        )

        # Sentiment recovery bonus
        if state.customer_sentiment >= 0.80:
            breakdown.progress += 0.10
            explanation_parts.append(
                f"Excellent customer sentiment recovery: "
                f"{state.customer_sentiment:.2f} (+0.10)."
            )
        elif state.customer_sentiment >= 0.60:
            breakdown.progress += 0.05
            explanation_parts.append(
                f"Good customer sentiment: {state.customer_sentiment:.2f} (+0.05)."
            )

    return RewardSignal(
        scalar=breakdown.total,
        breakdown=breakdown,
        explanation=" | ".join(explanation_parts),
        step=state.step_count,
    )


# ---------------------------------------------------------------------------
# Final Task Graders (Deterministic 0.0 → 1.0)
# ---------------------------------------------------------------------------


def grade_task_1(final_state: "InternalState") -> float:
    """
    Grade Task 1: Order Tracking.

    Scoring:
        Correctness (40%): tracking_checked AND crm_updated
        Efficiency (30%): inverse of steps_used / max_steps
        Decision Quality (30%): customer sentiment + CRM completeness

    Returns:
        Grade in [0.0, 1.0].
    """
    scores: dict[str, float] = {}

    # Correctness
    correctness = 0.0
    if final_state.tracking_checked:
        correctness += 0.5
    if final_state.crm_updated:
        correctness += 0.5
    scores["correctness"] = correctness * 0.40

    # Efficiency
    ratio = final_state.step_count / max(final_state.max_steps, 1)
    efficiency = max(0.0, 1.0 - ratio)
    scores["efficiency"] = efficiency * 0.30

    # Decision quality
    dq = 0.0
    if final_state.customer_sentiment >= 0.70:
        dq += 0.6
    elif final_state.customer_sentiment >= 0.50:
        dq += 0.3
    if not final_state.refund_issued:  # Refund in Task 1 = wrong decision
        dq += 0.4
    scores["decision_quality"] = (dq / 1.0) * 0.30

    total = sum(scores.values())
    logger.info(
        f"[grade_task_1] Correctness={scores['correctness']:.3f}, "
        f"Efficiency={scores['efficiency']:.3f}, "
        f"DecisionQuality={scores['decision_quality']:.3f} → Total={total:.3f}"
    )
    return round(min(total, 1.0), 4)


def grade_task_2(final_state: "InternalState") -> float:
    """
    Grade Task 2: Shipment Rerouting.

    Scoring:
        Correctness (40%): full sequence adherence
        Efficiency (30%): steps economy
        Decision Quality (30%): rerouting correctness + sentiment recovery

    Returns:
        Grade in [0.0, 1.0].
    """
    scores: dict[str, float] = {}

    # Correctness — all 5 required actions performed
    required = {"tracking_checked", "inventory_checked", "warehouse_found",
                "rerouted", "crm_updated"}
    performed = {
        k for k, v in {
            "tracking_checked": final_state.tracking_checked,
            "inventory_checked": final_state.inventory_checked,
            "warehouse_found": final_state.warehouse_found,
            "rerouted": final_state.rerouted,
            "crm_updated": final_state.crm_updated,
        }.items() if v
    }
    correctness = len(performed & required) / len(required)
    scores["correctness"] = correctness * 0.40

    # Efficiency
    ratio = final_state.step_count / max(final_state.max_steps, 1)
    efficiency = max(0.0, 1.0 - ratio)
    scores["efficiency"] = efficiency * 0.30

    # Decision quality
    dq = 0.0
    if final_state.rerouted:
        dq += 0.5
    if not final_state.refund_issued:
        dq += 0.3
    if final_state.customer_sentiment >= 0.50:
        dq += 0.2
    scores["decision_quality"] = min(dq, 1.0) * 0.30

    total = sum(scores.values())
    logger.info(
        f"[grade_task_2] Correctness={scores['correctness']:.3f}, "
        f"Efficiency={scores['efficiency']:.3f}, "
        f"DecisionQuality={scores['decision_quality']:.3f} → Total={total:.3f}"
    )
    return round(min(total, 1.0), 4)


def grade_task_3(final_state: "InternalState") -> float:
    """
    Grade Task 3: Stockout Crisis Resolution.

    Scoring:
        Correctness (40%): checked inventory + made correct refund decision
        Efficiency (30%): steps economy
        Decision Quality (30%): CRM completeness, avoided destructive reroute, sentiment

    Returns:
        Grade in [0.0, 1.0].
    """
    scores: dict[str, float] = {}

    # Correctness
    correctness = 0.0
    if final_state.tracking_checked:
        correctness += 0.20
    if final_state.inventory_checked:
        correctness += 0.30
    if final_state.refund_issued and not final_state.rerouted:
        correctness += 0.50  # Correct: refund when OOS
    elif final_state.rerouted and not final_state.refund_issued:
        # Wrong: tried to reroute with no stock
        correctness -= 0.40
    scores["correctness"] = max(0.0, correctness) * 0.40

    # Efficiency
    ratio = final_state.step_count / max(final_state.max_steps, 1)
    efficiency = max(0.0, 1.0 - ratio)
    scores["efficiency"] = efficiency * 0.30

    # Decision quality
    dq = 0.0
    if final_state.crm_updated:
        dq += 0.40
    if final_state.customer_sentiment >= 0.30:
        dq += 0.30
    if not final_state.loop_detected:
        dq += 0.30
    scores["decision_quality"] = min(dq, 1.0) * 0.30

    total = sum(scores.values())
    logger.info(
        f"[grade_task_3] Correctness={scores['correctness']:.3f}, "
        f"Efficiency={scores['efficiency']:.3f}, "
        f"DecisionQuality={scores['decision_quality']:.3f} → Total={total:.3f}"
    )
    return round(min(total, 1.0), 4)


# ---------------------------------------------------------------------------
# Grade Dispatcher
# ---------------------------------------------------------------------------

TASK_GRADERS = {
    "order_tracking": grade_task_1,
    "shipment_rerouting": grade_task_2,
    "stockout_crisis": grade_task_3,
}


def grade(task_id: str, final_state: "InternalState") -> float:
    """
    Deterministic grader dispatcher. Returns a score in [0.0, 1.0].

    Args:
        task_id: One of the canonical task IDs.
        final_state: The fully resolved internal state.

    Returns:
        Float score in [0.0, 1.0].

    Raises:
        ValueError: If task_id is unrecognized.
    """
    if task_id not in TASK_GRADERS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid: {list(TASK_GRADERS.keys())}"
        )
    return TASK_GRADERS[task_id](final_state)
