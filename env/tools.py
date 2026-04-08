"""
LogisticsHub-360: Tool Implementations
Each tool simulates a real logistics API, modifies internal state,
produces logs, and contributes to reward computation.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from env.models import InternalState

logger = logging.getLogger("logisticshub360.tools")


# ---------------------------------------------------------------------------
# Tool Result Schema
# ---------------------------------------------------------------------------

class ToolResult:
    """Encapsulates the result of a tool call."""

    def __init__(
        self,
        success: bool,
        data: Dict[str, Any],
        log_message: str,
        state_change_summary: str,
    ):
        self.success = success
        self.data = data
        self.log_message = log_message
        self.state_change_summary = state_change_summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "log": self.log_message,
            "state_change": self.state_change_summary,
        }


# ---------------------------------------------------------------------------
# Tool: get_tracking
# ---------------------------------------------------------------------------

def get_tracking(order_id: str, state: "InternalState") -> ToolResult:
    """
    Retrieve real-time tracking information for an order.

    Args:
        order_id: The order identifier to track.
        state: Current internal environment state (mutated in place).

    Returns:
        ToolResult with tracking details.
    """
    if state.order.order_id != order_id:
        msg = f"[get_tracking] Order '{order_id}' not found in current session."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "order_not_found", "order_id": order_id},
            log_message=msg,
            state_change_summary="No state change — invalid order_id.",
        )

    state.tracking_checked = True
    tracking_data = {
        "order_id": state.order.order_id,
        "status": state.order.status,
        "carrier": state.order.carrier,
        "last_location": state.order.last_known_location or "Unknown",
        "estimated_delivery": state.order.estimated_delivery,
        "delay_reason": state.order.delay_reason,
    }
    msg = (
        f"[get_tracking] Order {order_id} → status={state.order.status}, "
        f"location={state.order.last_known_location}, carrier={state.order.carrier}."
    )
    logger.info(msg)
    state.system_logs.append(msg)

    return ToolResult(
        success=True,
        data=tracking_data,
        log_message=msg,
        state_change_summary="tracking_checked set to True.",
    )


# ---------------------------------------------------------------------------
# Tool: check_inventory
# ---------------------------------------------------------------------------

def check_inventory(product_id: str, state: "InternalState") -> ToolResult:
    """
    Query current inventory level and warehouse availability for a product.

    Args:
        product_id: The product SKU or identifier.
        state: Current internal environment state (mutated in place).

    Returns:
        ToolResult with inventory details and warehouse list.
    """
    if state.inventory.product_id != product_id:
        msg = f"[check_inventory] Product '{product_id}' not found in catalog."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "product_not_found", "product_id": product_id},
            log_message=msg,
            state_change_summary="No state change — invalid product_id.",
        )

    state.inventory_checked = True
    inv = state.inventory
    warehouse_list = [
        {
            "warehouse_id": w.warehouse_id,
            "location": w.location,
            "available_stock": w.available_stock,
            "distance_km": w.distance_km,
            "estimated_reroute_days": w.estimated_reroute_days,
        }
        for w in inv.warehouses
    ]

    msg = (
        f"[check_inventory] Product {product_id} → level={inv.level}, "
        f"qty={inv.quantity}, warehouses_available={len(inv.warehouses)}."
    )
    logger.info(msg)
    state.system_logs.append(msg)

    return ToolResult(
        success=True,
        data={
            "product_id": inv.product_id,
            "product_name": inv.product_name,
            "level": inv.level,
            "quantity": inv.quantity,
            "warehouses": warehouse_list,
        },
        log_message=msg,
        state_change_summary="inventory_checked set to True.",
    )


# ---------------------------------------------------------------------------
# Tool: find_warehouse
# ---------------------------------------------------------------------------

def find_warehouse(location: str, state: "InternalState") -> ToolResult:
    """
    Locate the nearest warehouse with available stock to a given destination.

    Args:
        location: Target delivery location or city name.
        state: Current internal environment state (mutated in place).

    Returns:
        ToolResult with the best warehouse candidate.
    """
    if not state.inventory_checked:
        msg = (
            "[find_warehouse] Inventory has not been checked. "
            "Call check_inventory before find_warehouse."
        )
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "inventory_not_checked"},
            log_message=msg,
            state_change_summary="No state change — prerequisite not met.",
        )

    candidates = [w for w in state.inventory.warehouses if w.available_stock > 0]
    if not candidates:
        msg = f"[find_warehouse] No warehouses with available stock near '{location}'."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "no_stock_available", "location": location},
            log_message=msg,
            state_change_summary="No state change — all warehouses out of stock.",
        )

    # Select closest warehouse by simulated distance match
    best = min(candidates, key=lambda w: w.distance_km)
    state.warehouse_found = True
    state.warehouse_id_selected = best.warehouse_id

    msg = (
        f"[find_warehouse] Best warehouse for '{location}': {best.warehouse_id} "
        f"({best.location}), stock={best.available_stock}, "
        f"reroute_days={best.estimated_reroute_days}."
    )
    logger.info(msg)
    state.system_logs.append(msg)

    return ToolResult(
        success=True,
        data={
            "warehouse_id": best.warehouse_id,
            "location": best.location,
            "available_stock": best.available_stock,
            "distance_km": best.distance_km,
            "estimated_reroute_days": best.estimated_reroute_days,
        },
        log_message=msg,
        state_change_summary=f"warehouse_found=True, warehouse_id_selected={best.warehouse_id}.",
    )


# ---------------------------------------------------------------------------
# Tool: reroute_order
# ---------------------------------------------------------------------------

def reroute_order(
    order_id: str, warehouse_id: str, state: "InternalState"
) -> ToolResult:
    """
    Reroute an order to a specified warehouse.

    Requires tracking to have been checked and a warehouse to have been found.

    Args:
        order_id: The order to reroute.
        warehouse_id: Target warehouse identifier.
        state: Current internal environment state (mutated in place).

    Returns:
        ToolResult confirming rerouting or describing failure.
    """
    if state.order.order_id != order_id:
        msg = f"[reroute_order] Order '{order_id}' not found."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "order_not_found"},
            log_message=msg,
            state_change_summary="No state change.",
        )

    if not state.tracking_checked:
        msg = "[reroute_order] Tracking has not been verified before rerouting."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "tracking_not_checked"},
            log_message=msg,
            state_change_summary="No state change — tracking prerequisite not met.",
        )

    if not state.warehouse_found:
        msg = "[reroute_order] No warehouse selected. Run find_warehouse first."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "warehouse_not_selected"},
            log_message=msg,
            state_change_summary="No state change — warehouse prerequisite not met.",
        )

    if warehouse_id != state.warehouse_id_selected:
        msg = (
            f"[reroute_order] Warehouse '{warehouse_id}' does not match selected "
            f"warehouse '{state.warehouse_id_selected}'."
        )
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "warehouse_mismatch", "expected": state.warehouse_id_selected},
            log_message=msg,
            state_change_summary="No state change — warehouse mismatch.",
        )

    # Perform reroute
    state.rerouted = True
    state.order.status = "resolved"  # type: ignore[assignment]
    state.order.warehouse_id = warehouse_id
    state.customer_sentiment = min(1.0, state.customer_sentiment + 0.15)

    msg = (
        f"[reroute_order] Order {order_id} successfully rerouted to warehouse "
        f"{warehouse_id}. Customer sentiment improved to {state.customer_sentiment:.2f}."
    )
    logger.info(msg)
    state.system_logs.append(msg)

    return ToolResult(
        success=True,
        data={
            "order_id": order_id,
            "new_warehouse": warehouse_id,
            "new_status": state.order.status,
            "customer_sentiment": state.customer_sentiment,
        },
        log_message=msg,
        state_change_summary="rerouted=True, order.status=resolved, sentiment++.",
    )


# ---------------------------------------------------------------------------
# Tool: issue_refund
# ---------------------------------------------------------------------------

def issue_refund(order_id: str, state: "InternalState") -> ToolResult:
    """
    Issue a full refund for a given order.

    Should only be used when no viable resolution path exists
    (e.g., stockout with no alternatives).

    Args:
        order_id: The order to refund.
        state: Current internal environment state (mutated in place).

    Returns:
        ToolResult confirming refund or detailing failure reason.
    """
    if state.order.order_id != order_id:
        msg = f"[issue_refund] Order '{order_id}' not found."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "order_not_found"},
            log_message=msg,
            state_change_summary="No state change.",
        )

    if state.order.status in ("refunded", "resolved", "delivered"):
        msg = (
            f"[issue_refund] Order {order_id} is already in terminal state: "
            f"{state.order.status}. Refund not applicable."
        )
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "already_terminal", "status": state.order.status},
            log_message=msg,
            state_change_summary="No state change — already terminal.",
        )

    state.refund_issued = True
    state.order.status = "refunded"  # type: ignore[assignment]
    state.customer_sentiment = min(1.0, state.customer_sentiment + 0.1)

    msg = (
        f"[issue_refund] Refund issued for order {order_id}. "
        f"Customer sentiment: {state.customer_sentiment:.2f}."
    )
    logger.info(msg)
    state.system_logs.append(msg)

    return ToolResult(
        success=True,
        data={
            "order_id": order_id,
            "refund_status": "approved",
            "new_order_status": state.order.status,
            "customer_sentiment": state.customer_sentiment,
        },
        log_message=msg,
        state_change_summary="refund_issued=True, order.status=refunded, sentiment++.",
    )


# ---------------------------------------------------------------------------
# Tool: update_crm
# ---------------------------------------------------------------------------

def update_crm(order_id: str, message: str, state: "InternalState") -> ToolResult:
    """
    Log a CRM note for the customer associated with the given order.

    Args:
        order_id: The relevant order identifier.
        message: The message or note to log in the CRM system.
        state: Current internal environment state (mutated in place).

    Returns:
        ToolResult confirming CRM update.
    """
    if state.order.order_id != order_id:
        msg = f"[update_crm] Order '{order_id}' not found."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "order_not_found"},
            log_message=msg,
            state_change_summary="No state change.",
        )

    if not message or len(message.strip()) < 5:
        msg = "[update_crm] CRM message is too short or empty."
        logger.warning(msg)
        state.system_logs.append(msg)
        return ToolResult(
            success=False,
            data={"error": "message_too_short"},
            log_message=msg,
            state_change_summary="No state change.",
        )

    state.crm_updated = True
    state.customer_sentiment = min(1.0, state.customer_sentiment + 0.05)

    msg = (
        f"[update_crm] CRM updated for order {order_id}: '{message[:80]}'. "
        f"Customer sentiment: {state.customer_sentiment:.2f}."
    )
    logger.info(msg)
    state.system_logs.append(msg)

    return ToolResult(
        success=True,
        data={
            "order_id": order_id,
            "crm_entry": message,
            "customer_sentiment": state.customer_sentiment,
        },
        log_message=msg,
        state_change_summary="crm_updated=True, sentiment++.",
    )


# ---------------------------------------------------------------------------
# Tool Dispatch
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_tracking": get_tracking,
    "check_inventory": check_inventory,
    "find_warehouse": find_warehouse,
    "reroute_order": reroute_order,
    "issue_refund": issue_refund,
    "update_crm": update_crm,
}
