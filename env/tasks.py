"""
LogisticsHub-360: Task Definitions
Defines the three canonical evaluation tasks (Easy / Medium / Hard)
with their initial states, expected action sequences, and hints.
"""

from __future__ import annotations

from typing import Dict, Any

from env.models import (
    TaskID,
    TaskDifficulty,
    OrderInfo,
    OrderStatus,
    WarehouseInfo,
    InventoryInfo,
    InventoryLevel,
    InternalState,
)


# ---------------------------------------------------------------------------
# Task Scaffolding Builders
# ---------------------------------------------------------------------------

def _make_wh(wid: str, loc: str, stock: int, dist: float, days: int) -> WarehouseInfo:
    return WarehouseInfo(
        warehouse_id=wid,
        location=loc,
        available_stock=stock,
        distance_km=dist,
        estimated_reroute_days=days,
    )


# ---------------------------------------------------------------------------
# Task 1 — Order Tracking (Easy)
# ---------------------------------------------------------------------------

TASK_1_ID = TaskID.ORDER_TRACKING.value
TASK_1_DESCRIPTION = (
    "A customer (ID: C-1001) is inquiring about the status of their order "
    "ORD-88421 for a 'Wireless Noise-Cancelling Headphones' unit. "
    "The order was shipped 3 days ago but the customer has not received a "
    "tracking update. Your objective: retrieve the latest tracking information "
    "and communicate an accurate status back to the customer via the CRM."
)
TASK_1_HINTS = [
    "Start by calling get_tracking with the correct order_id.",
    "After retrieving status, update the CRM to inform the customer.",
]
TASK_1_CONSTRAINTS = [
    "You must not issue a refund unless the order is provably lost.",
    "CRM update must include the current order status.",
]
TASK_1_SEQUENCE = ["get_tracking", "update_crm"]
TASK_1_MAX_STEPS = 8


def build_task_1_state() -> InternalState:
    order = OrderInfo(
        order_id="ORD-88421",
        customer_id="C-1001",
        product_id="PROD-WH-7723",
        product_name="Wireless Noise-Cancelling Headphones",
        quantity=1,
        status=OrderStatus.SHIPPED,
        warehouse_id="WH-EAST-01",
        destination="New York, NY",
        carrier="FedEx",
        estimated_delivery="2026-04-12",
        last_known_location="Atlanta, GA Distribution Center",
        delay_reason=None,
    )

    inventory = InventoryInfo(
        product_id="PROD-WH-7723",
        product_name="Wireless Noise-Cancelling Headphones",
        level=InventoryLevel.HIGH,
        quantity=142,
        warehouses=[
            _make_wh("WH-EAST-01", "Atlanta, GA", 142, 120.5, 1),
            _make_wh("WH-CENTRAL-02", "Chicago, IL", 87, 340.2, 2),
        ],
    )

    return InternalState(
        task_id=TASK_1_ID,
        difficulty=TaskDifficulty.EASY,
        max_steps=TASK_1_MAX_STEPS,
        order=order,
        inventory=inventory,
        customer_sentiment=0.65,
        expected_sequence=TASK_1_SEQUENCE,
        system_logs=["[SYSTEM] Task 1 initialized: Order Tracking scenario."],
    )


# ---------------------------------------------------------------------------
# Task 2 — Shipment Rerouting (Medium)
# ---------------------------------------------------------------------------

TASK_2_ID = TaskID.SHIPMENT_REROUTING.value
TASK_2_DESCRIPTION = (
    "Order ORD-44790 for a 'Smart Home Security Camera System' is currently "
    "DELAYED due to a severe weather disruption at the WH-WEST-05 hub. "
    "The customer (C-2047) is frustrated with a sentiment score of 0.35. "
    "Your objective: detect the delay, verify inventory availability, locate "
    "the nearest alternative warehouse, and reroute the shipment to ensure "
    "fastest possible delivery. Update the CRM with the resolution."
)
TASK_2_HINTS = [
    "Call get_tracking to confirm the delay status.",
    "Then check_inventory for alternate stock.",
    "Use find_warehouse to identify the best rerouting candidate.",
    "Complete the reroute with reroute_order.",
    "Notify the customer via update_crm.",
]
TASK_2_CONSTRAINTS = [
    "Do not issue a refund — the product is available.",
    "Rerouting must happen before CRM update.",
    "Use only the warehouse returned by find_warehouse.",
]
TASK_2_SEQUENCE = [
    "get_tracking",
    "check_inventory",
    "find_warehouse",
    "reroute_order",
    "update_crm",
]
TASK_2_MAX_STEPS = 15


def build_task_2_state() -> InternalState:
    order = OrderInfo(
        order_id="ORD-44790",
        customer_id="C-2047",
        product_id="PROD-CAM-3310",
        product_name="Smart Home Security Camera System",
        quantity=1,
        status=OrderStatus.DELAYED,
        warehouse_id="WH-WEST-05",
        destination="Los Angeles, CA",
        carrier="UPS",
        estimated_delivery="2026-04-15",
        last_known_location="Phoenix, AZ — Delayed (Weather)",
        delay_reason="Severe weather conditions at distribution hub WH-WEST-05.",
    )

    inventory = InventoryInfo(
        product_id="PROD-CAM-3310",
        product_name="Smart Home Security Camera System",
        level=InventoryLevel.LOW,
        quantity=23,
        warehouses=[
            _make_wh("WH-SOUTH-03", "Las Vegas, NV", 15, 280.0, 2),
            _make_wh("WH-WEST-08", "San Diego, CA", 8, 190.5, 1),
            _make_wh("WH-CENTRAL-04", "Dallas, TX", 0, 1500.0, 4),
        ],
    )

    return InternalState(
        task_id=TASK_2_ID,
        difficulty=TaskDifficulty.MEDIUM,
        max_steps=TASK_2_MAX_STEPS,
        order=order,
        inventory=inventory,
        customer_sentiment=0.35,
        expected_sequence=TASK_2_SEQUENCE,
        system_logs=["[SYSTEM] Task 2 initialized: Shipment Rerouting scenario."],
    )


# ---------------------------------------------------------------------------
# Task 3 — Stockout Crisis Resolution (Hard)
# ---------------------------------------------------------------------------

TASK_3_ID = TaskID.STOCKOUT_CRISIS.value
TASK_3_DESCRIPTION = (
    "CRITICAL ALERT: Order ORD-99123 for 'Ultra Gaming Laptop Pro' is DELAYED "
    "and the product is completely OUT OF STOCK across all primary warehouses. "
    "Customer C-3391 is extremely unhappy (sentiment: 0.20). "
    "\n\nThe agent must: "
    "(1) Check the current order status via tracking. "
    "(2) Verify inventory to confirm complete stockout. "
    "(3) Attempt to find any warehouse with available stock. "
    "(4) Based on findings, decide to EITHER reroute (if stock found) OR "
    "issue a refund (if no stock is available). "
    "(5) Update CRM with a detailed resolution message. "
    "\nBranching paths exist — incorrect decisions are penalized. "
    "Recover customer trust by maximizing sentiment through the correct sequence."
)
TASK_3_HINTS = [
    "Begin with get_tracking to understand order status.",
    "Always check inventory before making a resolution decision.",
    "If no stock is available in any warehouse, issue a refund — do not reroute.",
    "Rerouting to a stockout warehouse is a destructive action.",
    "CRM update must reflect the specific resolution action taken.",
]
TASK_3_CONSTRAINTS = [
    "You MUST verify inventory before deciding between refund or reroute.",
    "Do NOT reroute if all warehouses are out of stock.",
    "CRM must be updated LAST, after the resolution action.",
    "A reroute attempt to a fully stocked-out warehouse triggers a -1.0 penalty.",
]
TASK_3_SEQUENCE = [
    "get_tracking",
    "check_inventory",
    "find_warehouse",         # Will fail — no stock
    "issue_refund",           # Correct fallback
    "update_crm",
]
TASK_3_MAX_STEPS = 20


def build_task_3_state() -> InternalState:
    order = OrderInfo(
        order_id="ORD-99123",
        customer_id="C-3391",
        product_id="PROD-LPT-0055",
        product_name="Ultra Gaming Laptop Pro",
        quantity=1,
        status=OrderStatus.DELAYED,
        warehouse_id="WH-NORTH-09",
        destination="Seattle, WA",
        carrier="USPS Priority",
        estimated_delivery="2026-04-10",
        last_known_location="Portland, OR — Held (Product Recall Review)",
        delay_reason="Product batch flagged for quality review. All primary stock exhausted.",
    )

    inventory = InventoryInfo(
        product_id="PROD-LPT-0055",
        product_name="Ultra Gaming Laptop Pro",
        level=InventoryLevel.OUT_OF_STOCK,
        quantity=0,
        warehouses=[
            _make_wh("WH-NORTH-09", "Portland, OR", 0, 175.0, 0),
            _make_wh("WH-WEST-02", "San Francisco, CA", 0, 800.0, 0),
            _make_wh("WH-EAST-07", "New York, NY", 0, 2900.0, 0),
        ],
    )

    return InternalState(
        task_id=TASK_3_ID,
        difficulty=TaskDifficulty.HARD,
        max_steps=TASK_3_MAX_STEPS,
        order=order,
        inventory=inventory,
        customer_sentiment=0.20,
        expected_sequence=TASK_3_SEQUENCE,
        system_logs=["[SYSTEM] Task 3 initialized: Stockout Crisis scenario — HIGH PRIORITY."],
    )


# ---------------------------------------------------------------------------
# Public Task Registry
# ---------------------------------------------------------------------------

TASK_BUILDERS: Dict[str, Any] = {
    TASK_1_ID: {
        "builder": build_task_1_state,
        "id": TASK_1_ID,
        "description": TASK_1_DESCRIPTION,
        "difficulty": TaskDifficulty.EASY,
        "hints": TASK_1_HINTS,
        "constraints": TASK_1_CONSTRAINTS,
        "max_steps": TASK_1_MAX_STEPS,
    },
    TASK_2_ID: {
        "builder": build_task_2_state,
        "id": TASK_2_ID,
        "description": TASK_2_DESCRIPTION,
        "difficulty": TaskDifficulty.MEDIUM,
        "hints": TASK_2_HINTS,
        "constraints": TASK_2_CONSTRAINTS,
        "max_steps": TASK_2_MAX_STEPS,
    },
    TASK_3_ID: {
        "builder": build_task_3_state,
        "id": TASK_3_ID,
        "description": TASK_3_DESCRIPTION,
        "difficulty": TaskDifficulty.HARD,
        "hints": TASK_3_HINTS,
        "constraints": TASK_3_CONSTRAINTS,
        "max_steps": TASK_3_MAX_STEPS,
    },
}

TASK_ORDER = [TASK_1_ID, TASK_2_ID, TASK_3_ID]
