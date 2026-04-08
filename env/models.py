"""
LogisticsHub-360: Pydantic Data Models
Strongly typed models for observation, action, reward, and internal state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import time


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ToolName(str, Enum):
    GET_TRACKING = "get_tracking"
    CHECK_INVENTORY = "check_inventory"
    FIND_WAREHOUSE = "find_warehouse"
    REROUTE_ORDER = "reroute_order"
    ISSUE_REFUND = "issue_refund"
    UPDATE_CRM = "update_crm"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SHIPPED = "shipped"
    DELAYED = "delayed"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    RESOLVED = "resolved"


class InventoryLevel(str, Enum):
    HIGH = "high"
    LOW = "low"
    OUT_OF_STOCK = "out_of_stock"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskID(str, Enum):
    ORDER_TRACKING = "order_tracking"
    SHIPMENT_REROUTING = "shipment_rerouting"
    STOCKOUT_CRISIS = "stockout_crisis"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class OrderInfo(BaseModel):
    order_id: str
    customer_id: str
    product_id: str
    product_name: str
    quantity: int
    status: OrderStatus
    warehouse_id: str
    destination: str
    carrier: str
    estimated_delivery: str
    last_known_location: Optional[str] = None
    delay_reason: Optional[str] = None

    model_config = {"use_enum_values": True}


class WarehouseInfo(BaseModel):
    warehouse_id: str
    location: str
    available_stock: int
    distance_km: float
    estimated_reroute_days: int


class InventoryInfo(BaseModel):
    product_id: str
    product_name: str
    level: InventoryLevel
    quantity: int
    warehouses: List[WarehouseInfo] = Field(default_factory=list)

    model_config = {"use_enum_values": True}


class RewardBreakdown(BaseModel):
    progress: float = 0.0
    tool_use: float = 0.0
    sequence_bonus: float = 0.0
    penalty: float = 0.0
    completion_bonus: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.progress + self.tool_use + self.sequence_bonus
            + self.penalty + self.completion_bonus,
            4,
        )


class RewardSignal(BaseModel):
    scalar: float
    breakdown: RewardBreakdown
    explanation: str
    step: int


class ActionHistoryEntry(BaseModel):
    step: int
    tool: str
    parameters: Dict[str, Any]
    result_summary: str
    reward: float
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Primary Interface Models
# ---------------------------------------------------------------------------

class Action(BaseModel):
    tool: ToolName
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

    model_config = {"use_enum_values": True}

    @field_validator("parameters")
    @classmethod
    def parameters_must_be_serializable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        for key, val in v.items():
            if not isinstance(key, str):
                raise ValueError(f"Parameter key '{key}' must be a string.")
        return v


class Observation(BaseModel):
    task_id: str
    task_description: str
    difficulty: TaskDifficulty
    system_logs: List[str]
    order_status: Optional[OrderInfo] = None
    inventory_state: Optional[InventoryInfo] = None
    customer_sentiment: float = Field(ge=0.0, le=1.0, default=0.8)
    available_tools: List[str]
    action_history: List[ActionHistoryEntry] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    step_count: int = 0
    max_steps: int = 20
    is_done: bool = False
    last_reward: Optional[float] = None

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Internal State (not exposed to agent directly — debug only)
# ---------------------------------------------------------------------------

class InternalState(BaseModel):
    """Full internal state of the environment — for graders and debugging."""

    task_id: str
    difficulty: TaskDifficulty
    step_count: int = 0
    max_steps: int = 20
    resolved: bool = False
    failed: bool = False
    loop_detected: bool = False

    order: OrderInfo
    inventory: InventoryInfo

    customer_sentiment: float = Field(ge=0.0, le=1.0, default=0.8)
    crm_updated: bool = False
    refund_issued: bool = False
    rerouted: bool = False
    tracking_checked: bool = False
    inventory_checked: bool = False
    warehouse_found: bool = False
    warehouse_id_selected: Optional[str] = None

    action_history: List[ActionHistoryEntry] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    system_logs: List[str] = Field(default_factory=list)
    repeated_actions: Dict[str, int] = Field(default_factory=dict)

    # Task-specific expected action sequences
    expected_sequence: List[str] = Field(default_factory=list)
    completed_sequence_steps: List[str] = Field(default_factory=list)

    model_config = {"use_enum_values": True}
