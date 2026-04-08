---
title: LogisticsHub-360
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
fullWidth: true
short_description: AI benchmarking for real-world logistics ops
tags:
  - openenv
  - logistics
  - ecommerce
  - ai-agent
  - benchmark
  - reinforcement-learning
models:
  - mistralai/Mistral-7B-Instruct-v0.3
  - meta-llama/Llama-3.1-8B-Instruct
disable_embedding: false
---


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference



# LogisticsHub-360: Intelligent E-Commerce Operations Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-1.0-blue.svg)](https://openenv.ai)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E.svg)](https://huggingface.co/spaces)

---

## Overview

**LogisticsHub-360** is a production-grade, OpenEnv-compliant AI evaluation environment designed to benchmark intelligent agents on real-world e-commerce backend operations. Built for the **Meta + Hugging Face OpenEnv Hackathon**, it provides a fully stateful, multi-step simulation of logistics workflows—where the AI agent must reason, plan, and execute sequences of API calls to resolve business-critical scenarios.

Unlike toy environments with binary rewards, LogisticsHub-360 implements **dense, per-step reward signals**, **partial observability**, **loop detection**, and **deterministic multi-dimensional graders** that score agents on correctness, efficiency, and decision quality.

### Why This Environment Matters

E-commerce logistics is one of the most high-stakes, time-sensitive domains in modern business. A delayed shipment costs customer trust. A stockout mishandled costs both a sale and a relationship. Logan the logistics agent must navigate these situations with the same care and precision as a senior operations team. This environment tests whether an AI can meet that bar.

---

## Environment Design

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     LogisticsHub-360 Environment                  │
│                                                                    │
│  ┌──────────┐   Action   ┌───────────────┐   ToolResult          │
│  │  Agent   │ ─────────► │  Tool Layer   │ ──────────┐           │
│  │  (LLM)   │            │  (6 APIs)     │           │           │
│  └──────────┘            └───────────────┘           ▼           │
│        ▲                                    ┌─────────────────┐  │
│        │  Observation                       │  State Manager  │  │
│        └────────────────────────────────── │  (InternalState)│  │
│                                            └─────────────────┘  │
│                                                     │            │
│                                            ┌────────▼────────┐  │
│                                            │  Reward Engine  │  │
│                                            │  + Graders      │  │
│                                            └─────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Core Properties

| Property | Value |
|---|---|
| **Interaction Model** | Stateful, sequential tool-call API |
| **Observability** | Partial (agent sees logs and status, not internal flags) |
| **Reward Type** | Dense per-step + terminal completion bonus |
| **Determinism** | Fully deterministic (fixed scenarios, reproducible grading) |
| **Max Episode Length** | Configurable per task (8–20 steps) |

---

## Observation Space

Each step returns a structured `Observation` object:

```python
class Observation(BaseModel):
    task_id: str                          # Task being evaluated
    task_description: str                 # Full natural language description
    difficulty: TaskDifficulty            # easy | medium | hard
    system_logs: List[str]                # Recent system log entries (last 10)
    order_status: Optional[OrderInfo]     # Current order details
    inventory_state: Optional[InventoryInfo]  # Product inventory and warehouses
    customer_sentiment: float             # 0.0 (hostile) → 1.0 (satisfied)
    available_tools: List[str]            # List of callable tool names
    action_history: List[ActionHistoryEntry]  # History of past actions + rewards
    hints: List[str]                      # Task-specific guidance hints
    constraints: List[str]                # Rules the agent must not violate
    step_count: int                       # Current step number
    max_steps: int                        # Episode step budget
    is_done: bool                         # Whether the episode has terminated
    last_reward: Optional[float]          # Reward from the previous action
```

---

## Action Space

Agents submit actions as structured `Action` objects:

```python
class Action(BaseModel):
    tool: ToolName           # One of 6 available tools (enum)
    parameters: Dict[str, Any]  # Tool-specific parameters
    metadata: Optional[Dict]    # Optional agent metadata
```

### Available Tools

| Tool | Parameters | Description |
|---|---|---|
| `get_tracking` | `order_id: str` | Retrieve real-time shipment tracking info |
| `check_inventory` | `product_id: str` | Query stock levels and warehouse locations |
| `find_warehouse` | `location: str` | Locate optimal alternate warehouse |
| `reroute_order` | `order_id: str, warehouse_id: str` | Reroute shipment to specified warehouse |
| `issue_refund` | `order_id: str` | Issue full refund when no resolution exists |
| `update_crm` | `order_id: str, message: str` | Log customer communication in CRM |

---

## Task Descriptions

### 🟢 Task 1: Order Tracking (Easy)

**Scenario**: A customer inquires about order `ORD-88421` (Wireless Headphones). The order was shipped 3 days ago but tracking hasn't updated. Customer sentiment: `0.65`.

**Objective**: Call `get_tracking` to retrieve the current status, then notify the customer via `update_crm`.

**Expected Sequence**:
```
get_tracking → update_crm
```

**Max Steps**: 8 | **Success Criteria**: Both tracking checked + CRM updated.

---

### 🟡 Task 2: Shipment Rerouting (Medium)

**Scenario**: Order `ORD-44790` (Security Camera) is **DELAYED** due to severe weather at hub `WH-WEST-05`. Customer sentiment is critically low at `0.35`.

**Objective**: Detect the delay, identify alternate stock, locate the optimal rerouting warehouse, execute the reroute, and notify the customer.

**Expected Sequence**:
```
get_tracking → check_inventory → find_warehouse → reroute_order → update_crm
```

**Max Steps**: 15 | **Success Criteria**: Full 5-step sequence completed correctly.

---

### 🔴 Task 3: Stockout Crisis Resolution (Hard)

**Scenario**: Order `ORD-99123` (Gaming Laptop) is **DELAYED** and the product is **COMPLETELY OUT OF STOCK** across all warehouses. Customer sentiment: `0.20` (critical).

**Objective**: Verify the stockout through inventory check, attempt warehouse lookup (which fails), then **issue a refund** (not a reroute), and update the CRM.

**Critical Branching Decision**: If the agent attempts to reroute instead of issuing a refund despite confirmed stockout, it receives a `-1.0 destructive action penalty`.

**Expected Sequence**:
```
get_tracking → check_inventory → find_warehouse → issue_refund → update_crm
```

**Max Steps**: 20 | **Success Criteria**: Refund issued (not reroute) + CRM updated.

---

## Reward Function

### Per-Step Rewards

| Event | Reward |
|---|---|
| Valid tool executed successfully | **+0.30** |
| Correct sequence step executed | **+0.20** |
| Out-of-order but correct tool | **+0.10** |
| Invalid action (wrong params/prereqs) | **−0.10** |
| Same invalid action repeated | **−0.30** |
| Wrong strategic decision | **−0.50** |
| Destructive action (e.g., reroute during full stockout) | **−1.00** |

### Terminal Rewards

| Event | Reward |
|---|---|
| Task successfully completed | **+1.00** |
| Efficiency bonus (steps saved) | **up to +0.30** |
| High sentiment recovery (≥0.80) | **+0.10** |
| Moderate sentiment recovery (≥0.60) | **+0.05** |

### Final Grading Weights

| Dimension | Weight |
|---|---|
| Correctness (correct tools, sequence, terminal state) | **40%** |
| Efficiency (steps used vs. budget) | **30%** |
| Decision Quality (sentiment, avoidance of wrong paths) | **30%** |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- A Hugging Face account with API token (free tier works)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/your-org/logisticshub-360-openenv.git
cd logisticshub-360-openenv

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Validation

```bash
# Validate environment can be imported and run
python -c "
from env.environment import LogisticsHub360Env
env = LogisticsHub360Env('order_tracking')
obs = env.reset()
print('✅ Environment validated successfully')
print(f'   Task: {obs.task_id}')
print(f'   Max Steps: {obs.max_steps}')
print(f'   Tools: {obs.available_tools}')
```

### Running the Interactive Web App (Gradio)

LogisticsHub-360 includes a rich interactive web UI where you can play as the agent in Human Mode, or watch the AI solve tasks in AI Agent Mode.

```bash
# Set your HF token (Required for AI Agent Mode)
export HF_TOKEN="your_token_here" # On Windows PowerShell: $env:HF_TOKEN="your_token_here"

# Start the web app
python app.py

# To specify a different port (default is 7860)
PORT=7865 python app.py  # On Windows PowerShell: $env:PORT="7865"; python app.py
```

Open `http://127.0.0.1:7860` (or your chosen port) in your browser to view the interface.

### Running the Baseline Agent

```bash
# Set your HF token
export HF_TOKEN=your_token_here

# Run all tasks
python scripts/run_inference.py

# Run a specific task
python scripts/run_inference.py --task order_tracking

# Save results to JSON
python scripts/run_inference.py --output results.json

# Quiet mode (no step-by-step output)
python scripts/run_inference.py --quiet
```

---

## Docker Usage

### Build

```bash
docker build -t logistics-env .
```

### Run (requires HF token)

```bash
# Run the full evaluation
docker run -e HF_TOKEN=your_token_here logistics-env

# Run a specific task
docker run -e HF_TOKEN=your_token_here logistics-env --task shipment_rerouting

# With custom model
docker run \
  -e HF_TOKEN=your_token_here \
  -e LH360_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  logistics-env
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Hugging Face API token |
| `LH360_BASE_URL` | `https://api-inference.huggingface.co/v1` | OpenAI-compatible API base URL |
| `LH360_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | Model to use for inference |
| `LH360_MAX_RETRIES` | `3` | API call retry attempts |
| `LH360_TEMPERATURE` | `0.2` | Sampling temperature |

---

## Hugging Face Spaces Deployment

1. Fork this repository to your HF account
2. Create a new **Docker** Space
3. Link to your repository
4. Add `HF_TOKEN` as a Space secret
5. Push — the Space will auto-build from the `Dockerfile`

---

## API Usage (Python)

```python
from env.environment import LogisticsHub360Env
from env.models import Action, ToolName

# Initialize environment
env = LogisticsHub360Env(task_id="shipment_rerouting")
obs = env.reset()

# Step through manually
action = Action(
    tool=ToolName.GET_TRACKING,
    parameters={"order_id": "ORD-44790"}
)
obs, reward, done, info = env.step(action)
print(f"Reward: {reward:.3f}")
print(f"Info: {info['explanation']}")

# Get debug state (full internal state — not visible to agent)
debug_state = env.state()

# Final grade
grade = env.grade_episode()
print(f"Episode Grade: {grade:.4f}")
```

---

## Example Output

```
======================================================================
  TASK: ORDER_TRACKING | Difficulty: easy
======================================================================
  Description: A customer (ID: C-1001) is inquiring about the status of their order...

  [Step 1/8] Tool: get_tracking | Params: {'order_id': 'ORD-88421'}
             → Reward: +0.500 | Cumulative: +0.500
             → Valid tool 'get_tracking' executed (+0.30). Correct sequence step #1 (+0.20).

  [Step 2/8] Tool: update_crm | Params: {'order_id': 'ORD-88421', 'message': 'Your order ORD-88421 is currently...'}
             → Reward: +1.850 | Cumulative: +2.350
             → Valid tool 'update_crm' executed (+0.30). Correct sequence step #2 (+0.20). Completion bonus (+1.00). Efficiency bonus (+0.30).

  ✅ Task 'order_tracking' complete.
     Cumulative Reward : 2.3500
     Final Grade       : 0.9500 / 1.0
     Steps Used        : 2 / 8
     Customer Sentiment: 0.70
```

## Baseline Performance

| Task | Optimal Grade | Mistral-7B | Llama-3.1-8B |
|---|---|---|---|
| Order Tracking (Easy) | 1.00 | ~0.85 | ~0.88 |
| Shipment Rerouting (Medium) | 1.00 | ~0.65 | ~0.72 |
| Stockout Crisis (Hard) | 1.00 | ~0.50 | ~0.61 |
| **Average** | **1.00** | **~0.67** | **~0.74** |

> *Baseline numbers are indicative. Actual results vary with model version and API latency.*

---

## Project Structure

```
logisticshub-360-openenv/
├── env/
│   ├── __init__.py          # Public API exports
│   ├── environment.py       # OpenEnv interface: reset(), step(), state()
│   ├── models.py            # Pydantic data models (Action, Observation, etc.)
│   ├── tasks.py             # Task definitions and initial state builders
│   ├── graders.py           # Dense reward engine + deterministic graders
│   ├── tools.py             # Tool implementations (6 logistics APIs)
│   └── utils.py             # Logging, loop detection, metrics, serialization
├── scripts/
│   └── run_inference.py     # Baseline LLM agent runner
├── configs/
│   └── config.yaml          # Tunable parameters (rewards, steps, model)
├── openenv.yaml             # OpenEnv specification manifest
├── requirements.txt         # Python dependencies
├── Dockerfile               # Production container
└── README.md                # This file
```

---

## Configuration

Edit `configs/config.yaml` to adjust:
- Task difficulty and step budgets
- Reward magnitudes (penalties and bonuses)
- Loop detection sensitivity
- Model and inference settings

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Citation

If you use LogisticsHub-360 in your research or evaluation work, please cite:

```bibtex
@misc{logisticshub360,
  title  = {LogisticsHub-360: Intelligent E-Commerce Operations Environment},
  year   = {2026},
  note   = {Submitted to Meta + Hugging Face OpenEnv Hackathon},
  url    = {https://github.com/your-org/logisticshub-360-openenv}
}
```
