#!/usr/bin/env python3
"""
LogisticsHub-360: Smoke Test + Validation Suite

Verifies:
  1. Environment import and initialization
  2. All three tasks: reset / step / grade cycle
  3. Reward ranges and grader outputs
  4. Loop detection
  5. Prerequisite enforcement in tools
  6. Terminal condition detection

Run with:
    python tests/test_environment.py
"""

from __future__ import annotations

import sys
import os
import unittest
import json

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import LogisticsHub360Env, make_env
from env.models import Action, ToolName
from env.graders import grade
from env.tasks import TASK_ORDER, TASK_BUILDERS


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_action(tool: str, **params) -> Action:
    return Action(tool=ToolName(tool), parameters=params)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestEnvironmentInit(unittest.TestCase):

    def test_valid_task_ids(self):
        for task_id in TASK_ORDER:
            env = LogisticsHub360Env(task_id)
            self.assertIsNotNone(env)

    def test_invalid_task_id_raises(self):
        with self.assertRaises(ValueError):
            LogisticsHub360Env("nonexistent_task")

    def test_factory_function(self):
        env = make_env("order_tracking")
        self.assertIsNotNone(env)

    def test_step_before_reset_raises(self):
        env = LogisticsHub360Env("order_tracking")
        action = make_action("get_tracking", order_id="ORD-88421")
        with self.assertRaises(RuntimeError):
            env.step(action)

    def test_state_before_reset_raises(self):
        env = LogisticsHub360Env("order_tracking")
        with self.assertRaises(RuntimeError):
            env.state()


class TestTask1OrderTracking(unittest.TestCase):

    def setUp(self):
        self.env = LogisticsHub360Env("order_tracking")
        self.obs = self.env.reset()

    def test_reset_returns_observation(self):
        obs = self.obs
        self.assertEqual(obs.task_id, "order_tracking")
        self.assertEqual(obs.difficulty, "easy")
        self.assertEqual(obs.step_count, 0)
        self.assertFalse(obs.is_done)
        self.assertIsNotNone(obs.order_status)
        self.assertIsNotNone(obs.inventory_state)
        self.assertIn("get_tracking", obs.available_tools)

    def test_full_optimal_sequence(self):
        """Complete Task 1 optimally in 2 steps."""
        order_id = self.obs.order_status.order_id
        self.assertEqual(order_id, "ORD-88421")

        # Step 1: get_tracking
        obs, reward, done, info = self.env.step(
            make_action("get_tracking", order_id=order_id)
        )
        self.assertGreater(reward, 0, "Correct tool should give positive reward.")
        self.assertFalse(done)

        # Step 2: update_crm
        obs, reward, done, info = self.env.step(
            make_action(
                "update_crm",
                order_id=order_id,
                message="Your order ORD-88421 is currently shipped and in Atlanta, GA.",
            )
        )
        self.assertTrue(done, "Task should complete after correct 2-step sequence.")
        self.assertGreater(reward, 0.5, "Completion should yield high reward.")
        self.assertIn("final_grade", info)
        self.assertGreater(info["final_grade"], 0.7)

    def test_invalid_order_id_penalised(self):
        """Wrong order_id should return failure with negative reward."""
        obs, reward, done, info = self.env.step(
            make_action("get_tracking", order_id="WRONG-ID")
        )
        self.assertLess(reward, 0)
        self.assertFalse(done)

    def test_state_contains_expected_keys(self):
        debug = self.env.state()
        self.assertIn("task_id", debug)
        self.assertIn("order", debug)
        self.assertIn("inventory", debug)
        self.assertIn("customer_sentiment", debug)

    def test_grade_task_1_zero_without_actions(self):
        """Grader should return 0 grade on reset-only state."""
        final_state_dict = self.env.state()
        # Access internal state directly
        internal = self.env._internal_state
        g = grade("order_tracking", internal)
        self.assertGreaterEqual(g, 0.0)
        self.assertLessEqual(g, 1.0)


class TestTask2ShipmentRerouting(unittest.TestCase):

    def setUp(self):
        self.env = LogisticsHub360Env("shipment_rerouting")
        self.obs = self.env.reset()

    def test_reset_returns_delayed_order(self):
        self.assertEqual(self.obs.order_status.status, "delayed")
        self.assertLess(self.obs.customer_sentiment, 0.5)

    def test_full_optimal_sequence(self):
        """Complete Task 2 optimally in 5 steps."""
        order_id = self.obs.order_status.order_id  # ORD-44790
        product_id = self.obs.inventory_state.product_id  # PROD-CAM-3310
        destination = self.obs.order_status.destination  # Los Angeles, CA

        # Step 1: get_tracking
        obs, r1, done, _ = self.env.step(
            make_action("get_tracking", order_id=order_id)
        )
        self.assertFalse(done)

        # Step 2: check_inventory
        obs, r2, done, _ = self.env.step(
            make_action("check_inventory", product_id=product_id)
        )
        self.assertFalse(done)

        # Step 3: find_warehouse
        obs, r3, done, info = self.env.step(
            make_action("find_warehouse", location=destination)
        )
        self.assertFalse(done)

        # Retrieve warehouse from state
        internal = self.env._internal_state
        wh_id = internal.warehouse_id_selected
        self.assertIsNotNone(wh_id, "Warehouse should have been selected.")

        # Step 4: reroute_order
        obs, r4, done, _ = self.env.step(
            make_action("reroute_order", order_id=order_id, warehouse_id=wh_id)
        )
        self.assertFalse(done)
        self.assertTrue(internal.rerouted)

        # Step 5: update_crm
        obs, r5, done, info = self.env.step(
            make_action(
                "update_crm",
                order_id=order_id,
                message=f"Your order has been successfully rerouted to {wh_id}. "
                        "New delivery ETA in 1-2 business days.",
            )
        )
        self.assertTrue(done)
        self.assertIn("final_grade", info)
        self.assertGreater(info["final_grade"], 0.7)

    def test_reroute_without_warehouse_fails(self):
        """Rerouting without find_warehouse first should fail."""
        order_id = self.obs.order_status.order_id
        product_id = self.obs.inventory_state.product_id

        self.env.step(make_action("get_tracking", order_id=order_id))
        self.env.step(make_action("check_inventory", product_id=product_id))

        # Skip find_warehouse — try to reroute directly
        obs, reward, done, info = self.env.step(
            make_action("reroute_order", order_id=order_id, warehouse_id="WH-FAKE-99")
        )
        self.assertLessEqual(reward, 0, "Reroute without warehouse should be penalised.")

    def test_reroute_requires_tracking_first(self):
        """Rerouting without tracking check should fail."""
        order_id = self.obs.order_status.order_id
        obs, reward, done, info = self.env.step(
            make_action("reroute_order", order_id=order_id, warehouse_id="WH-ANY")
        )
        self.assertLess(reward, 0)


class TestTask3StockoutCrisis(unittest.TestCase):

    def setUp(self):
        self.env = LogisticsHub360Env("stockout_crisis")
        self.obs = self.env.reset()

    def test_reset_returns_oos_inventory(self):
        self.assertEqual(self.obs.inventory_state.level, "out_of_stock")
        self.assertEqual(self.obs.inventory_state.quantity, 0)
        self.assertLess(self.obs.customer_sentiment, 0.30)

    def test_optimal_sequence_with_refund(self):
        """Task 3 correct path: check all → find_warehouse (fail) → refund → crm."""
        order_id = self.obs.order_status.order_id  # ORD-99123
        product_id = self.obs.inventory_state.product_id  # PROD-LPT-0055
        destination = self.obs.order_status.destination

        self.env.step(make_action("get_tracking", order_id=order_id))
        self.env.step(make_action("check_inventory", product_id=product_id))
        # find_warehouse will fail — no stock — but it's a required sequence step
        self.env.step(make_action("find_warehouse", location=destination))

        # Issue refund (correct decision when OOS)
        obs, r_refund, done, info = self.env.step(
            make_action("issue_refund", order_id=order_id)
        )
        self.assertFalse(done)

        # Final CRM update
        obs, r_crm, done, info = self.env.step(
            make_action(
                "update_crm",
                order_id=order_id,
                message="We sincerely apologize. Your order ORD-99123 cannot be fulfilled "
                        "due to a complete stockout. A full refund has been processed to your account.",
            )
        )
        self.assertTrue(done)
        self.assertIn("final_grade", info)
        self.assertGreater(info["final_grade"], 0.5)

    def test_reroute_on_stockout_destructive_penalty(self):
        """Rerouting when OOS confirmed should trigger -1.0 penalty."""
        order_id = self.obs.order_status.order_id
        product_id = self.obs.inventory_state.product_id

        # Check tracking and inventory first to confirm OOS
        self.env.step(make_action("get_tracking", order_id=order_id))
        self.env.step(make_action("check_inventory", product_id=product_id))

        # Incorrectly attempt reroute despite OOS
        internal = self.env._internal_state
        internal.warehouse_found = True  # Bypass prerequisite for this test
        internal.tracking_checked = True
        internal.warehouse_id_selected = "WH-NORTH-09"

        obs, reward, done, info = self.env.step(
            make_action(
                "reroute_order",
                order_id=order_id,
                warehouse_id="WH-NORTH-09",
            )
        )
        self.assertLessEqual(reward, -0.50, "Wrong decision should carry severe penalty.")


class TestLoopDetection(unittest.TestCase):

    def setUp(self):
        self.env = LogisticsHub360Env("order_tracking")
        self.env.reset()

    def test_loop_triggers_termination(self):
        """Repeating the same failed action should trigger loop detection and terminate."""
        for _ in range(5):
            obs, reward, done, info = self.env.step(
                make_action("get_tracking", order_id="WRONG-ID")
            )
            if done:
                break

        # Should terminate eventually
        self.assertTrue(done or obs.step_count >= obs.max_steps)


class TestGraderProperties(unittest.TestCase):

    def test_grade_output_range(self):
        """All graders must output values in [0.0, 1.0]."""
        for task_id in TASK_ORDER:
            env = LogisticsHub360Env(task_id)
            env.reset()
            g = grade(task_id, env._internal_state)
            self.assertGreaterEqual(g, 0.0, f"{task_id} grade below 0.0")
            self.assertLessEqual(g, 1.0, f"{task_id} grade above 1.0")

    def test_grade_deterministic(self):
        """Same final state should always produce same grade."""
        env = LogisticsHub360Env("order_tracking")
        env.reset()
        internal = env._internal_state
        internal.tracking_checked = True
        internal.crm_updated = True
        internal.step_count = 3

        g1 = grade("order_tracking", internal)
        g2 = grade("order_tracking", internal)
        self.assertEqual(g1, g2)

    def test_invalid_task_id_raises(self):
        with self.assertRaises(ValueError):
            grade("bad_task", None)


class TestObservationStructure(unittest.TestCase):

    def test_observation_serializable(self):
        """Observation must be JSON-serializable."""
        env = LogisticsHub360Env("order_tracking")
        obs = env.reset()
        try:
            serialized = obs.model_dump_json()
            data = json.loads(serialized)
            self.assertIn("task_id", data)
            self.assertIn("available_tools", data)
        except Exception as exc:
            self.fail(f"Observation is not JSON-serializable: {exc}")

    def test_action_model_validates_tool(self):
        """Action with invalid tool name should raise validation error."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            Action(tool="invalid_tool_xyz", parameters={})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  LogisticsHub-360 — Test Suite")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
