"""Tests for cost calculation and pricing comparison."""

import pytest
from claude_context_manager import ContextManager


class TestCostEstimation:
    def test_cost_starts_zero(self, cm):
        assert cm.stats.estimated_cost == 0.0

    def test_cost_increases_with_messages(self, cm):
        cm.add_message("user", "Hello " * 50)
        assert cm.stats.estimated_cost > 0

    def test_cost_monotonically_increases(self, cm):
        cm.add_message("user", "first")
        c1 = cm.stats.estimated_cost
        cm.add_message("user", "second much longer message with lots of tokens " * 10)
        assert cm.stats.estimated_cost > c1

    def test_batch_cheaper_than_standard(self):
        std = ContextManager(model="claude-3-5-sonnet", use_batch_api=False)
        batch = ContextManager(model="claude-3-5-sonnet", use_batch_api=True)
        text = "Pricing comparison message. " * 100
        std.add_message("user", text)
        batch.add_message("user", text)
        assert batch.stats.estimated_cost < std.stats.estimated_cost

    def test_cost_resets_on_clear(self, cm):
        cm.add_message("user", "something")
        cm.clear()
        assert cm.stats.estimated_cost == 0.0

    def test_unknown_model_cost_zero(self):
        cm = ContextManager(model="claude-99-future")
        cm.add_message("user", "test")
        # Unknown model has no pricing entry → cost stays 0
        assert cm.stats.estimated_cost == 0.0


class TestComparePricing:
    def test_compare_returns_keys(self, cm):
        result = cm.compare_pricing(tokens=10000)
        assert "standard_api" in result
        assert "batch_api" in result
        assert "savings" in result

    def test_savings_positive(self, cm):
        result = cm.compare_pricing(tokens=100000)
        assert result["savings"]["amount"] > 0
        assert result["savings"]["percent"] > 0

    def test_savings_is_50_percent(self, cm):
        """Batch API is 50% discount."""
        result = cm.compare_pricing(tokens=100000)
        assert abs(result["savings"]["percent"] - 50.0) < 0.1

    def test_compare_default_uses_current_tokens(self, cm):
        cm.add_message("user", "hello " * 50)
        result = cm.compare_pricing()
        assert result["tokens"] == cm.stats.total_tokens

    def test_compare_unknown_model(self):
        cm = ContextManager(model="claude-99-future")
        result = cm.compare_pricing(tokens=1000)
        assert "error" in result

    def test_compare_zero_tokens(self, cm):
        result = cm.compare_pricing(tokens=0)
        assert result["standard_api"]["cost"] == 0
        assert result["batch_api"]["cost"] == 0

    def test_model_field_present(self, cm):
        result = cm.compare_pricing(tokens=1000)
        assert result["model"] == "claude-3-5-sonnet"

    def test_break_even_hours(self, cm):
        result = cm.compare_pricing(tokens=100000)
        assert result["break_even_hours"] == 24
