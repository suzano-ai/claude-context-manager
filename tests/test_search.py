"""Tests for search_messages()."""

import pytest
from claude_context_manager import ContextManager


class TestSearchMessages:
    def test_search_by_role(self, populated_cm):
        users = populated_cm.search_messages(role="user")
        assert all(m.role == "user" for m in users)
        assert len(users) == 2

    def test_search_by_content(self, populated_cm):
        results = populated_cm.search_messages(content_contains="decorator")
        assert len(results) >= 1
        assert any("decorator" in m.content.lower() for m in results)

    def test_search_case_insensitive(self, populated_cm):
        results = populated_cm.search_messages(content_contains="DECORATOR")
        assert len(results) >= 1

    def test_search_by_metadata(self, populated_cm):
        results = populated_cm.search_messages(metadata_filter={"topic": "python"})
        assert len(results) == 2
        assert all(m.metadata.get("topic") == "python" for m in results)

    def test_search_by_created_after(self, cm):
        cm.add_message("user", "old message")
        # All messages created "now" should pass a filter from the past
        results = cm.search_messages(created_after="2000-01-01T00:00:00")
        assert len(results) == 1

    def test_search_combined_filters(self, populated_cm):
        results = populated_cm.search_messages(
            role="user",
            content_contains="decorator",
        )
        assert len(results) == 1
        assert results[0].role == "user"

    def test_search_no_results(self, populated_cm):
        results = populated_cm.search_messages(content_contains="xyznonexistent")
        assert results == []

    def test_search_empty_conversation(self, cm):
        results = cm.search_messages(role="user")
        assert results == []

    def test_search_metadata_partial_match(self, cm):
        """Only exact key-value match should count."""
        cm.add_message("user", "a", metadata={"x": 1, "y": 2})
        cm.add_message("user", "b", metadata={"x": 1})
        results = cm.search_messages(metadata_filter={"x": 1, "y": 2})
        assert len(results) == 1

    def test_search_all_roles(self, populated_cm):
        for role in ["user", "assistant", "system"]:
            results = populated_cm.search_messages(role=role)
            assert all(m.role == role for m in results)
