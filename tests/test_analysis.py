"""Tests for analyze_conversation() and export_summary()."""

import pytest
from claude_context_manager import ContextManager, ConversationAnalysis


class TestAnalyzeConversation:
    def test_empty_conversation(self, cm):
        analysis = cm.analyze_conversation()
        assert analysis.total_messages == 0
        assert analysis.total_tokens == 0
        assert analysis.avg_message_length == 0
        assert analysis.user_assistant_ratio == 0
        assert analysis.message_role_distribution == {}

    def test_single_message(self, cm):
        cm.add_message("user", "Hello")
        a = cm.analyze_conversation()
        assert a.total_messages == 1
        assert a.user_messages == 1
        assert a.assistant_messages == 0
        assert a.user_assistant_ratio == 0  # no assistant → 0

    def test_balanced_conversation(self, populated_cm):
        a = populated_cm.analyze_conversation()
        assert a.total_messages == 5
        assert a.user_messages == 2
        assert a.assistant_messages == 2
        assert a.system_messages == 1
        assert a.user_assistant_ratio == 1.0

    def test_role_distribution(self, populated_cm):
        a = populated_cm.analyze_conversation()
        assert a.message_role_distribution["user"] == 2
        assert a.message_role_distribution["assistant"] == 2
        assert a.message_role_distribution["system"] == 1

    def test_token_stats(self, populated_cm):
        a = populated_cm.analyze_conversation()
        assert a.longest_message_tokens >= a.shortest_message_tokens
        assert a.avg_tokens_per_message > 0

    def test_avg_message_length(self, cm):
        cm.add_message("user", "12345")      # len 5
        cm.add_message("user", "1234567890")  # len 10
        a = cm.analyze_conversation()
        assert a.avg_message_length == 7.5

    def test_to_dict(self, populated_cm):
        a = populated_cm.analyze_conversation()
        d = a.to_dict()
        assert isinstance(d, dict)
        assert "total_messages" in d


class TestExportSummary:
    def test_export_keys(self, populated_cm):
        summary = populated_cm.export_summary()
        assert "model" in summary
        assert "message_count" in summary
        assert "total_tokens" in summary
        assert "analysis" in summary
        assert "messages" in summary
        assert "trim_history" in summary

    def test_message_previews_truncated(self, cm):
        cm.add_message("user", "A" * 200)
        summary = cm.export_summary()
        preview = summary["messages"][0]["preview"]
        assert len(preview) <= 103  # 100 + "..."

    def test_export_empty(self, cm):
        summary = cm.export_summary()
        assert summary["message_count"] == 0
        assert summary["messages"] == []
