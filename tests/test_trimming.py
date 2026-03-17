"""Tests for all trimming strategies."""

import pytest
from claude_context_manager import ContextManager, TrimStrategy


def _fill(cm, n=20, prefix="msg"):
    """Add n user+assistant pairs to push token count up."""
    for i in range(n):
        cm.add_message("user", f"{prefix} user message number {i} " * 5)
        cm.add_message("assistant", f"{prefix} assistant reply number {i} " * 5)


class TestShouldTrim:
    def test_below_threshold(self, cm):
        cm.add_message("user", "short")
        assert cm.should_trim() is False

    def test_above_threshold(self, cm_small):
        _fill(cm_small, n=10)
        assert cm_small.should_trim() is True


class TestOldestFirst:
    def test_removes_messages(self, cm_oldest):
        """Trim oldest non-system messages first."""
        cm_oldest.max_tokens = 500
        cm_oldest.trim_threshold = int(500 * 0.85)
        cm_oldest.add_message("system", "System prompt.")
        _fill(cm_oldest, n=10)
        removed_tokens, removed = cm_oldest.trim_conversation(target_tokens=200)
        assert len(removed) > 0
        assert removed_tokens > 0
        # System message should still be there
        assert any(m.role == "system" for m in cm_oldest.messages)

    def test_preserves_system(self, cm_oldest):
        """System messages are never trimmed by oldest-first."""
        cm_oldest.max_tokens = 500
        cm_oldest.trim_threshold = int(500 * 0.85)
        cm_oldest.add_message("system", "Keep me.")
        _fill(cm_oldest, n=10)
        cm_oldest.trim_conversation(target_tokens=200)
        system_msgs = [m for m in cm_oldest.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "Keep me."

    def test_no_trim_when_under_target(self, cm_oldest):
        cm_oldest.add_message("user", "tiny")
        removed_tokens, removed = cm_oldest.trim_conversation(target_tokens=999999)
        assert removed_tokens == 0
        assert removed == []


class TestSlidingWindow:
    def test_keeps_recent(self, cm_sliding):
        cm_sliding.add_message("system", "sys")
        for i in range(20):
            cm_sliding.add_message("user", f"old message {i} " * 10)
        cm_sliding.add_message("user", "LATEST")
        _, removed = cm_sliding.trim_conversation(target_tokens=100)
        contents = [m.content for m in cm_sliding.messages]
        assert "LATEST" in contents

    def test_preserves_system(self, cm_sliding):
        cm_sliding.add_message("system", "keep me")
        _fill(cm_sliding, n=20)
        cm_sliding.trim_conversation(target_tokens=100)
        assert any(m.role == "system" for m in cm_sliding.messages)

    def test_removes_oldest(self, cm_sliding):
        cm_sliding.add_message("user", "OLDEST_MARKER " * 20)
        _fill(cm_sliding, n=20)
        cm_sliding.trim_conversation(target_tokens=200)
        contents = " ".join(m.content for m in cm_sliding.messages)
        assert "OLDEST_MARKER" not in contents


class TestSmartTrim:
    def test_keeps_system_and_recent(self, cm_smart):
        cm_smart.add_message("system", "I am the system prompt.")
        for i in range(20):
            cm_smart.add_message("user", f"filler message {i} " * 10)
        cm_smart.add_message("user", "RECENT_MARKER")
        cm_smart.trim_conversation(target_tokens=200)
        contents = [m.content for m in cm_smart.messages]
        assert any("system prompt" in c for c in contents)
        assert "RECENT_MARKER" in contents

    def test_stats_updated_after_trim(self, cm_smart):
        _fill(cm_smart, n=20)
        cm_smart.trim_conversation(target_tokens=200)
        actual_tokens = sum(m.token_count or 0 for m in cm_smart.messages)
        assert cm_smart.stats.total_tokens == actual_tokens

    def test_trimmed_count_increases(self, cm_smart):
        _fill(cm_smart, n=20)
        cm_smart.trim_conversation(target_tokens=200)
        assert cm_smart.stats.trimmed_count > 0

    def test_last_trimmed_at_set(self, cm_smart):
        _fill(cm_smart, n=20)
        cm_smart.trim_conversation(target_tokens=200)
        assert cm_smart.stats.last_trimmed_at is not None


class TestTrimEdgeCases:
    def test_trim_empty_conversation(self, cm):
        removed_tokens, removed = cm.trim_conversation(target_tokens=100)
        assert removed_tokens == 0
        assert removed == []

    def test_trim_single_message(self, cm_smart):
        cm_smart.add_message("user", "only message")
        removed_tokens, removed = cm_smart.trim_conversation(target_tokens=999)
        assert removed_tokens == 0

    def test_trim_with_zero_target(self, cm_smart):
        """target_tokens=0 should remove everything except system."""
        cm_smart.add_message("system", "sys")
        _fill(cm_smart, n=5)
        cm_smart.trim_conversation(target_tokens=0)
        # System message may remain; content messages should be mostly gone
        assert len(cm_smart.messages) <= 2  # system + maybe 0-1 kept
