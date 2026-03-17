"""Tests for message creation, storage, and retrieval."""

import pytest
from claude_context_manager import ContextManager, Message


class TestAddMessage:
    """Adding messages and verifying internal state."""

    def test_add_user_message(self, cm):
        msg = cm.add_message("user", "Hello!")
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.token_count is not None and msg.token_count > 0

    def test_add_assistant_message(self, cm):
        msg = cm.add_message("assistant", "Hi there!")
        assert msg.role == "assistant"

    def test_add_system_message(self, cm):
        msg = cm.add_message("system", "You are helpful.")
        assert msg.role == "system"

    def test_message_count_increments(self, cm):
        cm.add_message("user", "one")
        cm.add_message("user", "two")
        assert cm.stats.total_messages == 2

    def test_token_total_updates(self, cm):
        cm.add_message("user", "first")
        t1 = cm.stats.total_tokens
        cm.add_message("user", "second message is longer than the first")
        assert cm.stats.total_tokens > t1

    def test_metadata_stored(self, cm):
        meta = {"topic": "testing", "priority": 1}
        msg = cm.add_message("user", "meta test", metadata=meta)
        assert msg.metadata == meta

    def test_default_metadata_empty(self, cm):
        msg = cm.add_message("user", "no meta")
        assert msg.metadata == {}

    def test_created_at_populated(self, cm):
        msg = cm.add_message("user", "timestamp")
        assert msg.created_at is not None
        assert "T" in msg.created_at  # ISO format

    def test_empty_content(self, cm):
        """Empty content should still work."""
        msg = cm.add_message("user", "")
        assert msg.content == ""
        assert msg.token_count >= 4  # overhead

    def test_unicode_content(self, cm, unicode_message):
        msg = cm.add_message("user", unicode_message)
        assert msg.content == unicode_message
        assert msg.token_count > 0


class TestMessageSerialization:
    """Message.to_dict() and to_serializable()."""

    def test_to_dict_api_format(self, cm):
        msg = cm.add_message("user", "Hi")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hi"}
        assert "metadata" not in d
        assert "token_count" not in d

    def test_to_serializable_complete(self, cm):
        msg = cm.add_message("user", "Hi", metadata={"k": "v"})
        s = msg.to_serializable()
        assert s["role"] == "user"
        assert s["content"] == "Hi"
        assert s["metadata"] == {"k": "v"}
        assert "token_count" in s
        assert "created_at" in s


class TestGetMessagesForAPI:
    """get_messages_for_api() returns clean API payloads."""

    def test_format(self, populated_cm):
        msgs = populated_cm.get_messages_for_api()
        assert isinstance(msgs, list)
        for m in msgs:
            assert set(m.keys()) == {"role", "content"}

    def test_order_preserved(self, cm):
        cm.add_message("system", "sys")
        cm.add_message("user", "u1")
        cm.add_message("assistant", "a1")
        roles = [m["role"] for m in cm.get_messages_for_api()]
        assert roles == ["system", "user", "assistant"]
