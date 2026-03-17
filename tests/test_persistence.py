"""Tests for save/load and state serialisation."""

import json
import pytest
from claude_context_manager import ContextManager, TrimStrategy


class TestSaveLoad:
    def test_save_creates_file(self, populated_cm, tmp_json):
        populated_cm.save_to_file(tmp_json)
        with open(tmp_json) as f:
            data = json.load(f)
        assert "messages" in data
        assert "stats" in data

    def test_load_restores_messages(self, populated_cm, tmp_json):
        populated_cm.save_to_file(tmp_json)
        cm2 = ContextManager()
        cm2.load_from_file(tmp_json)
        assert len(cm2.messages) == len(populated_cm.messages)
        for orig, loaded in zip(populated_cm.messages, cm2.messages):
            assert orig.role == loaded.role
            assert orig.content == loaded.content

    def test_load_restores_stats(self, populated_cm, tmp_json):
        populated_cm.save_to_file(tmp_json)
        cm2 = ContextManager()
        cm2.load_from_file(tmp_json)
        assert cm2.stats.total_messages == populated_cm.stats.total_messages
        assert cm2.stats.total_tokens == populated_cm.stats.total_tokens

    def test_load_restores_model(self, tmp_json):
        cm = ContextManager(model="claude-3-opus")
        cm.add_message("user", "test")
        cm.save_to_file(tmp_json)
        cm2 = ContextManager()
        cm2.load_from_file(tmp_json)
        assert cm2.model == "claude-3-opus"

    def test_load_restores_trim_strategy(self, tmp_json):
        cm = ContextManager(trim_strategy=TrimStrategy.SLIDING_WINDOW)
        cm.add_message("user", "x")
        cm.save_to_file(tmp_json)
        cm2 = ContextManager()
        cm2.load_from_file(tmp_json)
        assert cm2.trim_strategy == TrimStrategy.SLIDING_WINDOW

    def test_roundtrip_metadata(self, tmp_json):
        cm = ContextManager()
        cm.add_message("user", "meta", metadata={"key": "value", "num": 42})
        cm.save_to_file(tmp_json)
        cm2 = ContextManager()
        cm2.load_from_file(tmp_json)
        assert cm2.messages[0].metadata == {"key": "value", "num": 42}

    def test_roundtrip_unicode(self, cm, unicode_message, tmp_json):
        cm.add_message("user", unicode_message)
        cm.save_to_file(tmp_json)
        cm2 = ContextManager()
        cm2.load_from_file(tmp_json)
        assert cm2.messages[0].content == unicode_message


class TestConversationState:
    def test_get_state_keys(self, populated_cm):
        state = populated_cm.get_conversation_state()
        assert "model" in state
        assert "messages" in state
        assert "stats" in state
        assert "trim_strategy" in state

    def test_load_state_dict(self, populated_cm):
        state = populated_cm.get_conversation_state()
        cm2 = ContextManager()
        cm2.load_conversation_state(state)
        assert len(cm2.messages) == len(populated_cm.messages)

    def test_load_empty_state(self):
        """Loading empty state raises TypeError (known limitation: ConversationStats requires fields)."""
        cm = ContextManager()
        with pytest.raises(TypeError):
            cm.load_conversation_state({})
