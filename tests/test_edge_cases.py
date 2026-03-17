"""Edge case and integration tests."""

import pytest
from claude_context_manager import ContextManager, TrimStrategy, ConversationStats


class TestClearAndReset:
    def test_clear_empties_messages(self, populated_cm):
        populated_cm.clear()
        assert len(populated_cm.messages) == 0

    def test_clear_resets_stats(self, populated_cm):
        populated_cm.clear()
        assert populated_cm.stats.total_messages == 0
        assert populated_cm.stats.total_tokens == 0
        assert populated_cm.stats.estimated_cost == 0.0

    def test_clear_preserves_model(self, populated_cm):
        model = populated_cm.model
        populated_cm.clear()
        assert populated_cm.model == model


class TestRepr:
    def test_repr_format(self, cm):
        cm.add_message("user", "test")
        r = repr(cm)
        assert "ContextManager" in r
        assert "claude-3-5-sonnet" in r
        assert "messages=1" in r


class TestEnums:
    def test_trim_strategy_values(self):
        assert TrimStrategy.OLDEST_FIRST.value == "oldest_first"
        assert TrimStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert TrimStrategy.SMART.value == "smart"
        assert TrimStrategy.SUMMARIZE.value == "summarize"

    def test_trim_strategy_from_string(self):
        assert TrimStrategy("oldest_first") == TrimStrategy.OLDEST_FIRST


class TestConversationStatsDataclass:
    def test_to_dict(self):
        stats = ConversationStats(
            total_messages=5,
            total_tokens=100,
            estimated_cost=0.01,
            model="claude-3-5-sonnet",
        )
        d = stats.to_dict()
        assert d["total_messages"] == 5
        assert d["model"] == "claude-3-5-sonnet"


class TestComplexMetadata:
    def test_nested_metadata(self, cm):
        meta = {"tags": ["python", "ai"], "scores": {"relevance": 0.9}}
        msg = cm.add_message("user", "complex meta", metadata=meta)
        assert msg.metadata["tags"] == ["python", "ai"]
        assert msg.metadata["scores"]["relevance"] == 0.9

    def test_metadata_survives_serialization(self, cm, tmp_json):
        meta = {"nested": {"deep": {"value": 42}}}
        cm.add_message("user", "nested", metadata=meta)
        cm.save_to_file(tmp_json)
        cm2 = ContextManager()
        cm2.load_from_file(tmp_json)
        assert cm2.messages[0].metadata["nested"]["deep"]["value"] == 42


class TestMultipleModels:
    """Ensure different models work correctly."""

    @pytest.mark.parametrize("model", [
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
    ])
    def test_model_initialization(self, model):
        cm = ContextManager(model=model)
        cm.add_message("user", "Testing model init")
        assert cm.stats.total_tokens > 0

    @pytest.mark.parametrize("model", [
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
    ])
    def test_model_has_pricing(self, model):
        assert model in ContextManager.PRICING
        assert model in ContextManager.BATCH_PRICING


class TestVerboseMode:
    def test_verbose_does_not_crash(self, capsys):
        cm = ContextManager(verbose=True)
        cm.add_message("user", "verbose test")
        captured = capsys.readouterr()
        assert "Added user message" in captured.out


class TestLargeConversation:
    def test_100_messages(self, cm):
        for i in range(100):
            role = "user" if i % 2 == 0 else "assistant"
            cm.add_message(role, f"Message number {i}")
        assert cm.stats.total_messages == 100
        assert cm.stats.total_tokens > 0
        analysis = cm.analyze_conversation()
        assert analysis.user_messages == 50
        assert analysis.assistant_messages == 50


class TestBatchApiFlag:
    def test_default_is_standard(self, cm):
        assert cm.use_batch_api is False

    def test_batch_flag_set(self, cm_batch):
        assert cm_batch.use_batch_api is True


class TestTrimThreshold:
    def test_custom_threshold(self):
        cm = ContextManager(max_tokens=1000, trim_threshold=0.5)
        assert cm.trim_threshold == 500

    def test_default_threshold(self):
        cm = ContextManager(max_tokens=1000)
        assert cm.trim_threshold == 850  # 0.85 * 1000
