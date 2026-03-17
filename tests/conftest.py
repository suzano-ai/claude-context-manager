"""
Shared fixtures for claude-context-manager test suite.
"""

import pytest
import sys
import os

# Ensure the package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claude_context_manager import (
    ContextManager,
    Message,
    TrimStrategy,
    MessageRole,
    ConversationStats,
    ConversationAnalysis,
)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cm():
    """Default ContextManager (claude-3-5-sonnet, SMART trim)."""
    return ContextManager(model="claude-3-5-sonnet", verbose=False)


@pytest.fixture
def cm_oldest():
    """ContextManager with OLDEST_FIRST trimming."""
    return ContextManager(
        model="claude-3-5-sonnet",
        trim_strategy=TrimStrategy.OLDEST_FIRST,
        verbose=False,
    )


@pytest.fixture
def cm_sliding():
    """ContextManager with SLIDING_WINDOW trimming."""
    return ContextManager(
        model="claude-3-5-sonnet",
        trim_strategy=TrimStrategy.SLIDING_WINDOW,
        verbose=False,
    )


@pytest.fixture
def cm_smart():
    """ContextManager with SMART trimming."""
    return ContextManager(
        model="claude-3-5-sonnet",
        trim_strategy=TrimStrategy.SMART,
        verbose=False,
    )


@pytest.fixture
def cm_batch():
    """ContextManager configured for Batch API pricing."""
    return ContextManager(
        model="claude-3-5-sonnet",
        use_batch_api=True,
        verbose=False,
    )


@pytest.fixture
def cm_small():
    """ContextManager with a tiny max_tokens for easy trim testing."""
    return ContextManager(
        model="claude-3-5-sonnet",
        max_tokens=200,
        trim_strategy=TrimStrategy.SMART,
        trim_threshold=0.5,  # trigger at 100 tokens
        verbose=False,
    )


@pytest.fixture
def populated_cm(cm):
    """ContextManager pre-loaded with a realistic conversation."""
    cm.add_message("system", "You are a helpful assistant.")
    cm.add_message("user", "Hello! What can you help me with?")
    cm.add_message("assistant", "I can help with many things including coding, writing, and analysis.")
    cm.add_message("user", "Tell me about Python decorators.", metadata={"topic": "python"})
    cm.add_message(
        "assistant",
        "Decorators are a powerful feature in Python that allow you to modify "
        "the behavior of functions or classes. They use the @syntax and are "
        "essentially higher-order functions.",
        metadata={"topic": "python"},
    )
    return cm


@pytest.fixture
def long_message():
    """A deliberately long message string (~2000 words)."""
    return " ".join(["word"] * 2000)


@pytest.fixture
def unicode_message():
    """Message with diverse Unicode content."""
    return (
        "こんにちは世界 🌍 مرحبا بالعالم "
        "Привет мир 你好世界 "
        "café résumé naïve "
        "∑∏∫∂√∞ "
        "emoji: 🚀🔥💡🎉👾"
    )


@pytest.fixture
def tmp_json(tmp_path):
    """Temporary JSON file path for save/load tests."""
    return str(tmp_path / "conversation.json")
