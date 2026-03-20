"""Tests for token counting accuracy."""

import pytest
from claude_context_manager import ContextManager


class TestTokenCounting:
    """Verify count_tokens behaviour across inputs."""

    def test_empty_string(self, cm):
        """Empty string should still have formatting overhead."""
        tokens = cm.count_tokens("")
        assert tokens == 4  # overhead only

    def test_short_string(self, cm):
        """Short text should return reasonable count."""
        tokens = cm.count_tokens("Hello")
        assert tokens > 4  # at least overhead + 1 token

    def test_longer_text(self, cm):
        """More text → more tokens, monotonically."""
        short = cm.count_tokens("Hi")
        long = cm.count_tokens("Hi " * 100)
        assert long > short

    def test_unicode_tokens(self, cm, unicode_message):
        """Unicode should be tokenisable without error."""
        tokens = cm.count_tokens(unicode_message)
        assert tokens > 4

    def test_long_message_tokens(self, cm, long_message):
        """Long messages should produce proportionally more tokens."""
        tokens = cm.count_tokens(long_message)
        assert tokens > 100

    def test_consistency(self, cm):
        """Same input → same output (deterministic)."""
        text = "Determinism is important for testing."
        assert cm.count_tokens(text) == cm.count_tokens(text)

    def test_whitespace_matters(self, cm):
        """Different whitespace should (potentially) change token count."""
        a = cm.count_tokens("hello world")
        b = cm.count_tokens("hello  world")
        # They may or may not differ, but both must be valid
        assert a > 0 and b > 0

    def test_special_characters(self, cm):
        """Special chars should be counted without errors."""
        tokens = cm.count_tokens("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert tokens > 4

    def test_newlines_and_tabs(self, cm):
        tokens = cm.count_tokens("line1\nline2\n\ttabbed")
        assert tokens > 4

    def test_model_fallback_encoding(self):
        """Unknown model should fall back to cl100k_base without crashing."""
        cm = ContextManager(model="claude-3-5-haiku")
        tokens = cm.count_tokens("Fallback encoding test")
        assert tokens > 4

    def test_role_overhead_applied(self, cm):
        """Different roles should produce different token counts due to overhead."""
        text = "Same content for all roles"
        system_tokens = cm.count_tokens(text, role="system")
        user_tokens = cm.count_tokens(text, role="user")
        assistant_tokens = cm.count_tokens(text, role="assistant")
        # system has overhead 6, user 4, assistant 2
        assert system_tokens > user_tokens
        assert user_tokens > assistant_tokens

    def test_add_message_uses_role_overhead(self):
        """add_message should pass role to count_tokens for accurate overhead."""
        cm = ContextManager()
        text = "Test message content"
        msg_user = cm.add_message("user", text)
        cm.clear()
        msg_assistant = cm.add_message("assistant", text)
        cm.clear()
        msg_system = cm.add_message("system", text)
        # system (6) > user (4) > assistant (2)
        assert msg_system.token_count > msg_user.token_count
        assert msg_user.token_count > msg_assistant.token_count


class TestClaude4Models:
    """Verify Claude 4 model support."""

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ])
    def test_claude4_initialization(self, model):
        cm = ContextManager(model=model)
        cm.add_message("user", "Hello Claude 4")
        assert cm.stats.total_tokens > 0

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ])
    def test_claude4_has_pricing(self, model):
        assert model in ContextManager.PRICING
        assert model in ContextManager.BATCH_PRICING
