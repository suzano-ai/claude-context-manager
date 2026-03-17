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
