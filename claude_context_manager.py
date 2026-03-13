"""
Claude Context Manager - Production-ready conversation management for Claude API.

Solves:
- Token counting accuracy for Claude messages
- Automatic conversation trimming with smart strategies
- Cost tracking and estimation
- Conversation persistence and recovery
- Intelligent context preservation
"""

import json
import tiktoken
from typing import List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
import hashlib


class TrimStrategy(str, Enum):
    """Conversation trimming strategies."""
    OLDEST_FIRST = "oldest_first"  # Remove oldest messages first
    SUMMARIZE = "summarize"  # Mark old messages for summarization
    SLIDING_WINDOW = "sliding_window"  # Keep only recent N messages
    SMART = "smart"  # Remove oldest non-critical messages, keep system prompt


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Single message in conversation."""
    role: str
    content: str
    metadata: Dict = field(default_factory=dict)
    token_count: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        """Convert to API-compatible dict."""
        return {
            "role": self.role,
            "content": self.content,
        }

    def to_serializable(self) -> Dict:
        """Convert to JSON-serializable format."""
        return asdict(self)


@dataclass
class ConversationStats:
    """Statistics about conversation state."""
    total_messages: int
    total_tokens: int
    estimated_cost: float
    model: str
    last_trimmed_at: Optional[str] = None
    trimmed_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dict."""
        return asdict(self)


class ContextManager:
    """
    Production-ready context manager for Claude conversations.
    
    Handles:
    - Token counting with Claude-specific logic
    - Conversation trimming with configurable strategies
    - Cost estimation and tracking
    - Message persistence and recovery
    """
    
    # Claude API pricing (updates needed for newer models)
    PRICING = {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    }
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        max_tokens: int = 200000,
        trim_strategy: TrimStrategy = TrimStrategy.SMART,
        trim_threshold: float = 0.85,
        verbose: bool = False,
    ):
        """
        Initialize context manager.
        
        Args:
            model: Claude model to use for token calculations
            max_tokens: Maximum context window size (default: 200k for Sonnet)
            trim_strategy: How to trim when approaching limit
            trim_threshold: Trim when reaching this % of max_tokens (0-1.0)
            verbose: Print debug info
        """
        self.model = model
        self.max_tokens = max_tokens
        self.trim_strategy = trim_strategy
        self.trim_threshold = int(max_tokens * trim_threshold)
        self.verbose = verbose
        
        # Initialize tokenizer for this model
        try:
            self.encoder = tiktoken.encoding_for_model(
                self._map_model_to_tiktoken(model)
            )
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        self.messages: List[Message] = []
        self.stats = ConversationStats(
            total_messages=0,
            total_tokens=0,
            estimated_cost=0.0,
            model=model,
        )
    
    @staticmethod
    def _map_model_to_tiktoken(model: str) -> str:
        """Map Claude model name to tiktoken equivalent."""
        mapping = {
            "claude-3-opus": "gpt-4",
            "claude-3-sonnet": "gpt-3.5-turbo",
            "claude-3-haiku": "gpt-3.5-turbo",
            "claude-3-5-sonnet": "gpt-4-turbo",
        }
        return mapping.get(model, "gpt-3.5-turbo")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text with Claude-appropriate logic.
        
        Claude's tokenization is slightly different from GPT models.
        This includes overhead for message formatting.
        """
        # Count base tokens
        tokens = len(self.encoder.encode(text))
        
        # Add ~4 tokens per message for role/formatting overhead
        # This matches Claude's actual behavior more closely
        tokens += 4
        
        return tokens
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> Message:
        """
        Add message to conversation with token counting.
        
        Args:
            role: "user", "assistant", or "system"
            content: Message content
            metadata: Optional metadata (tags, importance, etc.)
        
        Returns:
            Created Message object
        """
        token_count = self.count_tokens(content)
        
        message = Message(
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )
        
        self.messages.append(message)
        self.stats.total_messages += 1
        self.stats.total_tokens += token_count
        self._update_cost()
        
        if self.verbose:
            print(f"[+] Added {role} message: {token_count} tokens")
        
        return message
    
    def _update_cost(self) -> None:
        """Estimate conversation cost based on current tokens."""
        pricing = self.PRICING.get(self.model)
        if pricing:
            # Rough estimate: assume 50/50 input/output split
            avg_tokens = self.stats.total_tokens
            self.stats.estimated_cost = (
                (avg_tokens * 0.5 * pricing["input"]) +
                (avg_tokens * 0.5 * pricing["output"])
            ) / 1000
    
    def should_trim(self) -> bool:
        """Check if conversation exceeds trim threshold."""
        return self.stats.total_tokens > self.trim_threshold
    
    def trim_conversation(self, target_tokens: Optional[int] = None) -> Tuple[int, List[Message]]:
        """
        Trim conversation to fit within limits.
        
        Args:
            target_tokens: Target token count (default: 60% of max)
        
        Returns:
            Tuple of (tokens_removed, removed_messages)
        """
        if target_tokens is None:
            target_tokens = int(self.max_tokens * 0.6)
        
        if self.stats.total_tokens <= target_tokens:
            if self.verbose:
                print("[-] No trimming needed")
            return 0, []
        
        removed = []
        
        if self.trim_strategy == TrimStrategy.OLDEST_FIRST:
            removed = self._trim_oldest_first(target_tokens)
        elif self.trim_strategy == TrimStrategy.SLIDING_WINDOW:
            removed = self._trim_sliding_window(target_tokens)
        elif self.trim_strategy == TrimStrategy.SMART:
            removed = self._trim_smart(target_tokens)
        
        # Recalculate stats
        self.stats.total_tokens = sum(m.token_count or 0 for m in self.messages)
        self.stats.trimmed_count += len(removed)
        self.stats.last_trimmed_at = datetime.utcnow().isoformat()
        self._update_cost()
        
        if self.verbose:
            total_removed = sum(m.token_count or 0 for m in removed)
            print(f"[*] Trimmed {len(removed)} messages ({total_removed} tokens)")
        
        return sum(m.token_count or 0 for m in removed), removed
    
    def _trim_oldest_first(self, target_tokens: int) -> List[Message]:
        """Remove oldest messages first."""
        removed = []
        current_tokens = self.stats.total_tokens
        
        # Never remove system messages or most recent message
        for i, msg in enumerate(self.messages[:-1]):
            if msg.role == "system" or current_tokens <= target_tokens:
                continue
            
            current_tokens -= (msg.token_count or 0)
            self.messages.pop(i)
            removed.append(msg)
        
        return removed
    
    def _trim_sliding_window(self, target_tokens: int) -> List[Message]:
        """Keep only recent messages within token limit."""
        removed = []
        current_tokens = 0
        kept = []
        
        # Work backwards from most recent
        for msg in reversed(self.messages):
            msg_tokens = msg.token_count or 0
            if current_tokens + msg_tokens <= target_tokens or msg.role == "system":
                kept.append(msg)
                current_tokens += msg_tokens
            else:
                removed.append(msg)
        
        kept.reverse()
        removed.reverse()
        
        # Keep system messages always
        system_msgs = [m for m in self.messages if m.role == "system"]
        if system_msgs and system_msgs[0] not in kept:
            kept = system_msgs + kept
            removed = [m for m in removed if m not in system_msgs]
        
        self.messages = kept
        return removed
    
    def _trim_smart(self, target_tokens: int) -> List[Message]:
        """
        Smart trimming: preserve system messages and recent context,
        remove oldest non-critical messages.
        """
        removed = []
        system_msgs = [m for m in self.messages if m.role == "system"]
        content_msgs = [m for m in self.messages if m.role != "system"]
        
        # Always keep system messages
        kept = system_msgs[:]
        current_tokens = sum(m.token_count or 0 for m in kept)
        
        # Add messages from most recent backwards
        for msg in reversed(content_msgs):
            msg_tokens = msg.token_count or 0
            if current_tokens + msg_tokens <= target_tokens:
                kept.insert(len(system_msgs), msg)  # Insert after system messages
                current_tokens += msg_tokens
            else:
                removed.append(msg)
        
        removed.reverse()
        self.messages = kept
        return removed
    
    def get_messages_for_api(self) -> List[Dict]:
        """Get messages in format ready for Claude API."""
        return [msg.to_dict() for msg in self.messages]
    
    def get_conversation_state(self) -> Dict:
        """Get full conversation state for persistence."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "trim_strategy": self.trim_strategy.value,
            "trim_threshold": self.trim_threshold,
            "messages": [msg.to_serializable() for msg in self.messages],
            "stats": self.stats.to_dict(),
        }
    
    def load_conversation_state(self, state: Dict) -> None:
        """Restore conversation from saved state."""
        self.model = state.get("model", self.model)
        self.max_tokens = state.get("max_tokens", self.max_tokens)
        self.trim_strategy = TrimStrategy(state.get("trim_strategy", self.trim_strategy.value))
        self.trim_threshold = state.get("trim_threshold", self.trim_threshold)
        
        # Restore messages
        self.messages = []
        for msg_data in state.get("messages", []):
            msg = Message(**msg_data)
            self.messages.append(msg)
        
        # Restore stats
        stats_data = state.get("stats", {})
        self.stats = ConversationStats(**stats_data)
    
    def save_to_file(self, filepath: str) -> None:
        """Save conversation to JSON file."""
        state = self.get_conversation_state()
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load conversation from JSON file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.load_conversation_state(state)
    
    def get_stats(self) -> ConversationStats:
        """Get current conversation statistics."""
        return self.stats
    
    def clear(self) -> None:
        """Clear conversation."""
        self.messages = []
        self.stats = ConversationStats(
            total_messages=0,
            total_tokens=0,
            estimated_cost=0.0,
            model=self.model,
        )
    
    def __repr__(self) -> str:
        return (
            f"ContextManager(model={self.model}, messages={self.stats.total_messages}, "
            f"tokens={self.stats.total_tokens}, cost=${self.stats.estimated_cost:.4f})"
        )
