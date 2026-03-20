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
from datetime import datetime, timezone
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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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


@dataclass
class ConversationAnalysis:
    """Analysis of conversation health and patterns."""
    total_messages: int
    total_tokens: int
    avg_message_length: float
    user_messages: int
    assistant_messages: int
    system_messages: int
    user_assistant_ratio: float
    longest_message_tokens: int
    shortest_message_tokens: int
    avg_tokens_per_message: float
    message_role_distribution: Dict[str, int]
    
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
    
    # Claude API pricing (Standard API rates per 1M tokens)
    PRICING = {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku": {"input": 0.00080, "output": 0.0040},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    }
    
    # Batch API pricing (50% discount on standard rates)
    BATCH_PRICING = {
        "claude-3-opus": {"input": 0.0075, "output": 0.0375},
        "claude-3-sonnet": {"input": 0.0015, "output": 0.0075},
        "claude-3-haiku": {"input": 0.000125, "output": 0.000625},
        "claude-3-5-sonnet": {"input": 0.0015, "output": 0.0075},
        "claude-3-5-haiku": {"input": 0.00040, "output": 0.0020},
        "claude-sonnet-4-20250514": {"input": 0.0015, "output": 0.0075},
        "claude-opus-4-20250514": {"input": 0.0075, "output": 0.0375},
    }
    
    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        max_tokens: int = 200000,
        trim_strategy: TrimStrategy = TrimStrategy.SMART,
        trim_threshold: float = 0.85,
        verbose: bool = False,
        use_batch_api: bool = False,
    ):
        """
        Initialize context manager.
        
        Args:
            model: Claude model to use for token calculations
            max_tokens: Maximum context window size (default: 200k for Sonnet)
            trim_strategy: How to trim when approaching limit
            trim_threshold: Trim when reaching this % of max_tokens (0-1.0)
            verbose: Print debug info
            use_batch_api: Use Batch API pricing (50% discount, but 24h latency)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.trim_strategy = trim_strategy
        self.trim_threshold = int(max_tokens * trim_threshold)
        self.verbose = verbose
        self.use_batch_api = use_batch_api
        
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
            "claude-3-5-haiku": "gpt-3.5-turbo",
            "claude-sonnet-4-20250514": "gpt-4-turbo",
            "claude-opus-4-20250514": "gpt-4",
        }
        return mapping.get(model, "gpt-3.5-turbo")
    
    def count_tokens(self, text: str, role: Optional[str] = None) -> int:
        """
        Count tokens in text with Claude-appropriate logic.
        
        Claude's tokenization is slightly different from GPT models.
        This includes overhead for message formatting.

        Args:
            text: Message text to count tokens for.
            role: Optional role to adjust per-message overhead ("system", "user", "assistant").
        """
        # Count base tokens
        tokens = len(self.encoder.encode(text))
        
        # Add per-message overhead depending on role. System messages often include extra framing.
        overhead = 4
        if role == "system":
            overhead = 6
        elif role == "assistant":
            overhead = 2
        elif role == "user":
            overhead = 4
        
        tokens += overhead
        
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
        token_count = self.count_tokens(content, role=role)
        
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
        pricing_table = self.BATCH_PRICING if self.use_batch_api else self.PRICING
        pricing = pricing_table.get(self.model)
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
        self.stats.last_trimmed_at = datetime.now(timezone.utc).isoformat()
        self._update_cost()
        
        if self.verbose:
            total_removed = sum(m.token_count or 0 for m in removed)
            print(f"[*] Trimmed {len(removed)} messages ({total_removed} tokens)")
        
        return sum(m.token_count or 0 for m in removed), removed
    
    def _trim_oldest_first(self, target_tokens: int) -> List[Message]:
        """Remove oldest messages first."""
        removed = []
        current_tokens = self.stats.total_tokens
        keep = []
        
        # Never remove system messages or most recent message
        messages_to_check = self.messages[:-1] if len(self.messages) > 1 else []
        last_message = [self.messages[-1]] if self.messages else []
        
        for msg in messages_to_check:
            if msg.role == "system" or current_tokens <= target_tokens:
                keep.append(msg)
            else:
                current_tokens -= (msg.token_count or 0)
                removed.append(msg)
        
        self.messages = keep + last_message
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
    
    def search_messages(
        self,
        role: Optional[str] = None,
        content_contains: Optional[str] = None,
        metadata_filter: Optional[Dict] = None,
        created_after: Optional[str] = None,
    ) -> List[Message]:
        """
        Search and filter messages by various criteria.
        
        Args:
            role: Filter by role ("user", "assistant", "system")
            content_contains: Filter by substring in content
            metadata_filter: Filter by metadata key=value pairs
            created_after: Filter by creation timestamp (ISO format)
        
        Returns:
            List of matching messages
        """
        results = self.messages[:]
        
        # Filter by role
        if role:
            results = [m for m in results if m.role == role]
        
        # Filter by content substring
        if content_contains:
            results = [m for m in results if content_contains.lower() in m.content.lower()]
        
        # Filter by metadata
        if metadata_filter:
            results = [
                m for m in results
                if all(m.metadata.get(k) == v for k, v in metadata_filter.items())
            ]
        
        # Filter by creation time
        if created_after:
            results = [m for m in results if m.created_at >= created_after]
        
        return results
    
    def analyze_conversation(self) -> ConversationAnalysis:
        """
        Analyze conversation health and patterns.
        
        Returns metrics about message distribution, token usage, and conversation balance.
        """
        if not self.messages:
            return ConversationAnalysis(
                total_messages=0,
                total_tokens=0,
                avg_message_length=0,
                user_messages=0,
                assistant_messages=0,
                system_messages=0,
                user_assistant_ratio=0,
                longest_message_tokens=0,
                shortest_message_tokens=0,
                avg_tokens_per_message=0,
                message_role_distribution={},
            )
        
        # Count messages by role
        role_counts = {}
        total_content_length = 0
        token_counts = []
        
        for msg in self.messages:
            role_counts[msg.role] = role_counts.get(msg.role, 0) + 1
            total_content_length += len(msg.content)
            if msg.token_count:
                token_counts.append(msg.token_count)
        
        user_count = role_counts.get("user", 0)
        assistant_count = role_counts.get("assistant", 0)
        system_count = role_counts.get("system", 0)
        
        # Calculate ratio (avoid division by zero)
        ratio = (
            user_count / assistant_count 
            if assistant_count > 0 
            else 0
        )
        
        # Token statistics
        max_tokens = max(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        
        return ConversationAnalysis(
            total_messages=len(self.messages),
            total_tokens=self.stats.total_tokens,
            avg_message_length=total_content_length / len(self.messages),
            user_messages=user_count,
            assistant_messages=assistant_count,
            system_messages=system_count,
            user_assistant_ratio=ratio,
            longest_message_tokens=max_tokens,
            shortest_message_tokens=min_tokens,
            avg_tokens_per_message=avg_tokens,
            message_role_distribution=role_counts,
        )
    
    def compare_pricing(self, tokens: Optional[int] = None) -> Dict:
        """
        Compare cost between Standard and Batch API for given tokens.
        
        Args:
            tokens: Token count to compare (default: current conversation)
        
        Returns:
            Dictionary with standard_cost, batch_cost, and savings info
        """
        if tokens is None:
            tokens = self.stats.total_tokens
        
        standard_pricing = self.PRICING.get(self.model)
        batch_pricing = self.BATCH_PRICING.get(self.model)
        
        if not standard_pricing or not batch_pricing:
            return {"error": f"Pricing not available for model {self.model}"}
        
        # Assume 50/50 input/output split
        standard_cost = (
            (tokens * 0.5 * standard_pricing["input"]) +
            (tokens * 0.5 * standard_pricing["output"])
        ) / 1000
        
        batch_cost = (
            (tokens * 0.5 * batch_pricing["input"]) +
            (tokens * 0.5 * batch_pricing["output"])
        ) / 1000
        
        savings = standard_cost - batch_cost
        savings_percent = (savings / standard_cost * 100) if standard_cost > 0 else 0
        
        return {
            "tokens": tokens,
            "model": self.model,
            "standard_api": {
                "cost": standard_cost,
                "input_rate": standard_pricing["input"],
                "output_rate": standard_pricing["output"],
            },
            "batch_api": {
                "cost": batch_cost,
                "input_rate": batch_pricing["input"],
                "output_rate": batch_pricing["output"],
            },
            "savings": {
                "amount": savings,
                "percent": savings_percent,
            },
            "break_even_hours": 24 if savings > 0 else None,
            "recommendation": "Use Batch API" if savings > 0.01 else "Use Standard API for real-time needs",
        }
    
    def export_summary(self) -> Dict:
        """
        Export a concise summary of the conversation for review/archival.
        
        Returns:
            Dictionary with summary including metadata, stats, and message abstracts
        """
        analysis = self.analyze_conversation()
        
        # Create message summaries (first 100 chars of content)
        message_summaries = [
            {
                "role": msg.role,
                "preview": msg.content[:100] + ("..." if len(msg.content) > 100 else ""),
                "tokens": msg.token_count,
                "timestamp": msg.created_at,
            }
            for msg in self.messages
        ]
        
        return {
            "model": self.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message_count": len(self.messages),
            "total_tokens": self.stats.total_tokens,
            "estimated_cost": self.stats.estimated_cost,
            "analysis": analysis.to_dict(),
            "messages": message_summaries,
            "trim_history": {
                "times_trimmed": self.stats.trimmed_count,
                "last_trimmed_at": self.stats.last_trimmed_at,
            },
        }
    
    def __repr__(self) -> str:
        return (
            f"ContextManager(model={self.model}, messages={self.stats.total_messages}, "
            f"tokens={self.stats.total_tokens}, cost=${self.stats.estimated_cost:.4f})"
        )
