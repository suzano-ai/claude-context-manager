"""
Example usage of Claude Context Manager.

Demonstrates:
- Building a multi-turn conversation
- Monitoring token usage
- Automatic trimming when approaching limits
- Saving/loading conversation state
"""

import json
import os
from claude_context_manager import ContextManager, TrimStrategy


def example_basic_conversation():
    """Basic conversation with token tracking."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Conversation with Token Tracking")
    print("=" * 60)
    
    manager = ContextManager(model="claude-3-5-sonnet", verbose=True)
    
    # Add system message
    manager.add_message(
        role="system",
        content="You are a helpful AI assistant specializing in Python programming.",
        metadata={"importance": "critical"}
    )
    
    # Simulate a conversation
    manager.add_message(
        role="user",
        content="How do I manage token counts in API conversations?"
    )
    
    manager.add_message(
        role="assistant",
        content="Token counting is important for managing API costs and context windows. "
                "You can use libraries like tiktoken to count tokens accurately. "
                "Each message has overhead, and you should track total tokens to avoid hitting limits."
    )
    
    manager.add_message(
        role="user",
        content="Can you show me a code example?"
    )
    
    manager.add_message(
        role="assistant",
        content="""Here's a basic example:
        
import tiktoken
encoder = tiktoken.encoding_for_model("gpt-4")
text = "Your text here"
tokens = len(encoder.encode(text))
print(f"Token count: {tokens}")

This gives you an accurate count of how many tokens your text uses."""
    )
    
    # Show stats
    stats = manager.get_stats()
    print(f"\n[STATS] {stats}")
    print(f"[MESSAGES] Total: {stats.total_messages}")
    print(f"[TOKENS] Total: {stats.total_tokens}")
    print(f"[COST] Estimated: ${stats.estimated_cost:.6f}")
    print()


def example_with_trimming():
    """Demonstrate automatic conversation trimming."""
    print("=" * 60)
    print("EXAMPLE 2: Automatic Conversation Trimming")
    print("=" * 60)
    
    # Create manager with low token limit for demo
    manager = ContextManager(
        model="claude-3-5-sonnet",
        max_tokens=500,  # Small for demo
        trim_strategy=TrimStrategy.SMART,
        trim_threshold=0.8,  # Trim at 80%
        verbose=True
    )
    
    # Add system message (critical - should be preserved during trimming)
    manager.add_message(
        role="system",
        content="You are a concise AI assistant that helps debug code.",
        metadata={"importance": "critical"}
    )
    
    # Add many messages to trigger trimming
    for i in range(5):
        manager.add_message(
            role="user",
            content=f"Question {i+1}: " + ("This is a sample question about Python. " * 20)
        )
        manager.add_message(
            role="assistant",
            content=f"Answer {i+1}: " + ("This is a detailed answer about the topic. " * 20)
        )
    
    print(f"\n[BEFORE TRIM] Total tokens: {manager.stats.total_tokens}")
    print(f"[BEFORE TRIM] Messages: {manager.stats.total_messages}")
    
    # Trigger trimming if needed
    if manager.should_trim():
        removed_tokens, removed_msgs = manager.trim_conversation()
        print(f"\n[AFTER TRIM] Tokens removed: {removed_tokens}")
        print(f"[AFTER TRIM] Messages removed: {len(removed_msgs)}")
        print(f"[AFTER TRIM] Total tokens now: {manager.stats.total_tokens}")
        print(f"[AFTER TRIM] Messages remaining: {manager.stats.total_messages}")
    
    # Check that system message is preserved
    system_msgs = [m for m in manager.messages if m.role == "system"]
    print(f"\n[CHECK] System messages preserved: {len(system_msgs) > 0}")
    print()


def example_persistence():
    """Save and restore conversation."""
    print("=" * 60)
    print("EXAMPLE 3: Conversation Persistence")
    print("=" * 60)
    
    manager1 = ContextManager(model="claude-3-5-sonnet", verbose=False)
    
    # Build conversation
    manager1.add_message(role="system", content="You are a helpful assistant.")
    manager1.add_message(role="user", content="What is machine learning?")
    manager1.add_message(
        role="assistant",
        content="Machine learning is a subset of artificial intelligence..."
    )
    
    print(f"[ORIGINAL] Messages: {manager1.stats.total_messages}")
    print(f"[ORIGINAL] Tokens: {manager1.stats.total_tokens}")
    
    # Save to file
    filepath = "/tmp/conversation_state.json"
    manager1.save_to_file(filepath)
    print(f"\n[SAVED] Conversation saved to {filepath}")
    
    # Load into new manager
    manager2 = ContextManager()
    manager2.load_from_file(filepath)
    
    print(f"\n[RESTORED] Messages: {manager2.stats.total_messages}")
    print(f"[RESTORED] Tokens: {manager2.stats.total_tokens}")
    print(f"[RESTORED] Model: {manager2.model}")
    
    # Verify content
    last_msg = manager2.messages[-1]
    print(f"[VERIFIED] Last message starts with: {last_msg.content[:50]}...")
    print()


def example_api_ready():
    """Show how to prepare messages for Claude API."""
    print("=" * 60)
    print("EXAMPLE 4: Preparing Messages for Claude API")
    print("=" * 60)
    
    manager = ContextManager(verbose=False)
    
    manager.add_message(role="system", content="You are a helpful coding assistant.")
    manager.add_message(role="user", content="How do I sort a list in Python?")
    manager.add_message(
        role="assistant",
        content="You can use the sorted() function or the .sort() method."
    )
    
    # Get API-ready format
    api_messages = manager.get_messages_for_api()
    
    print("[API FORMAT] Messages ready for Claude API:")
    print(json.dumps(api_messages, indent=2))
    
    print(f"\n[STATS] Tokens: {manager.stats.total_tokens}")
    print(f"[STATS] Cost estimate: ${manager.stats.estimated_cost:.6f}")
    print()


def example_different_strategies():
    """Compare different trimming strategies."""
    print("=" * 60)
    print("EXAMPLE 5: Comparing Trimming Strategies")
    print("=" * 60)
    
    strategies = [TrimStrategy.OLDEST_FIRST, TrimStrategy.SLIDING_WINDOW, TrimStrategy.SMART]
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.value} ---")
        
        manager = ContextManager(
            max_tokens=300,
            trim_strategy=strategy,
            trim_threshold=0.8,
            verbose=False
        )
        
        # Add messages
        manager.add_message(role="system", content="You are helpful.")
        for i in range(4):
            manager.add_message(role="user", content=f"Question {i}: " + ("test " * 30))
            manager.add_message(role="assistant", content=f"Answer {i}: " + ("response " * 30))
        
        print(f"Before trim: {manager.stats.total_messages} messages, {manager.stats.total_tokens} tokens")
        
        if manager.should_trim():
            manager.trim_conversation()
        
        print(f"After trim: {manager.stats.total_messages} messages, {manager.stats.total_tokens} tokens")
        
        # Show which messages remain
        roles = [m.role for m in manager.messages]
        print(f"Message roles remaining: {roles}")


if __name__ == "__main__":
    print("\n")
    print("Claude Context Manager - Usage Examples")
    print("=" * 60)
    print()
    
    example_basic_conversation()
    example_with_trimming()
    example_persistence()
    example_api_ready()
    example_different_strategies()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
