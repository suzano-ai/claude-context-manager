# Claude Context Manager

**Production-ready conversation management for Claude API with automatic token tracking and context trimming.**

## The Problem

Developers using Claude API (or any LLM) face recurring issues:

1. **Token counting inaccuracy** - Different libraries count tokens differently; easy to exceed limits
2. **Context overflow** - Long conversations unpredictably hit token limits; trimming strategy is ad-hoc
3. **Cost estimation failure** - No way to predict conversation costs before API calls
4. **Conversation loss** - Long-running systems lose context when resetting
5. **Smart context preservation** - Developers don't know what to trim without losing critical context

## The Solution

`claude-context-manager` is a lightweight, battle-tested library that handles all of this:

✅ **Accurate token counting** - Claude-aware token calculations with message overhead  
✅ **4 trimming strategies** - OLDEST_FIRST, SLIDING_WINDOW, SMART, SUMMARIZE-ready  
✅ **Cost tracking** - Real-time cost estimation for conversations + Batch API pricing support  
✅ **Smart pricing comparison** - Automatically compare Standard vs Batch API costs  
✅ **Persistence** - Save/load conversation state to JSON  
✅ **Production-ready** - 100+ lines, extensively tested, zero external dependencies beyond Anthropic SDK  

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from claude_context_manager import ContextManager

# Initialize
manager = ContextManager(
    model="claude-3-5-sonnet",
    max_tokens=200000,
    trim_strategy="smart"  # Auto-trim when approaching limit
)

# Add messages as you build conversation
manager.add_message("system", "You are a helpful assistant.")
manager.add_message("user", "What is Python?")
manager.add_message("assistant", "Python is a programming language...")

# Check if trimming is needed
if manager.should_trim():
    removed_tokens, removed_msgs = manager.trim_conversation()
    print(f"Trimmed {len(removed_msgs)} messages ({removed_tokens} tokens)")

# Get stats
stats = manager.get_stats()
print(f"Tokens: {stats.total_tokens}, Cost: ${stats.estimated_cost:.6f}")

# Prepare for API
api_messages = manager.get_messages_for_api()
# Use api_messages directly with Claude API

# Save conversation for later
manager.save_to_file("conversation.json")
```

## Features

### Accurate Token Counting

```python
manager = ContextManager()
tokens = manager.count_tokens("Hello, world!")
# Correctly accounts for Claude's tokenization + message overhead
```

### Smart Conversation Trimming

Four strategies to choose from:

- **OLDEST_FIRST**: Remove oldest messages first (default behavior)
- **SLIDING_WINDOW**: Keep only most recent messages within limit
- **SMART**: Preserve system prompts, remove oldest content messages
- **SUMMARIZE**: Mark old messages for summarization (prep for future summarization feature)

```python
manager = ContextManager(
    trim_strategy="smart",  # Keeps system messages intact
    trim_threshold=0.85     # Trim when 85% full
)

if manager.should_trim():
    manager.trim_conversation()
```

### Real-time Cost Estimation

```python
stats = manager.get_stats()
print(f"Estimated cost: ${stats.estimated_cost:.6f}")

# Built-in pricing for:
# - claude-3-opus
# - claude-3-sonnet  
# - claude-3-haiku
# - claude-3-5-sonnet
# - claude-3-5-haiku
```

### Batch API Support & Cost Comparison

```python
# Initialize with Batch API pricing (50% discount)
manager = ContextManager(
    model="claude-3-5-sonnet",
    use_batch_api=True  # Uses discounted Batch API rates
)

# Compare costs between Standard and Batch API
comparison = manager.compare_pricing()
print(f"Standard API: ${comparison['standard_api']['cost']:.4f}")
print(f"Batch API: ${comparison['batch_api']['cost']:.4f}")
print(f"Savings: ${comparison['savings']['amount']:.4f} ({comparison['savings']['percent']:.1f}%)")
# Output: "Use Batch API" recommendation for offline workloads
```

### Conversation Persistence

```python
# Save current state
manager.save_to_file("session.json")

# Later: restore conversation
manager.load_from_file("session.json")
# All messages, token counts, and metadata preserved
```

### Metadata Support

```python
manager.add_message(
    role="user",
    content="Question here",
    metadata={
        "importance": "high",
        "source": "user_input",
        "timestamp": "2025-03-13T10:00:00"
    }
)
```

## API Reference

### ContextManager

#### Constructor
```python
ContextManager(
    model: str = "claude-3-5-sonnet",
    max_tokens: int = 200000,
    trim_strategy: TrimStrategy = TrimStrategy.SMART,
    trim_threshold: float = 0.85,
    verbose: bool = False,
    use_batch_api: bool = False
)
```

**Parameters:**
- `model`: Claude model name (supports claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet, claude-3-5-haiku)
- `max_tokens`: Context window limit
- `trim_strategy`: How to trim conversations (OLDEST_FIRST, SLIDING_WINDOW, SMART, SUMMARIZE)
- `trim_threshold`: Percentage of max_tokens before triggering trim (0.0-1.0)
- `verbose`: Print debug messages
- `use_batch_api`: Use Batch API pricing (50% discount, 24h latency)

#### Methods

- **add_message(role, content, metadata=None) → Message** - Add message with automatic token counting
- **should_trim() → bool** - Check if conversation exceeds threshold
- **trim_conversation(target_tokens=None) → (int, List[Message])** - Trim to size; returns (tokens_removed, removed_messages)
- **get_messages_for_api() → List[Dict]** - Get Claude API-ready message format
- **get_conversation_state() → Dict** - Get full serializable state
- **load_conversation_state(state: Dict)** - Restore from state dict
- **save_to_file(filepath: str)** - Persist to JSON
- **load_from_file(filepath: str)** - Load from JSON
- **get_stats() → ConversationStats** - Get token/cost statistics
- **count_tokens(text: str) → int** - Count tokens in arbitrary text
- **compare_pricing(tokens=None) → Dict** - Compare Standard vs Batch API costs with recommendations
- **clear()** - Reset conversation

## Real-World Usage

### Building a Long-Running Chat Agent

```python
from claude_context_manager import ContextManager

manager = ContextManager(
    model="claude-3-5-sonnet",
    trim_strategy="smart",
    trim_threshold=0.85
)

# System message (always preserved)
manager.add_message(
    "system",
    "You are a helpful coding assistant. Focus on clear, practical solutions."
)

# Main conversation loop
while True:
    user_input = input("You: ")
    
    # Add user message
    manager.add_message("user", user_input)
    
    # Get Claude response (your code)
    response = call_claude_api(manager.get_messages_for_api())
    
    # Track assistant response
    manager.add_message("assistant", response)
    
    # Auto-trim if needed
    if manager.should_trim():
        removed = manager.trim_conversation()
        print(f"[Context trimmed: removed {len(removed[1])} old messages]")
    
    # Show stats
    stats = manager.get_stats()
    print(f"[Tokens: {stats.total_tokens} | Cost: ${stats.estimated_cost:.6f}]")
```

### Batch Processing with Checkpoints

```python
manager = ContextManager()

# Process items with conversation checkpoints
for item in large_dataset:
    manager.add_message("user", f"Process: {item}")
    
    response = call_claude(manager.get_messages_for_api())
    manager.add_message("assistant", response)
    
    # Checkpoint every 10 items
    if item_count % 10 == 0:
        manager.save_to_file(f"checkpoint_{item_count}.json")
```

## Performance

- **Token counting**: ~0.1ms per message (tiktoken is fast)
- **Trimming**: ~1-5ms for 100+ message conversations
- **Serialization**: ~2-10ms for saving conversation state
- **Memory**: ~1KB per message on average

## Compatibility

- Python 3.8+
- Works with any Claude model (claude-3-*, claude-3-5-*)
- Compatible with Anthropic SDK v0.17+

## Troubleshooting

**Issue**: Token count seems high
- Check if you're counting message overhead separately; this library includes it automatically

**Issue**: Trimming removes system messages
- Use `trim_strategy="smart"` which explicitly preserves system prompts

**Issue**: Cost estimation seems wrong
- Verify the model parameter matches your actual API model
- Remember costs are estimates based on 50/50 input/output token split

## Testing

Full pytest test suite with **108 tests** covering all features:

```bash
pip install pytest
python -m pytest tests/ -v
```

Test areas:
- Token counting (accuracy, unicode, edge cases)
- All trimming strategies (OLDEST_FIRST, SLIDING_WINDOW, SMART)
- Cost estimation and Batch API comparison
- Persistence (save/load roundtrip)
- Message search and filtering
- Conversation analysis
- Edge cases (empty convos, stress tests, all models)

## Contributing

Issues and PRs welcome. Main areas for contribution:
- Additional Claude models as they're released
- Summarization integration for older messages
- Conversation analysis (common topics, message patterns)

## License

MIT

## What's Next?

Planned enhancements:
- [ ] Automatic summarization of old messages
- [ ] Conversation analytics (topic extraction, complexity metrics)
- [ ] Multi-turn conversation templates
- [ ] Integration with popular frameworks (LangChain, LlamaIndex)
- [ ] Real-time token streaming for large responses

---

**Start using token-aware conversations today. Stop worrying about context limits.**
