# Streamlit LangGraph Chat

[![CI](https://github.com/hirune924/streamlit-langgraph-chat/actions/workflows/ci.yml/badge.svg)](https://github.com/hirune924/streamlit-langgraph-chat/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://st-langgraph-chat.streamlit.app/)

**Extensible Reference Implementation for Streamlit + LangGraph Chat Applications**

A reference implementation for building chat applications with LangGraph and Streamlit. Includes practical features like streaming responses, persistent threads, and multimodal support, while providing clear extension points for customization.

![Demo](assets/demo.gif)

## What's Included

- **Streaming Responses**: Real-time token streaming with proper handling of thinking blocks and tool calls
- **Persistent Threads**: SQLite-based conversation history with thread switching and management
- **Multimodal Support**: Image input handling for compatible models
- **Extensible Agent System**: Abstract base class (`AgentConfig`) makes it straightforward to add custom agents with their own configuration UI

## Implemented Providers

- **OpenAI**: GPT-5, GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-4, o1, o3-mini
- **Anthropic**: Claude Sonnet 4, Claude 3.7 Sonnet, Claude 3.5 Sonnet/Haiku

Additional providers can be easily added by implementing the `AgentConfig` interface (see [Adding Custom Agents](#adding-custom-agents)).

## Quick Start

1. **Clone and install:**
```bash
git clone https://github.com/hirune924/streamlit-langgraph-chat.git
cd streamlit-langgraph-chat
uv sync
```

2. **Configure API keys:**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

3. **Run:**
```bash
uv run streamlit run app/app.py
```

The application opens at `http://localhost:8501`.

## Architecture Overview

The project is organized around a simple plugin system:

```
app/
├── app.py              # Main Streamlit UI
├── agent_config.py     # Agent registry (AVAILABLE_AGENTS)
├── utils.py            # Utility functions (streaming, thread management, etc.)
└── agents/
    ├── base.py         # AgentConfig abstract base class
    ├── openai_agent.py
    └── anthropic_agent.py
```

### Core Abstraction: `AgentConfig`

Each agent implements three methods:

```python
class AgentConfig(ABC):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Return display name for the agent."""

    @staticmethod
    @abstractmethod
    def render_options() -> dict[str, Any]:
        """Render Streamlit UI widgets and return selected options."""

    @staticmethod
    @abstractmethod
    def build(checkpoint: SqliteSaver, options: dict[str, Any]) -> Any:
        """Build and return the agent instance."""
```

This separation allows the main UI to be provider-agnostic while giving each agent full control over its configuration.

## Adding Custom Agents

The codebase provides two reference implementations: `OpenAIAgentConfig` and `AnthropicAgentConfig`. Both follow the same pattern:

```python
class OpenAIAgentConfig(AgentConfig):
    @staticmethod
    def get_name() -> str:
        return "OpenAI Agent"

    @staticmethod
    def render_options() -> dict[str, Any]:
        options = {}
        options["model"] = st.selectbox(
            "Model",
            ["gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "o1", "o3-mini"],
            index=0,
            key="openai_model",
            help="Select the OpenAI model to use"
        )
        # Additional options: temperature, max_tokens, etc.
        return options

    @staticmethod
    def build(checkpoint: SqliteSaver, options: dict[str, Any]) -> Any:
        llm_params = {
            "model": options["model"],
            "max_tokens": options["max_tokens"],
        }
        if "temperature" in options:
            llm_params["temperature"] = options["temperature"]

        return create_agent(
            ChatOpenAI(**llm_params),
            [DuckDuckGoSearchRun()],
            checkpointer=checkpoint
        )
```

To add a new agent, follow this pattern and register it in `AVAILABLE_AGENTS` in `app/agent_config.py`.

## Configuration

### Agent Settings

Each agent can define its own configuration options via `render_options()`. The included implementations provide:

**OpenAI Agent:**
- Model selection (GPT-4o, GPT-4 Turbo, o1, o3-mini, etc.)
- Temperature control
- Max tokens

**Anthropic Agent:**
- Model selection (Claude Sonnet 4, Claude 3.7 Sonnet, Claude 3.5 Sonnet/Haiku)
- Temperature control
- Max tokens
- Extended thinking toggle with token budget

You can customize these options or add your own when implementing custom agents.

### Thread Management

- **New Chat**: Create a fresh conversation
- **Select Thread**: Switch between saved conversations
- **Delete Current Chat**: Remove the active thread
- **Enable Streaming**: Toggle real-time response streaming

## Project Structure

```
streamlit-langgraph-chat/
├── app/
│   ├── app.py                    # Main application
│   ├── utils.py                  # Utilities (streaming, threads, display)
│   ├── agent_config.py           # Agent registry
│   └── agents/
│       ├── __init__.py
│       ├── base.py               # AgentConfig ABC
│       ├── anthropic_agent.py    # Anthropic implementation
│       └── openai_agent.py       # OpenAI implementation
├── pyproject.toml                # Dependencies and tool config
├── .env.example                  # Environment template
└── README.md
```

## Development

### Type Checking and Linting

```bash
# Type check
uv run pyright app/

# Lint
uv run ruff check app/
```

### Key Dependencies

- **streamlit**: Web UI framework
- **langchain** (>=1.0.0a9): LLM framework (alpha version)
- **langgraph** (>=1.0.0a3): State management and checkpointing (alpha version)
- **langchain-anthropic**: Anthropic integration
- **langchain-openai**: OpenAI integration
- **langchain-community**: Community tools (DuckDuckGo search, etc.)

## License

[MIT License](LICENSE)

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.
