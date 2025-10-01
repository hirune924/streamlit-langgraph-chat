"""Agent configuration registry."""

from agents import AgentConfig, AnthropicAgentConfig, OpenAIAgentConfig

# List of all available agent configurations
# Add new agents here to make them available in the UI
AVAILABLE_AGENTS = [
    AnthropicAgentConfig,
    OpenAIAgentConfig,
]


def get_agent_by_name(name: str) -> type[AgentConfig] | None:
    """Get agent config class by display name."""
    for agent in AVAILABLE_AGENTS:
        if agent.get_name() == name:
            return agent
    return None
