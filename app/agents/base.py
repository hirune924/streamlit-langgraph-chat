"""Base class for agent configurations."""

from abc import ABC, abstractmethod
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


class AgentConfig(ABC):
    """Base class for agent configurations.

    Each agent implementation should subclass this and implement:
    - get_name(): Return the display name
    - render_options(): Render Streamlit UI and return options dict
    - build(): Create agent instance from options
    """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Return the display name for this agent."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def render_options() -> dict[str, Any]:
        """Render Streamlit UI for agent options and return selected values.

        This method should use Streamlit widgets (st.selectbox, st.slider, etc.)
        to render the agent's configuration UI and return a dict of option values.

        Returns:
            dict: Configuration options selected by the user
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build(checkpoint: SqliteSaver, options: dict[str, Any]) -> Any:
        """Build and return an agent instance with the given options.

        Args:
            checkpoint: LangGraph checkpoint for persistence
            options: Configuration options from render_options()

        Returns:
            Agent instance ready to use
        """
        raise NotImplementedError
