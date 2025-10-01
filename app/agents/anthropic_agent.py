"""Anthropic agent configuration with DuckDuckGo search capability."""

from typing import Any

import streamlit as st
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite import SqliteSaver

from .base import AgentConfig


class AnthropicAgentConfig(AgentConfig):
    """Agent using Anthropic models with optional extended thinking."""

    @staticmethod
    def get_name() -> str:
        return "Anthropic Agent"

    @staticmethod
    def render_options() -> dict[str, Any]:
        """Render UI for Anthropic agent options."""
        options = {}

        options["model"] = st.selectbox(
            "Model",
            [
                "claude-sonnet-4-20250514",
                "claude-3-7-sonnet-20250219",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
            ],
            index=0,
            key="anthropic_model",
            help="Select the Anthropic model to use"
        )

        options["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
            key="anthropic_temp",
            help="Controls randomness in responses"
        )

        options["max_tokens"] = st.number_input(
            "Max Tokens",
            min_value=1024,
            max_value=8192,
            value=5000,
            step=512,
            key="anthropic_max_tokens",
            help="Maximum tokens in response"
        )

        options["thinking_enabled"] = st.checkbox(
            "Enable Extended Thinking",
            value=False,
            key="anthropic_thinking",
            help="Enable extended thinking for complex reasoning"
        )

        if options["thinking_enabled"]:
            options["thinking_budget"] = st.number_input(
                "Thinking Budget (tokens)",
                min_value=500,
                max_value=10000,
                value=2000,
                step=500,
                key="anthropic_thinking_budget",
                help="Token budget for thinking phase"
            )

        return options

    @staticmethod
    def build(checkpoint: SqliteSaver, options: dict[str, Any]) -> Any:
        """Build Anthropic agent with DuckDuckGo search tool."""
        llm_params = {
            "model": options["model"],
            "temperature": options["temperature"],
            "max_tokens": options["max_tokens"],
        }

        if options.get("thinking_enabled"):
            llm_params["thinking"] = {
                "type": "enabled",
                "budget_tokens": options.get("thinking_budget", 2000)
            }

        return create_agent(
            ChatAnthropic(**llm_params),
            [DuckDuckGoSearchRun()],
            checkpointer=checkpoint
        )
