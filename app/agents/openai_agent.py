"""OpenAI agent configuration with DuckDuckGo search capability."""

from typing import Any

import streamlit as st
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from .base import AgentConfig


class OpenAIAgentConfig(AgentConfig):
    """Agent using OpenAI models."""

    @staticmethod
    def get_name() -> str:
        return "OpenAI Agent"

    @staticmethod
    def render_options() -> dict[str, Any]:
        """Render UI for OpenAI agent options."""
        options = {}

        options["model"] = st.selectbox(
            "Model",
            [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "o1",
                "o1-mini",
                "o3-mini",
            ],
            index=0,
            key="openai_model",
            help="Select the OpenAI model to use"
        )

        # o1/o3 models don't support temperature parameter
        is_thinking_model = options["model"] in ["o1", "o1-mini", "o3-mini"]

        if not is_thinking_model:
            options["temperature"] = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                key="openai_temp",
                help="Controls randomness in responses"
            )

        options["max_tokens"] = st.number_input(
            "Max Tokens",
            min_value=512,
            max_value=16384,
            value=4096,
            step=512,
            key="openai_max_tokens",
            help="Maximum tokens in response"
        )

        return options

    @staticmethod
    def build(checkpoint: SqliteSaver, options: dict[str, Any]) -> Any:
        """Build OpenAI agent with DuckDuckGo search tool."""
        llm_params = {
            "model": options["model"],
            "max_tokens": options["max_tokens"],
        }

        # Only add temperature for non-thinking models
        if "temperature" in options:
            llm_params["temperature"] = options["temperature"]

        return create_agent(
            ChatOpenAI(**llm_params),
            [DuckDuckGoSearchRun()],
            checkpointer=checkpoint
        )
