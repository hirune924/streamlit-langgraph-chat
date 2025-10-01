"""Utility functions for the Streamlit chat application."""

import base64
import sqlite3
from typing import Any, Callable, Iterator

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

TOOL_TYPES = {"tool", "tool_message", "function"}


def render_tool(title: str, payload: Any) -> None:
    """Display a tool call result in an expandable container."""
    with st.expander(title or "Tool", expanded=False):
        st.write(payload)


def extract_text_chunks(
    message_stream: Iterator[Any],
    tool_callback: Callable[[str, Any], None] | None = None
) -> Iterator[str]:
    """Extract text chunks from a stream, handling tool and thinking messages separately."""
    thinking_buffer: list[str] = []

    def flush_thinking() -> None:
        if thinking_buffer and tool_callback:
            tool_callback("💭 Thinking", "".join(thinking_buffer))
            thinking_buffer.clear()

    for event in message_stream:
        chunk = event[0] if isinstance(event, tuple) else event
        msg_type = getattr(chunk, "type", None)

        # Handle ToolMessage (tool execution results)
        if isinstance(msg_type, str) and msg_type.lower() in TOOL_TYPES:
            if tool_callback:
                tool_name = getattr(chunk, "name", None) or getattr(chunk, "tool", None) or "Tool"
                payload = getattr(chunk, "content", None) or (chunk if isinstance(chunk, dict) else None)
                tool_callback(f"Tool: {tool_name}", payload)
            continue

        content = getattr(chunk, "content", None)

        # Anthropic: content is list of parts
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue

                part_type = part.get("type")

                if part_type == "thinking":
                    thinking_buffer.append(part.get("thinking", ""))
                elif part_type == "tool_use":
                    flush_thinking()
                elif part_type == "text":
                    flush_thinking()
                    if text := part.get("text"):
                        yield text

        # OpenAI: content is string
        elif isinstance(content, str):
            flush_thinking()
            yield content

    flush_thinking()


# Thread management functions
def get_threads(checkpoint: SqliteSaver) -> tuple[list[str], str | None]:
    """Return (threads_latest_first, latest_thread_id)."""
    with checkpoint.cursor() as cur:
        cur.execute(
            """
            SELECT thread_id, MAX(rowid) AS mr
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY mr DESC
            """
        )
        rows = cur.fetchall()

    threads = [row[0] for row in rows]
    return threads, threads[0] if threads else None


def get_thread_title(thread_id: str) -> str:
    """Get the title for a thread based on its first message."""
    tup = st.session_state.checkpoint.get_tuple({"configurable": {"thread_id": thread_id}})
    if not tup:
        return "New thread"

    messages = getattr(tup, "checkpoint", {}).get("channel_values", {}).get("messages", [])
    if not messages:
        return "New thread"

    content = getattr(messages[0], "content", None)

    # String content
    if isinstance(content, str) and content.strip():
        return content.strip().splitlines()[0]

    # List content (multimodal)
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                if text := part.get("text", "").strip():
                    return text.splitlines()[0]

    return "New thread"


def on_delete_thread(thread_id: str) -> None:
    """Delete a thread and reset the current thread ID."""
    st.session_state.checkpoint.delete_thread(thread_id)
    st.session_state.thread_id = None


# Message display functions
def render_part(part: Any) -> None:
    """Display a single part of multimodal message content (text or image)."""
    if not isinstance(part, dict):
        return

    part_type = part.get("type")
    if part_type == "text":
        st.write(part.get("text", ""))
    elif part_type == "image_url":
        url_part = part.get("image_url")
        url = url_part.get("url") if isinstance(url_part, dict) else url_part
        if url:
            st.image(url)


def render_content(content: str | list[Any]) -> None:
    """Display message content (string or list of multimodal parts)."""
    if isinstance(content, list):
        for part in content:
            render_part(part)
    else:
        st.write(content)


def convert_input_to_content(user_text: str, user_files: list[Any]) -> str | list[dict[str, Any]]:
    """Convert Streamlit chat input to LangChain message content format."""
    if not user_files:
        return user_text.strip() if isinstance(user_text, str) else ""

    parts: list[dict[str, Any]] = []
    if isinstance(user_text, str) and user_text.strip():
        parts.append({"type": "text", "text": user_text.strip()})

    for f in user_files:
        mime_type = getattr(f, 'type', None) or 'image/png'
        encoded = base64.b64encode(f.getvalue()).decode('utf-8')
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"}
        })

    return parts


def get_speaker(msg: Any) -> str:
    """Determine if message is from user or assistant."""
    msg_type = getattr(msg, "type", "").lower()
    return "user" if msg_type in {"user", "human"} else "assistant"


def show_message(msg: Any) -> None:
    """Display a single message based on its type."""
    msg_type = getattr(msg, "type", "").lower()
    content = getattr(msg, "content", "")

    if msg_type in {"user", "human", "ai"}:
        if not content:
            return

        # Display thinking blocks if present
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "thinking":
                    with st.expander("💭 Thinking", expanded=False):
                        st.write(part.get("thinking", ""))

        render_content(content)

    elif msg_type in TOOL_TYPES:
        tool_name = getattr(msg, "name", None) or "Tool"
        with st.expander(f"Tool: {tool_name}", expanded=False):
            st.write(content)


def display_chat_history(checkpoint: SqliteSaver, thread_id: str) -> None:
    """Display the chat history for a given thread."""
    tup = checkpoint.get_tuple({"configurable": {"thread_id": thread_id}})
    if not tup:
        return

    messages = getattr(tup, "checkpoint", {}).get("channel_values", {}).get("messages", [])
    if not messages:
        return

    i = 0
    while i < len(messages):
        speaker = get_speaker(messages[i])
        with st.chat_message(speaker):
            while i < len(messages) and get_speaker(messages[i]) == speaker:
                show_message(messages[i])
                i += 1


def initialize_checkpoint() -> None:
    """Initialize checkpoint for conversation persistence."""
    if "checkpoint" not in st.session_state:
        st.session_state.checkpoint = SqliteSaver(
            sqlite3.connect("checkpoint.db", check_same_thread=False)
        )
