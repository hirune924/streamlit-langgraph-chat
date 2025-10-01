import uuid

import streamlit as st
from agent_config import AVAILABLE_AGENTS, get_agent_by_name
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utils import (
    convert_input_to_content,
    display_chat_history,
    extract_text_chunks,
    get_thread_title,
    get_threads,
    initialize_checkpoint,
    on_delete_thread,
    render_content,
    render_tool,
)

load_dotenv()


def main():
    """Main Streamlit application for LangGraph Agent Chat."""
    st.title("LangGraph Agent Chat")
    initialize_checkpoint()

    # === Sidebar: Conversation and Agent Management ===
    with st.sidebar:
        st.header("💬 Conversations")

        # Thread management: Get existing threads and determine current thread
        threads, latest = get_threads(st.session_state.checkpoint)
        current = st.session_state.get('thread_id') or latest or str(uuid.uuid4())
        st.session_state.thread_id = current

        # Ensure current thread appears in the list
        if current not in threads:
            threads.insert(0, current)

        # Thread selection dropdown
        st.selectbox(
            "Select thread",
            threads,
            key="select_thread",
            index=(threads.index(current) if current in threads else 0),
            format_func=get_thread_title,
            on_change=lambda: setattr(st.session_state, 'thread_id', st.session_state.select_thread),
        )
        st.button("New chat", on_click=lambda: setattr(st.session_state, 'thread_id', str(uuid.uuid4())))
        st.button("Delete current chat", on_click=on_delete_thread, args=(st.session_state.thread_id,))

        # Streaming toggle
        use_streaming = st.checkbox(
            "Enable Streaming",
            value=True,
            key="use_streaming",
            help="Stream responses in real-time (disable for non-streaming models)"
        )

        st.divider()
        st.header("⚙️ Agent Settings")

        # Agent selection: Display available agents
        agent_names = [agent.get_name() for agent in AVAILABLE_AGENTS]
        if not agent_names:
            st.warning("No agents configured. Please add agents to AVAILABLE_AGENTS.")
            st.stop()

        selected_name = st.selectbox(
            "Select Agent",
            agent_names,
            key="agent_selector",
            help="Choose which agent to use for conversations"
        )

        selected_agent_config = get_agent_by_name(selected_name)  # type: ignore[arg-type]
        if not selected_agent_config:
            st.warning("No agents configured. Please add agents to AVAILABLE_AGENTS.")
            st.stop()

        # Render agent-specific configuration UI
        st.subheader("Agent Options")
        options = selected_agent_config.render_options()

        # Rebuild agent if configuration changed
        prev_name = st.session_state.get("current_agent_name")
        needs_rebuild = (
            prev_name != selected_name or
            st.session_state.get("agent_options") != options or
            "agent" not in st.session_state
        )

        if needs_rebuild:
            with st.spinner(f"Initializing {selected_name}..."):
                st.session_state.agent = selected_agent_config.build(st.session_state.checkpoint, options)
                st.session_state.current_agent_name = selected_name
                st.session_state.agent_options = options

    # === Main Area: Chat Interface ===
    # Chat input with multimodal support (text + images)
    submission = st.chat_input(
        "Ask me anything!",
        accept_file=True,
        file_type=["jpg", "jpeg", "png"],
    )

    # Display conversation history
    display_chat_history(st.session_state.checkpoint, st.session_state.thread_id)

    # Handle new user input
    if submission:
        # Parse submission (handles both dict and object formats)
        if isinstance(submission, dict):
            user_text = submission.get("text", "")
            user_files = submission.get("files", [])
        else:
            user_text = getattr(submission, "text", submission if isinstance(submission, str) else "")
            user_files = getattr(submission, "files", [])

        # Convert to LangChain message format
        content = convert_input_to_content(user_text, user_files)

        # Display user message
        with st.chat_message("user"):
            render_content(content)

        # Generate and display assistant response
        use_streaming = st.session_state.get("use_streaming", True)

        with st.chat_message("assistant"):
            if use_streaming:
                # Streaming mode: Display thinking, tools, and text in separate containers
                # This ensures proper ordering even when messages arrive out of order
                thinking_container = st.container()
                tools_container = st.container()
                text_container = st.container()
                text_buffer = []

                def render_to_container(title, payload):
                    """Route thinking/tool messages to appropriate containers."""
                    target = thinking_container if "Thinking" in title else tools_container
                    with target:
                        render_tool(title, payload)

                with text_container:
                    text_element = st.empty()

                # Stream agent response
                stream = st.session_state.agent.stream(
                    {"messages": [HumanMessage(content=content)]},  # type: ignore[arg-type]
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                    stream_mode="messages",
                )

                # Process stream and display text chunks incrementally
                for text_chunk in extract_text_chunks(stream, tool_callback=render_to_container):
                    text_buffer.append(text_chunk)
                    text_element.markdown("".join(text_buffer))
            else:
                # Non-streaming mode: Invoke agent and wait for complete response
                with st.spinner("Processing..."):
                    st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=content)]},  # type: ignore[arg-type]
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )

        # Rerun to display the new message from history
        st.rerun()


if __name__ == "__main__":
    main()
