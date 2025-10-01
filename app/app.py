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
    st.title("LangGraph Agent Chat")
    initialize_checkpoint()

    with st.sidebar:
        st.header("💬 Conversations")

        # Thread management
        threads, latest = get_threads(st.session_state.checkpoint)
        current = st.session_state.get('thread_id') or latest or str(uuid.uuid4())
        st.session_state.thread_id = current

        if current not in threads:
            threads.insert(0, current)

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

        use_streaming = st.checkbox(
            "Enable Streaming",
            value=True,
            key="use_streaming",
            help="Stream responses in real-time (disable for non-streaming models)"
        )

        st.divider()
        st.header("⚙️ Agent Settings")

        # Agent selection
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

        st.subheader("Agent Options")
        options = selected_agent_config.render_options()

        # Rebuild agent if needed
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

    submission = st.chat_input(
        "Ask me anything!",
        accept_file=True,
        file_type=["jpg", "jpeg", "png"],
    )

    display_chat_history(st.session_state.checkpoint, st.session_state.thread_id)

    if submission:
        # Parse submission
        if isinstance(submission, dict):
            user_text = submission.get("text", "")
            user_files = submission.get("files", [])
        else:
            user_text = getattr(submission, "text", submission if isinstance(submission, str) else "")
            user_files = getattr(submission, "files", [])

        content = convert_input_to_content(user_text, user_files)

        # Display user message
        with st.chat_message("user"):
            render_content(content)

        # Assistant response
        use_streaming = st.session_state.get("use_streaming", True)

        with st.chat_message("assistant"):
            if use_streaming:
                # Stream response with fixed order: thinking → tools → text
                thinking_container = st.container()
                tools_container = st.container()
                text_container = st.container()
                text_buffer = []

                def render_to_container(title, payload):
                    target = thinking_container if "Thinking" in title else tools_container
                    with target:
                        render_tool(title, payload)

                with text_container:
                    text_element = st.empty()

                stream = st.session_state.agent.stream(
                    {"messages": [HumanMessage(content=content)]},  # type: ignore[arg-type]
                    config={"configurable": {"thread_id": st.session_state.thread_id}},
                    stream_mode="messages",
                )

                for text_chunk in extract_text_chunks(stream, tool_callback=render_to_container):
                    text_buffer.append(text_chunk)
                    text_element.markdown("".join(text_buffer))
            else:
                # Invoke (non-streaming)
                with st.spinner("Processing..."):
                    st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=content)]},  # type: ignore[arg-type]
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )

        st.rerun()


if __name__ == "__main__":
    main()
