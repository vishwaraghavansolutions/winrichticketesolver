import streamlit as st
import pandas as pd

from agents.ticket_query_agent import TicketQueryAgent
from utils.s3storage import S3Storage
import utils.navbar as navbar
import utils.agentnavbar as agentnavbar


EXAMPLES = [
    "Which product has the most negative sentiment tickets?",
    "Show all open tickets grouped by product and customer request with ticket IDs",
    "What is the average resolution time per agent?",
    "Which customer requests appear more than 3 times?",
    "List tickets that breached SLA grouped by product",
    "How many tickets are open vs closed per product?",
    "Show me the top 5 agents by number of tickets handled",
]


@st.cache_data(ttl=300, show_spinner="Loading analytics data...")
def load_analytics(_s3, bucket, path):
    try:
        return _s3.read_parquet(bucket, path)
    except Exception as e:
        st.error(f"Failed to load analytics data: {e}")
        return pd.DataFrame()


def main():
    st.title("🤖 Ask Me Anything")
    st.caption("Ask plain-English questions about your ticket analytics data.")

    # Auth guard
    if "role" not in st.session_state:
        st.write("Session expired. Please log in again.")
        st.button("Go to Login Page", on_click=st.switch_page("login.py"))
        st.stop()

    if st.session_state["role"] == "admin":
        st.success("Logged in as Admin")
        navbar.navbar()
    else:
        agentnavbar.agentnavbar()
        st.success(f"Logged in as Agent: {st.session_state['agent_id']}")

    # Load analytics parquet
    s3 = S3Storage(
        aws_access_key=st.secrets["aws"]["access_key"],
        aws_secret_key=st.secrets["aws"]["secret_key"],
        region=st.secrets["aws"].get("region", "ap-south-1"),
    )
    bucket = st.secrets["aws"]["bucket"]
    analytics_path = st.secrets["aws"]["analytics_path"]

    df = load_analytics(s3, bucket, analytics_path)

    if df.empty:
        st.warning("No analytics data available. Run the Ticket Analytics page first to generate it.")
        st.stop()

    # Agent-level data scoping
    if st.session_state["role"] == "agent":
        df = df[df["assigned_agent"] == st.session_state["agent_id"]]

    # Session state initialisation
    if "ama_claude_history" not in st.session_state:
        st.session_state["ama_claude_history"] = []   # messages sent to Claude
    if "ama_display" not in st.session_state:
        st.session_state["ama_display"] = []           # {role, content, dataframes?}

    # Sidebar: controls + examples
    with st.sidebar:
        st.markdown("### 💡 Example questions")
        for ex in EXAMPLES:
            if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=True):
                st.session_state["ama_prefill"] = ex

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state["ama_claude_history"] = []
            st.session_state["ama_display"] = []
            st.rerun()

        st.markdown("### 📋 Data columns")
        schema_lines = []
        for col in df.columns:
            schema_lines.append(f"`{col}` ({df[col].dtype})")
        st.markdown("  \n".join(schema_lines))

    # Render conversation history
    for msg in st.session_state["ama_display"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for frame in msg.get("dataframes", []):
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download CSV",
                    data=frame.to_csv(index=False),
                    file_name="query_result.csv",
                    mime="text/csv",
                    key=f"dl_{id(frame)}",
                )

    # Prefill from sidebar example click
    prefill = st.session_state.pop("ama_prefill", "")

    # Chat input
    question = st.chat_input("Ask a question about your ticket data…") or prefill

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state["ama_display"].append({"role": "user", "content": question})

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                agent = TicketQueryAgent(df, api_key=st.secrets["OPENAI_API_KEY"])
                response = agent.run(
                    "chat",
                    {
                        "question": question,
                        "history": st.session_state["ama_claude_history"],
                    },
                )

            if response.status.value == "success":
                reply = response.output["reply"]
                dataframes = response.output.get("dataframes", [])
                new_history = response.output["history"]

                st.markdown(reply)
                for frame in dataframes:
                    st.dataframe(frame, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download CSV",
                        data=frame.to_csv(index=False),
                        file_name="query_result.csv",
                        mime="text/csv",
                        key=f"dl_new_{id(frame)}",
                    )

                # Persist
                st.session_state["ama_claude_history"] = new_history
                st.session_state["ama_display"].append({
                    "role": "assistant",
                    "content": reply,
                    "dataframes": dataframes,
                })
            else:
                error_msg = f"Sorry, I couldn't answer that: {response.error}"
                st.error(error_msg)
                st.session_state["ama_display"].append({
                    "role": "assistant",
                    "content": error_msg,
                })


if __name__ == "__main__":
    main()
