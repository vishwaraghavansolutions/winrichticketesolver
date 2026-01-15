import streamlit as st
import pandas as pd
import json
import re
from collections import Counter
import utils.navbar as navbar

from utils.s3storage import S3Storage

# ---------------------------------------------------------
# Utility: Extract request themes from msg_content
# ---------------------------------------------------------
def extract_request_themes(df, top_n=5):
    themes = {}

    for product, group in df.groupby("product_name"):
        requests = (
            group["nature_of_customer_request"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        themes[product] = requests

    return themes


# ---------------------------------------------------------
# Load parquet from S3
# ---------------------------------------------------------
def load_parquet_from_s3(storage, bucket, path):
    raw = storage.read_parquet(bucket, path)

    if raw is None:
        st.error("Parquet file not found in S3.")
        return pd.DataFrame()

    return raw


# ---------------------------------------------------------
# Load agents JSON
# ---------------------------------------------------------
def load_agents(storage, bucket, path):
    raw = storage.read_json(bucket, path)

    if raw is None:
        st.error("agents.json not found in S3.")
        return []

    return raw


# ---------------------------------------------------------
# Save queue resolver JSON
# ---------------------------------------------------------
def save_queue_resolver(storage, bucket, path, data):
    try:
        storage.write_json(data, bucket, path, overwrite=True)
        return True
    except Exception as e:
        st.error(f"Error saving queue_resolver.json: {e}")
        return False


# ---------------------------------------------------------
# Load queue resolver JSON
# ---------------------------------------------------------
def load_queue_resolver(storage, bucket, path):
    try:
        data = storage.read_json(bucket, path)
        return data if data else {}
    except Exception as e:
        st.error(f"Error loading queue_resolver.json: {e}")
        return {}


# ---------------------------------------------------------
# MAIN STREAMLIT APP
# ---------------------------------------------------------
def main():
    navbar.navbar()

    st.title("Queue Resolver Configuration")
    st.caption("Map customer request types to agents for intelligent routing.")

    with st.expander("ℹ️ How to Use the Queue Resolver Tool"):
        st.markdown("""
        The Queue Resolver allows you to assign a responsible agent to each product.
        These assignments are used by the ticketing system to automatically route new customer tickets.

        **How to use this tool:**
        - Review the list of products displayed.
        - Select the appropriate agent for each product.
        - Set the SLA (in hours) for each product.
        - Save your assignments when finished.
        """)

    # Admin check
    if "role" not in st.session_state or st.session_state["role"] != "admin":
        st.error("Access denied. Admins only.")
        st.button("Go to Login Page", on_click=st.switch_page("login.py"))
        st.stop()

    # -----------------------------------------------------
    # Storage Adapter Setup
    # -----------------------------------------------------
    s3 = S3Storage(
        aws_access_key=st.secrets["aws"]["access_key"],
        aws_secret_key=st.secrets["aws"]["secret_key"],
        region=st.secrets["aws"].get("region", "ap-south-1"),
    )

    bucket = st.secrets["aws"]["bucket"]
    parquet_path = st.secrets["aws"]["output_path"]
    agents_path = st.secrets["aws"]["agents_path"]
    resolver_path = st.secrets["aws"]["queue_resolver_path"]

    # -----------------------------------------------------
    # Load lifecycle parquet
    # -----------------------------------------------------
    df = load_parquet_from_s3(s3, bucket, parquet_path)
    if df.empty:
        st.stop()

    # -----------------------------------------------------
    # Load agents
    # -----------------------------------------------------
    agents = load_agents(s3, bucket, agents_path)
    if not agents:
        st.stop()

    agent_names = {a["agent_id"]: a["agent_name"] for a in agents}

    # -----------------------------------------------------
    # Load existing resolver (combined structure)
    # -----------------------------------------------------
    resolver_old = load_queue_resolver(s3, bucket, resolver_path)

    # -----------------------------------------------------
    # Display products
    # -----------------------------------------------------
    products = sorted(df["product_name"].dropna().unique().tolist())

    st.subheader("Map Products to Agents & SLA")
    st.caption("Assign an agent and SLA (in hours) for each product.")

    resolver_new = {}

    # Table header
    cols = st.columns([2, 3, 2])
    cols[0].markdown("**Product**")
    cols[1].markdown("**Assigned Agent**")
    cols[2].markdown("**SLA (hrs)**")

    # -----------------------------------------------------
    # Build UI
    # -----------------------------------------------------
    for product in products:
        col1, col2, col3 = st.columns([2, 3, 2])

        col1.write(product)

        # Load previous values
        previous = resolver_old.get(product, {})
        current_agent = previous.get("agent")
        current_sla = previous.get("sla", 24)

        # Agent selection
        agent_list = list(agent_names.keys())
        default_index = agent_list.index(current_agent) if current_agent in agent_list else 0

        agent_choice = col2.selectbox(
            f"Assign agent for {product}",
            options=agent_list,
            index=default_index,
            format_func=lambda x: f"{x} — {agent_names[x]}",
            key=f"agent_for_{product}"
        )

        # SLA input
        sla_value = col3.number_input(
            f"SLA for {product}",
            min_value=1,
            max_value=240,
            value=current_sla,
            key=f"sla_for_{product}"
        )

        # Combined structure
        resolver_new[product] = {
            "agent": agent_choice,
            "sla": sla_value
        }

    # -----------------------------------------------------
    # Save button
    # -----------------------------------------------------
    st.markdown("---")

    if st.button("Save Product → Agent + SLA Mapping"):
        ok = save_queue_resolver(s3, bucket, resolver_path, resolver_new)
        if ok:
            st.success("queue_resolver.json saved to S3 successfully!")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()