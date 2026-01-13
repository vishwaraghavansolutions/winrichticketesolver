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
        # Take the values exactly as they appear
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
# Save queue resolver JSON
# ---------------------------------------------------------
def load_queue_resolver(storage, bucket, path):
    try:
        return storage.read_json(bucket, path)

    except Exception as e:
        st.error(f"Error saving queue_resolver.json: {e}")
        return False
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
        These assignments are used by the ticketing system to automatically route new customer tickets to the correct agent.

        **How to use this tool:**
        - Review the list of products displayed.
        - Select the appropriate agent for each product using the dropdown menus.
        - Save your assignments when finished.

        Once saved, the system will consistently route incoming tickets based on your selections, ensuring clear ownership and efficient handling.
        """)
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
    # Cloud config from secrets
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

    # -----------------------------------------------------
    # Display products
    # -----------------------------------------------------
    products = sorted(df["product_name"].dropna().unique().tolist())
    # -----------------------------------------------------
    # Build Product → Agent mapping UI (Tabular Layout)
    # -----------------------------------------------------
    st.subheader("Map Products to Agents")
    agent_names = {a["agent_id"]: a["agent_name"] for a in agents}
    resolver = {}

    st.caption("Use the dropdowns below to assign an agent to each product. This assignment will determine who handles tickets for that product.")
    # Table header
 
    cols = st.columns([2, 3])
    cols[0].markdown("**Product**")
    cols[1].markdown("**Assigned Agent**")

    for product in products:
        col1, col2 = st.columns([2, 3])

        col1.write(product)
        resolver_old = load_queue_resolver(s3, bucket, resolver_path)

        current_agent = resolver_old.get(product)
        agent_list = list(agent_names.keys())

        # Determine the index of the current agent (fallback to 0 if missing)
        default_index = agent_list.index(current_agent) if current_agent in agent_list else 0

        agent_choice = col2.selectbox(
            f"Assign agent for {product}",
            options=agent_list,
            index=default_index,
            format_func=lambda x: f"{x} — {agent_names[x]}",
            key=f"agent_for_{product}"
        )

        resolver[product] = agent_choice
    # -----------------------------------------------------
    # Save button
    # -----------------------------------------------------
    # -----------------------------------------
    # VALIDATION: Ensure every product is mapped
    # -----------------------------------------
    missing = [p for p, a in resolver.items() if not a]

    if missing:
        st.error(
            "Some products do not have an assigned agent. "
            "Please complete all assignments before saving."
        )

        st.markdown(
            "**Unassigned products:**\n" +
            "\n".join(f"- {p}" for p in missing)
        )
    else:
        if st.button("Save Product to Agent Mapping"):
            ok = save_queue_resolver(s3, bucket, resolver_path, resolver)
            if ok:
                st.success("queue_resolver.json saved to S3 successfully!")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()