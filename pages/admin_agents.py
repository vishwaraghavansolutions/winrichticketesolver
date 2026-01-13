import json
from time import time
import pandas as pd
import streamlit as st
from utils.s3storage import S3Storage
import utils.navbar as navbar

# =====================================================
# S3 JSON LOAD / SAVE HELPERS
# =====================================================

def load_agents_json(storage, bucket: str, path: str) -> pd.DataFrame:
    """
    Load agents from JSON in S3.
    Returns a DataFrame with columns:
    ['agent_id', 'agent_name', 'function', 'created_at']
    """
    try:
        raw = storage.read_json(bucket, path)
    except Exception as e:
        st.error(f"Error reading agents file from S3: {e}")
        return pd.DataFrame(columns=["agent_id", "agent_name", "function", "created_at"])

    if not raw:
        return pd.DataFrame(columns=["agent_id", "agent_name", "function", "created_at"])

    if not isinstance(raw, list):
        st.error("Agents JSON is not a list; resetting to empty.")
        return pd.DataFrame(columns=["agent_id", "agent_name", "function", "created_at"])

    df = pd.DataFrame(raw)

    # Ensure schema
    for col in ["agent_id", "agent_name", "function", "created_at"]:
        if col not in df.columns:
            df[col] = ""

    return df[["agent_id", "agent_name", "function", "created_at"]]

def save_agents_json(storage, bucket: str, path: str, agents_df: pd.DataFrame):
    """
    Save agents DataFrame as JSON to S3.
    """
    cols = ["agent_id", "agent_name", "function", "created_at"]
    for col in cols:
        if col not in agents_df.columns:
            agents_df[col] = ""

    agents_df = agents_df[cols]

    data = agents_df.to_dict(orient="records")
    json_bytes = json.dumps(data, indent=2).encode("utf-8")

    try:
        storage.write_json(data, bucket, path, overwrite=True)
    except Exception as e:
        st.error(f"Error writing agents JSON to S3: {e}")


# =====================================================
# INTERNAL HANDLERS
# =====================================================

def handle_create_agent(agents_df, storage, bucket, path, agent_id, agent_name, agent_function):
    agent_id = agent_id.strip()
    agent_name = agent_name.strip()
    agent_function = agent_function.strip()

    if not agent_id or not agent_name or not agent_function:
        st.error("All fields are required.")
        return

    if agent_id in agents_df["agent_id"].astype(str).values:
        st.error(f"Agent ID '{agent_id}' already exists.")
        return

    new_row = pd.DataFrame([{
        "agent_id": agent_id,
        "agent_name": agent_name,
        "function": agent_function,
        "created_at": pd.Timestamp.now().isoformat()
    }])

    updated_df = pd.concat([agents_df, new_row], ignore_index=True)
    save_agents_json(storage, bucket, path, updated_df)

    st.success(f"Agent '{agent_name}' created successfully.")
    st.rerun()


def handle_remove_agent(agents_df, storage, bucket, path, selected_value):
    agent_id_remove = selected_value.split(" — ")[0].strip()

    filtered_df = agents_df[agents_df["agent_id"] != agent_id_remove].reset_index(drop=True)

    save_agents_json(storage, bucket, path, filtered_df)

    st.success(f"Agent '{selected_value}' removed successfully.")
    st.rerun()


# =====================================================
# MAIN ADMIN SCREEN
# =====================================================

def admin_manage_agents(storage, bucket: str, path: str):
    st.title("Admin: Manage Agents")

    agents_df = load_agents_json(storage, bucket, path)

    # -----------------------------
    # CREATE AGENT FORM
    # -----------------------------
    with st.form("create_agent_form"):
        st.subheader("Create New Agent")

        agent_id = st.text_input("Agent ID", placeholder="e.g., AGT-001")
        agent_name = st.text_input("Agent Name", placeholder="e.g., Priya Sharma")
        agent_function = st.text_input("Function / Role", placeholder="e.g., Customer Support")

        create_btn = st.form_submit_button("Create Agent", type="primary")

    if create_btn:
        handle_create_agent(agents_df, storage, bucket, path, agent_id, agent_name, agent_function)

    st.markdown("---")

    # -----------------------------
    # REMOVE AGENT
    # -----------------------------
    st.subheader("Remove Agent")

    if len(agents_df) == 0:
        st.info("No agents available.")
    else:
        display_list = agents_df["agent_id"] + " — " + agents_df["agent_name"]
        selected = st.selectbox("Select agent to remove", display_list)

        if st.button("Remove Selected Agent", type="secondary"):
            handle_remove_agent(agents_df, storage, bucket, path, selected)

    st.markdown("---")

    # -----------------------------
    # SHOW CURRENT AGENTS
    # -----------------------------
    st.subheader("Current Agents")
    st.dataframe(agents_df.sort_values("agent_id"), use_container_width=True)

    # -----------------------------
    # REMOVE PASSWORD
    # -----------------------------
    st.subheader("Remove Password")

    if len(agents_df) == 0:
        st.info("No agents available.")
    else:
        display_list = [
            f"{data['agent_id']} — {data['agent_name']}"
            for _, data in agents_df.iterrows()
        ]

    selected = st.selectbox("Select agent to clear password", display_list)

    if st.button("Remove Password", type="secondary"):
        agent_id = selected.split(" — ")[0]
        st.write(f"Removing password for agent: {agent_id}")
        st.write(agents_df)
        # Remove the password field entirely
        agents_df.loc[agents_df["agent_id"] == agent_id, "password"] = None

        # Save back to storage
        save_agents_json(storage, bucket, path, agents_df)
        
        st.success(f"Password removed for agent: {agent_id}")
# =====================================================
# STREAMLIT ENTRY POINT
# =====================================================

def main():
    # Replace with your actual storage adapter

    navbar.navbar()
    s3 = S3Storage(aws_access_key=st.secrets["aws"]["access_key"],
                    aws_secret_key=st.secrets["aws"]["secret_key"],
                    region=st.secrets["aws"].get("region", "ap-south-1"),)

    bucket = st.secrets["aws"]["bucket"]    
    agents_path = st.secrets["aws"]["agents_path"]

    admin_manage_agents(s3, bucket, agents_path)


if __name__ == "__main__":
    if "role" not in st.session_state or st.session_state["role"] != "admin":
        st.error("Access denied. Admins only.")
        st.button("Go to Login Page", on_click=st.switch_page("login.py"))
        st.stop()
    else:
        main()