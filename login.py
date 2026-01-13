import streamlit as st
import json
import hashlib
from utils.s3storage import S3Storage
import time
# ---------------------------------------------------------
# Password encoding
# ---------------------------------------------------------
def encode_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()
# ---------------------------------------------------------
# Load agents JSON
# ---------------------------------------------------------
def load_agents(storage, bucket, agents_path):
    raw = storage.read_json(bucket, agents_path)
    return raw

# ---------------------------------------------------------
# Save agents JSON
# ---------------------------------------------------------
def save_agents(storage, bucket, agents_path, agents):
    storage.write_json(agents, bucket, agents_path, overwrite=True)
    return
# ---------------------------------------------------------
# LOGIN PAGE
# ---------------------------------------------------------
st.title("Login")
s3 = S3Storage(
    aws_access_key=st.secrets["aws"]["access_key"],
    aws_secret_key=st.secrets["aws"]["secret_key"],
    region=st.secrets["aws"].get("region", "ap-south-1"),
)
s3_bucket = st.secrets["aws"]["bucket"]
agents_path = st.secrets["aws"]["agents_path"]  


admin_encoded_password = st.secrets["admin_encoded_password"]
# -----------------------------------------------------
# ADMIN LOGIN
# -----------------------------------------------------

# -------------------------------
# PHASE 1 — MAIN LOGIN FORM
# -------------------------------
with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")

if submitted:
    # -------------------------------
    # ADMIN LOGIN
    # -------------------------------
    if username == "admin":
        if encode_password(password) == admin_encoded_password:
            st.success("Admin login successful")
            st.session_state["role"] = "admin"
            st.switch_page("pages/admin_home.py")
        else:
            st.error("Invalid admin credentials")
            st.stop()

    # -------------------------------
    # AGENT LOGIN
    # -------------------------------
    else:
        agent_id = username

        agents = load_agents(s3, s3_bucket, agents_path)
        agent = next((a for a in agents if a["agent_id"] == agent_id), None)

        if agent is None:
            st.error("Agent ID not found")
            st.stop()

        # -------------------------------
        # AGENT HAS NO PASSWORD → MOVE TO PHASE 2
        # -------------------------------
        if "password" not in agent or not agent["password"] or agent["password"] == "":
            st.info("No password found. Please set a new password.")

            # Store agent info in session so next form can use it
            st.session_state["pending_agent"] = agent_id
            st.session_state["pending_agents_list"] = agents
            st.session_state["agent_name"] = agent["agent_name"]
        else:
            # -------------------------------
            # AGENT HAS PASSWORD → AUTHENTICATE
            # -------------------------------
            if encode_password(password) == agent["password"]:
                st.success("Login successful")
                st.session_state["role"] = "agent"
                st.session_state["agent_id"] = agent["agent_id"]
                st.session_state["agent_name"] = agent["agent_name"]
                time.sleep(3)
                st.switch_page("pages/ticket_resolver.py")
            else:
                st.error("Invalid password")
                st.stop()

# -------------------------------
# PHASE 2 — PASSWORD SETUP FORM
# -------------------------------
if "pending_agent" in st.session_state:

    st.info(f"Set a password for Agent ID: {st.session_state['pending_agent']}")

    with st.form("set_password_form"):
        pw1 = st.text_input("New Password", type="password")
        pw2 = st.text_input("Confirm Password", type="password")
        set_pw = st.form_submit_button("Set Password")

    if set_pw:
        if pw1 != pw2:
            st.error("Passwords do not match")
            st.stop()

        # Update password
        agents = st.session_state["pending_agents_list"]
        agent_id = st.session_state["pending_agent"]

        for a in agents:
            if a["agent_id"] == agent_id:
                a["password"] = encode_password(pw1)
                agent_name = a["agent_name"]

        save_agents(s3, s3_bucket, agents_path, agents)

        # Cleanup
        st.session_state["agent_id"] = agent_id 
        st.session_state["agent_name"] = agent_name
        st.session_state["role"] = "agent"
        del st.session_state["pending_agent"]
        del st.session_state["pending_agents_list"]
   
        st.success("Password set successfully. Please log in again.")
        st.stop()