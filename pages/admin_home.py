from datetime import time
import streamlit as st

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Admin Console",
    page_icon="🛠️",
    layout="wide"
)

# ---------------------------------------------------------
# Hero Section
# ---------------------------------------------------------
st.markdown("""
    <div style="padding: 20px 0; text-align: center;">
        <h1 style="margin-bottom: 0;">🛠️ Admin Control Center</h1>
        <p style="font-size: 18px; color: #666;">
            Manage queue data, agents, and product resolver mappings — all in one place.
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

if st.session_state.get("role") != "admin":
    st.error("Access denied. Admins only.")
    st.button("Go to Login Page", on_click=st.switch_page("login.py"))
    st.stop()
# ---------------------------------------------------------
# Admin Action Panels
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

# ------------------ Refresh Queue Data -------------------
with col1:
    st.markdown("### 🔄 Refresh Queue Data")
    st.write("Reload the latest ticket queue from Parquet or S3.")
    if st.button("Refresh Now", use_container_width=True):
        st.session_state["admin_page"] = "refresh_queue"
        # TODO: plug in your refresh logic
        # refresh_queue_data()
        st.success("Queue data refreshed successfully.")

# ------------------ Administer Agents --------------------
with col2:
    st.markdown("### 👥 Manage Agents")
    st.write("Add, remove, or update agent profiles and permissions.")
    if st.button("Open Agent Manager", use_container_width=True):
        st.session_state["admin_page"] = "agents"

# ------------------ Resolver Assignment ------------------
with col3:
    st.markdown("### 🧩 Product Resolver")
    st.write("Assign products to agents for ticket routing.")
    if st.button("Edit Resolver", use_container_width=True):
        st.session_state["admin_page"] = "resolver"

st.markdown("---")

# ---------------------------------------------------------
# Dynamic Navigation (Optional)
# ---------------------------------------------------------

if "admin_page" in st.session_state:

    # ------------------ Agent Manager Page ------------------
    if st.session_state["admin_page"] == "refresh_queue":
        st.header("� Refresh Queue Data")
        st.write("Reload the latest ticket queue from Parquet or S3.")

        # TODO: Replace with your agent management UI
        st.info("Agent management UI goes here.")
        st.switch_page("pages/newticketmanager.py")

    # ------------------ Agent Manager Page ------------------
    if st.session_state["admin_page"] == "agents":
        st.header("👥 Agent Manager")
        st.write("Manage agent list, roles, and assignments.")

        # TODO: Replace with your agent management UI
        st.info("Agent management UI goes here.")
        st.switch_page("pages/admin_agents.py")

    # ------------------ Resolver Editor Page ------------------
    elif st.session_state["admin_page"] == "resolver":
        st.header("🧩 Product → Agent Resolver")
        st.write("Assign each product to an agent.")

        st.switch_page("pages/queue_resolver.py")
        del st.session_state["admin_page"]
        # TODO: Replace with your resolver editor UI
        st.info("Resolver editor UI goes here.")


# Top-right logout button
logout_col = st.columns([6, 1])[1]

with logout_col:
    if st.button("🚪 Logout", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Redirect to login page
        st.switch_page("login.py")