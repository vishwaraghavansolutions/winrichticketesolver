import time
import streamlit as st

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Admin Console",
    page_icon="🛠️",
    layout="wide"
)

# st.markdown("""
# ### 📘 Getting started?
# You can read the full **Admin README** which explains the entire workflow from initial login, agent setup, importing GCP data, resolver assignment, and how agents begin working on tickets.

# 👉 **Click the link below to open the README in a new tab:**
# [Open Admin README](https://winrichticketesolver.streamlit.app/admin_read_me)
# """)

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

col1, col2, col3, col4 = st.columns(4)

# ------------------ Refresh Queue Data -------------------
# with col1:
#     #st.markdown("### 🔄 Refresh Queue Data")
#     #st.write("Reload the latest ticket queue from Parquet or S3.")
#     #if st.button("Refresh Now", use_container_width=True):
#     #    st.session_state["admin_page"] = "refresh_queue"
#     st.write(" ")


# ------------------ Administer Agents --------------------
with col1:
    st.markdown("### 👥 Manage Agents")
    st.write("Add, remove, or update agent profiles and permissions.")
    if st.button("Open Agent Manager", use_container_width=True):
        st.session_state["admin_page"] = "agents"

# ------------------ Resolver Assignment ------------------
with col2:
    st.markdown("### 🧩 Assign SLA, Agents to Products ")
    st.write("Assign products to agents for ticket routing.")
    if st.button("Edit Resolver", use_container_width=True):
        st.session_state["admin_page"] = "resolver"

# ------------------ Resolver Assignment ------------------
with col3:
    st.markdown("### 🧩 Analytics")
    st.write("View metrics and Reports.")
    if st.button("View Analytics", use_container_width=True):
        st.session_state["admin_page"] = "Analytics"

# ------------------ Ask Me Anything ------------------
with col4:
    st.markdown("### 🤖 Ask Me Anything")
    st.write("Ask plain-English questions about your ticket data.")
    if st.button("Ask Me Anything", use_container_width=True):
        st.session_state["admin_page"] = "ask_me_anything"

st.markdown("---")

# ---------------------------------------------------------
# Dynamic Navigation (Optional)
# ---------------------------------------------------------

if "admin_page" in st.session_state:

    # ------------------ Agent Manager Page ------------------
        # if st.session_state["admin_page"] == "refresh_queue":
        #     st.success("Loading refresh queue page ...")
        #     time.sleep(3)
        #     del st.session_state["admin_page"]
        #     st.switch_page("pages/newticketmanager.py")

    # ------------------ Agent Manager Page ------------------
    if st.session_state["admin_page"] == "agents":
        st.success("Loading admin page...")
        time.sleep(3)
        del st.session_state["admin_page"]
        st.switch_page("pages/admin_agents.py")

    # ------------------ Resolver Editor Page ------------------
    elif st.session_state["admin_page"] == "resolver":
        st.success("Loading resolver editor...")
        time.sleep(3)
        del st.session_state["admin_page"]

        st.switch_page("pages/queue_resolver.py")

    # ------------------ Resolver Editor Page ------------------
    elif st.session_state["admin_page"] == "Analytics":
        st.success("Loading analytics page...")
        time.sleep(3)
        del st.session_state["admin_page"]
        st.switch_page("pages/TicketAnalytics.py")

    # ------------------ Ask Me Anything ------------------
    elif st.session_state["admin_page"] == "ask_me_anything":
        st.success("Loading Ask Me Anything...")
        time.sleep(3)
        del st.session_state["admin_page"]
        st.switch_page("pages/AskAnything.py")

# Top-right logout button
logout_col = st.columns([6, 1])[1]

with logout_col:
    if st.button("🚪 Logout", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Redirect to login page
        st.switch_page("login.py")