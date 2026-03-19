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
        <h1 style="margin-bottom: 0;">🛠️ Agent Control Center</h1>
        <p style="font-size: 18px; color: #666;">
            Agent can review their metrics and use AI coaching help
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

if st.session_state.get("role") != "agent":
    st.error("Access denied. Agents only.")
    st.button("Go to Login Page", on_click=st.switch_page("login.py"))
    st.stop()
# ---------------------------------------------------------
# Admin Action Panels
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

# ------------------ Refresh Queue Data -------------------
# with col1:
#     #st.markdown("### 🔄 Refresh Queue Data")
#     #st.write("Reload the latest ticket queue from Parquet or S3.")
#     #if st.button("Refresh Now", use_container_width=True):
#     #    st.session_state["admin_page"] = "refresh_queue"
#     st.write(" ")


# ------------------ Administer Agents --------------------
with col1:
    st.markdown("### 👥 View My metrics")
    st.write("View Detailed metrics about my performances")
    if st.button("View Analytics", use_container_width=True):
        st.session_state["admin_page"] = "agent_analytics"

# ------------------ Resolver Assignment ------------------
with col2:
    st.markdown("### 🧩 Agent Coach ")
    st.write("Deep Dive into tickets I resolved and view AI recommendations and suggestions ")
    if st.button("Agent Coach", use_container_width=True):
        st.session_state["admin_page"] = "agent_coach"

# ------------------ Ask Me Anything ------------------
with col3:
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
    if st.session_state["admin_page"] == "agent_analytics":
        st.success("Loading agent analytics page...")
        time.sleep(3)
        del st.session_state["admin_page"]
        st.switch_page("pages/TicketAnalytics.py")

    # ------------------ Resolver Editor Page ------------------
    elif st.session_state["admin_page"] == "agent_coach":
        st.success("Loading agent coach editor...")
        time.sleep(3)
        del st.session_state["admin_page"]
        st.switch_page("pages/agent_coach.py")

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