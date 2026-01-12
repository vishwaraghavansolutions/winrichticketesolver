import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
from utils.s3storage import S3Storage

# ---------------------------------------------------------
# Load Datast.dataframe(sentiment_product)
# ---------------------------------------------------------

REQUIRED_QUEUE_COLUMNS = [
    "ticket_id",
    "customer_id",
    "customer_name",
    "product_name",
    "message_from",
    "msg_content",
    "msg_datetime",
    "status",
    "posted_date",
    "closed_date",
    "number_of_interactions",
    "sla_deadline",
    "hours_to_sla",
    "nature_of_customer_request",
    "sentiment",
    "sla_risk",
    "recommended_response",
    "dt_posted_date",
    "dt_msg_datetime",
    "dt_closed_date",
    "agent_id",
]

s3 = S3Storage(
    aws_access_key=st.secrets["aws"]["access_key"],
    aws_secret_key=st.secrets["aws"]["secret_key"],
    region=st.secrets["aws"].get("region", "ap-south-1"),
)
# Cloud config from secrets
bucket = st.secrets["aws"]["bucket"]
output_path = st.secrets["aws"]["output_path"]
df = s3.read_parquet(bucket, output_path)

# Ensure datetime columns are parsed
df["dt_msg_datetime"] = pd.to_datetime(df["dt_msg_datetime"])
df["dt_closed_date"] = pd.to_datetime(df["dt_closed_date"])
df["sla_deadline"] = pd.to_datetime(df["sla_deadline"])

now = datetime.now(timezone.utc)


# ---------------------------------------------------------
# SLA Breach Computation
# ---------------------------------------------------------

def compute_sla_breached(row):
    if pd.isna(row["sla_deadline"]):
        return False
    if row["status"] == "closed" and not pd.isna(row["dt_closed_date"]):
        return row["dt_closed_date"] > row["sla_deadline"]
    return now > row["sla_deadline"]

df["sla_breached"] = df.apply(compute_sla_breached, axis=1)

# ---------------------------------------------------------
# KPI Metrics
# ---------------------------------------------------------

open_tickets = df[df["status"] != "closed"]
resolved_tickets = df[df["status"] == "closed"]
now = datetime.now(timezone.utc)
last_24h = resolved_tickets[
    resolved_tickets["dt_closed_date"] >= (now - timedelta(hours=24))
]

# ---------------------------------------------------------
# Dashboard Layout
# ---------------------------------------------------------

st.title("📊 Ticket Analytics Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Open Tickets", len(open_tickets))
col2.metric("Resolved Tickets", len(resolved_tickets))
col3.metric("Resolved in Last 24 Hours", len(last_24h))

# ---------------------------------------------------------
# SLA Breach Charts
# ---------------------------------------------------------

st.header("⏱️ SLA Breach Analytics")

if st.session_state.get("role") != "admin":
    st.error("Access denied. Admins only.")
    st.button("Go to Login Page", on_click=st.switch_page("login.py"))
    st.stop()

sla_daily = (
    df.assign(date=df["dt_msg_datetime"].dt.date)
      .groupby("date")
      .agg(
          total_tickets=("ticket_id", "count"),
          breached=("sla_breached", "sum"),
      )
)

sla_daily["breach_rate"] = sla_daily["breached"] / sla_daily["total_tickets"]

st.subheader("SLA Breaches Over Time")
st.line_chart(sla_daily[["breached"]])

st.subheader("SLA Breach Rate Over Time")
st.line_chart(sla_daily[["breach_rate"]])

# ---------------------------------------------------------
# Sentiment Heatmaps
# ---------------------------------------------------------

st.header("💬 Sentiment Heatmaps")

sentiment_product = (
    df.pivot_table(
        index="product_name",
        columns="sentiment",
        values="ticket_id",
        aggfunc="count",
        fill_value=0,
    )
)

st.subheader("Sentiment by Product")
st.dataframe(sentiment_product)


sentiment_agent = (
    df.pivot_table(
        index="agent_id",
        columns="sentiment",
        values="ticket_id",
        aggfunc="count",
        fill_value=0,
    )
)

st.subheader("Sentiment by Agent")
st.dataframe(sentiment_agent)

# ---------------------------------------------------------
# Product-Level Analytics
# ---------------------------------------------------------

st.header("📦 Product-Level Analytics")

product_stats = (
    df.groupby("product_name")
      .agg(
          total_tickets=("ticket_id", "count"),
          open_tickets=("status", lambda s: (s != "closed").sum()),
          closed_tickets=("status", lambda s: (s == "closed").sum()),
          sla_breaches=("sla_breached", "sum"),
          avg_interactions=("number_of_interactions", "mean"),
      )
      .sort_values("total_tickets", ascending=False)
)

st.dataframe(product_stats)

st.subheader("Tickets by Product")
st.bar_chart(product_stats["total_tickets"])

# ---------------------------------------------------------
# Agent Leaderboard
# ---------------------------------------------------------

st.header("🏆 Agent Leaderboard")

agent_stats = (
    df.groupby("agent_id")
      .agg(
          total_tickets=("ticket_id", "count"),
          resolved=("status", lambda s: (s == "closed").sum()),
          sla_breaches=("sla_breached", "sum"),
          avg_interactions=("number_of_interactions", "mean"),
      )
)

agent_stats["resolution_rate"] = agent_stats["resolved"] / agent_stats["total_tickets"]
agent_stats = agent_stats.sort_values(
    ["resolved", "resolution_rate"], ascending=[False, False]
)

st.dataframe(agent_stats)

colA, colB = st.columns(2)
with colA:
    st.subheader("Tickets Resolved by Agent")
    st.bar_chart(agent_stats["resolved"])

with colB:
    st.subheader("SLA Breaches by Agent")
    st.bar_chart(agent_stats["sla_breaches"])

# ---------------------------------------------------------
# Drill-Down Ticket Explorer
# ---------------------------------------------------------

st.header("🔍 Ticket Drill-Down Explorer")

product_filter = st.multiselect(
    "Filter by product",
    options=sorted(df["product_name"].unique()),
)

agent_filter = st.multiselect(
    "Filter by agent",
    options=sorted(df["agent_id"].dropna().unique()),
)

status_filter = st.multiselect(
    "Filter by status",
    options=sorted(df["status"].unique()),
)

drill_df = df.copy()

if product_filter:
    drill_df = drill_df[drill_df["product_name"].isin(product_filter)]

if agent_filter:
    drill_df = drill_df[drill_df["agent_id"].isin(agent_filter)]

if status_filter:
    drill_df = drill_df[drill_df["status"].isin(status_filter)]

columns_to_show = [
    "ticket_id",
    "product_name",
    "agent_id",
    "status",
    "sentiment",
    "sla_risk",
    "dt_msg_datetime",
    "dt_closed_date",
    "number_of_interactions",
]

st.dataframe(
    drill_df[columns_to_show].sort_values("dt_msg_datetime", ascending=False),
    use_container_width=True,
)

st.subheader("Ticket Details")
st.header("🔍 Ticket Lookup")

ticket_id_input = st.text_input("Enter Ticket ID")

if ticket_id_input:
    ticket_df = drill_df[drill_df["ticket_id"].astype(str) == str(ticket_id_input)]

    if ticket_df.empty:
        st.warning("No ticket found with that ID.")
    else:
        row = ticket_df.iloc[0]  # There should only be one match

        with st.expander(f"Ticket {row['ticket_id']} — {row['product_name']} — {row['status']}", expanded=True):
            st.write(f"**Customer:** {row['customer_name']} ({row['customer_id']})")
            st.write(f"**Product:** {row['product_name']}")
            st.write(f"**Agent:** {row['agent_id']}")
            st.write(f"**Sentiment:** {row['sentiment']}")
            st.write(f"**SLA Risk:** {row['sla_risk']}")
            st.write(f"**Posted:** {row['dt_msg_datetime']}")
            st.write(f"**Closed:** {row['dt_closed_date']}")
            st.write(f"**Interactions:** {row['number_of_interactions']}")
            st.write("**Message:**")
            st.write(row["msg_content"])