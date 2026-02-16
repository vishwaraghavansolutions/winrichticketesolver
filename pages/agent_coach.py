import streamlit as st
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from utils.s3storage import S3Storage
import utils.agentnavbar as navbar
import pandas as pd
import numpy as np
from openai import OpenAI
from utils.s3storage import S3Storage
import json
import pandas as pd
from datetime import datetime, timedelta

def compute_agent_performance(df: pd.DataFrame):
    agents = {}

    for agent, group in df.groupby("assigned_agent"):
        m = {}

        m["total_tickets"] = len(group)
        m["closed_tickets"] = group["ticket_closed_date"].notna().sum()
        m["open_tickets"] = group["ticket_closed_date"].isna().sum()

        # SLA
        m["sla_breached"] = (group["sla_breached"] == True).sum()
        m["sla_met"] = (group["sla_breached"] == False).sum()
        total_resolved = m["sla_breached"] + m["sla_met"]
        m["sla_compliance"] = (
            m["sla_met"] / total_resolved * 100 if total_resolved > 0 else 0
        )

        # Resolution time
        m["avg_resolution_hours"] = group["resolution_hours"].mean()
        m["median_resolution_hours"] = group["resolution_hours"].median()

        # Sentiment
        if "sentiment_label" in group.columns:
            m["sentiment_counts"] = group["sentiment_label"].value_counts().to_dict()
            m["negative_rate"] = (
                group["sentiment_label"].eq("negative").mean() * 100
            )
        else:
            m["sentiment_counts"] = {}
            m["negative_rate"] = 0

        agents[agent] = m

    return agents

def prepare_ai_input(ticket_agg):
    ticket_agg['ticket_opened_date'] = pd.to_datetime(ticket_agg['ticket_opened_date'])
    ticket_agg['ticket_closed_date'] = pd.to_datetime(ticket_agg['ticket_closed_date'])

    cutoff_date = datetime.now() - timedelta(days=30)
    recent = ticket_agg[ticket_agg['ticket_opened_date'] >= cutoff_date]

    worst_sla = (
        recent[recent['sla_breached'] == True]
        .sort_values(by='hours_to_close', ascending=False)
        .head(3)
    )

    negative = (
        recent[recent['sentiment_label'] == 'negative']
        .sort_values(by='hours_to_close', ascending=False)
        .head(3)
    )

    focus = pd.concat([worst_sla, negative]).drop_duplicates(subset=['ticket_id'])

    ai_records = focus[[
        'ticket_id', 'customer_name', 'assigned_agent', 'hours_to_close',
        'sentiment_label', 'sla_breached', 'transcript'
    ]]

    return ai_records.to_dict(orient='records')

def generate_agent_report(ai_ticket_data):

    prompt = f"""
You are an expert customer-experience auditor. Analyze the following ticket dataset and generate a detailed performance report for the agent.

### Input Data (JSON):
{json.dumps(ai_ticket_data, indent=2)}

### Your Tasks:
1. Identify patterns across negative-sentiment tickets.
2. Explain root causes behind SLA breaches.
3. Analyze customer frustration signals from transcripts.
4. Highlight agent behavior patterns (communication gaps, tone issues, delays, etc.).
5. Produce a final summary table with:
   - Issue Category
   - Evidence from Tickets
   - Impact on Customer Experience
   - Recommended Agent Actions
   - Examples of improved communication for each issue

### Tone & Style:
- Professional
- Insight-driven
- Actionable
- Clear and structured
- Tie every recommendation to patterns in the data
"""

    return llm_call(prompt)  #
    #return response.choices[0].message["content"]

def compute_agent_sentiment_trend(df: pd.DataFrame) -> pd.DataFrame:
    if "sentiment_label" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["ticket_opened_date"] = df["ticket_opened_date"].dt.date

    trend = (
        df.groupby(["assigned_agent", "ticket_opened_date"])["sentiment_label"]
        .apply(lambda x: (x == "negative").mean() * 100)
        .reset_index(name="pct_negative")
    )

    return trend


def build_agent_coaching_prompt(agent_name: str, metrics: dict) -> str:
    sent = metrics.get("sentiment_counts", {})

    return f"""
You are a customer support performance coach.

Agent: {agent_name}

Metrics:
- Total tickets: {metrics.get("total_tickets")}
- SLA compliance: {metrics.get("sla_compliance"):.1f}%
- Avg resolution time (hrs): {metrics.get("avg_resolution_hours"):.1f}
- Negative sentiment rate: {metrics.get("negative_rate"):.1f}%
- Sentiment counts: {sent}

Write a short, constructive coaching note (1-3 bullet points) that:
- Encourages the agent
- Suggests concrete ways to improve SLA
- Uses specific, actionable language
- Avoids generic platitudes
- Include a day wise plan to demonstrate improvement
"""

def prepare_ticket_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure datetime conversion
    df["ticket_opened_date"] = pd.to_datetime(df["ticket_opened_date"], errors="coerce")
    df["ticket_closed_date"] = pd.to_datetime(df["ticket_closed_date"], errors="coerce")

    # Compute resolution hours only for closed tickets
    df["resolution_hours"] = (
        df["ticket_closed_date"] - df["ticket_opened_date"]
    ).dt.total_seconds() / 3600

    # SLA breached (example: > 24 hours)
    df["sla_breached"] = df["resolution_hours"] > 24
    return df

def generate_coaching_message(agent_name: str, metrics: dict) -> str:
    prompt = build_agent_coaching_prompt(agent_name, metrics)
    return llm_call(prompt)  # your LLM client

def llm_call(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content

def generate_coaching_message_async(agent_name: str, metrics: dict) -> str:
    prompt = build_agent_coaching_prompt(agent_name, metrics)
    return llm_call(prompt)

client = OpenAI()


navbar.agentnavbar()

st.title("Agent Coaching & Performance App")
if "role" not in st.session_state: 
    st.error("Access denied. Admins only.")
    st.button("Go to Login Page", on_click=st.switch_page("login.py"))
    st.stop()
                    
s3 = S3Storage(
    aws_access_key=st.secrets["aws"]["access_key"],
    aws_secret_key=st.secrets["aws"]["secret_key"],
    region=st.secrets["aws"].get("region", "ap-south-1"),
)
bucket = st.secrets["aws"]["bucket"]
analytics_path = st.secrets["aws"]["analytics_path"]

# Load data
ticket_agg = s3.read_parquet(bucket, analytics_path) 
st.write(f"Loaded {len(ticket_agg)} existing aggregated tickets from analytics.")
ticket_agg = prepare_ticket_df(ticket_agg)

if st.session_state["role"] == "admin":
    display_list = ticket_agg['assigned_agent'].dropna().unique().tolist()
    selected_agent = st.selectbox("Select agent to coach", display_list)
else:
    selected_agent = st.session_state["agent_id"]

agent_metrics = compute_agent_performance(ticket_agg)
trend_df = compute_agent_sentiment_trend(ticket_agg)

agents = sorted(agent_metrics.keys())
m = agent_metrics[selected_agent]
# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Tickets", m["total_tickets"])
col2.metric("SLA Compliance", f"{m['sla_compliance']:.1f}%")
col3.metric("Negative Sentiment", f"{m['negative_rate']:.1f}%")

# Trend
st.subheader("Sentiment Trend Over Time")
agent_trend = trend_df[trend_df["assigned_agent"] == selected_agent]

if not agent_trend.empty:
    agent_trend = agent_trend.sort_values("ticket_opened_date")
    st.line_chart(
        agent_trend.set_index("ticket_opened_date")["pct_negative"],
        height=200,
    )
else:
    st.write("No sentiment trend data available.")

# Coaching
st.subheader("Coaching Suggestions")
# Create two columns
col1, col2 = st.columns([1, 2])   # left = controls, right = output
coaching = ""
with col1:
    if st.button("Coach me on performance"):
        with st.spinner("Generating..."):
            coaching = generate_coaching_message(selected_agent, m)
st.write(coaching)

report = ""
with col2:
    if st.button ("Coach me on communication"):
        with st.spinner("Generating detailed report..."):
            selected_agent_tickets = ticket_agg[ticket_agg["assigned_agent"] == selected_agent].copy()
            ai_data = prepare_ai_input(selected_agent_tickets)
            report = generate_agent_report(ai_data)

st.write(report)
