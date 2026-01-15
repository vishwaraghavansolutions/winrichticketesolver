import streamlit as st
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from utils.s3storage import S3Storage
from utils.gcs_storage import GCSStorage
import utils.navbar as navbar
import pandas as pd
import numpy as np
import json
from openai import AsyncOpenAI
import asyncio
import re


client = AsyncOpenAI()
SEM = asyncio.Semaphore(25)   # limit to 5 concurrent LLM calls

async def llm_sentiment_analysis(transcript: str, llm_client):
    """
    Async LLM-based sentiment analysis.
    Returns:
        {
            "sentiment_label": "...",
            "sentiment_rationale": "...",
            "sentiment_recommendation": "..."
            "customer_request": "..."
        }
    """
    async with SEM:
        prompt = f"""
    You are an expert customer experience analyst.

    Analyze the following customer–agent conversation and produce a JSON response with:
    1. sentiment_label: "positive", "neutral", or "negative"
    2. sentiment_rationale: a clear explanation of why you chose this sentiment
    3. sentiment_recommendation: what the agent could have done to improve the sentiment
    4. customer_request: what was the customer problem/request

    Conversation:
    \"\"\"
    {transcript}
    \"\"\"

    Respond ONLY in Valid JSON with keys:
    sentiment_label, sentiment_rationale, sentiment_recommendation, customer_request.
    """
        try:
            # Example for async OpenAI/Azure client
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            raw = response.choices[0].message.content.strip()
            parsed = extract_json(raw)
            if parsed is None:
                return {
                    "sentiment_label": "neutral",
                    "sentiment_rationale": f"LLM returned invalid JSON: {repr(raw)[:200]}",
                    "sentiment_recommendation": "Unable to parse model output.",
                    "customer_request": "Unable to get customer request"
                }
            return parsed
        except Exception as e:
            return {
                "sentiment_label": "neutral",
                "sentiment_rationale": f"LLM error: {e}",
                "sentiment_recommendation": "No recommendation available.",
                "customer_request": "Unable to get customer request"
            }

# ---------------------------------------------------------
# Load CSV from GCP/S3
# ---------------------------------------------------------
def load_csv_from_gcp(storage, bucket, path):
    raw = storage.read_csv(bucket, path)
    if raw is None:
        st.error(f"CSV file not found at {path} in bucket {bucket}.")
        return pd.DataFrame()
    return raw

def preprocess_ticket_messages(df):
    # Ensure datetime parsing
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
    df["msg_datetime"] = pd.to_datetime(df["msg_datetime"], errors="coerce")

    # Sort rows within each ticket by message timestamp
    df = df.sort_values(["ticket_id", "msg_datetime"])

    # Build consolidated conversation list
    def build_conversation(group):
        return [
            {
                "msg_content": row["msg_content"],
                "message_from": row["message_from"],
                "msg_datetime": row["msg_datetime"],
                "status": row["status"],
                "posted_date": row["posted_date"],
                "closed_date": row["closed_date"],
            }
            for _, row in group.iterrows()
        ]

    # Aggregate into one row per ticket
    aggregated = (
        df.groupby("ticket_id").agg(
            customer_id=("customer_id", "first"),
            customer_name=("customer_name", "first"),
            product_name=("product_name", "first"),
            status=("status", "first"),
            ticket_opened_date=("posted_date", "min"),
            ticket_closed_date=("closed_date", "max"),
        )
    )

    aggregated["conversation"] = (
        df.groupby("ticket_id").apply(build_conversation)
    )
    aggregated = aggregated.reset_index()

    return aggregated
#----------------------------------------------------------
#    write the analytics file to s3
# ---------------------------------------------------------
def save_analytics_to_s3(storage, bucket, path, queue):
    try:
        storage.write_parquet(
            queue,
            bucket=bucket,
            path=path,
            overwrite=True,
        )
    except Exception as e:
        st.error(f"Error saving analytics parquet to S3: {e}")
        return False
# ---------------------------------------------------------
# Normalize ticket dataframe
# ---------------------------------------------------------
def normalize_ticket_dataframe(df):
    expected_cols = [
        "ticket_id", "customer_id", "customer_name", "product_name",
        "message_from", "msg_content", "msg_datetime",
        "status", "posted_date", "closed_date"
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return pd.DataFrame()

    datetime_cols = ["msg_datetime", "posted_date", "closed_date"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ---------------------------------------------------------
# Load queue resolver JSON (combined structure)
# ---------------------------------------------------------
def load_queue_resolver(storage, bucket, path):
    try:
        data = storage.read_json(bucket, path)
        return data if data else {}
    except Exception as e:
        st.error(f"Error loading queue_resolver.json: {e}")
        return {}


def enrich_with_agent_and_sla(aggregated_df, resolver):
    """
    Enrich the aggregated ticket dataframe with:
    - assigned_agent
    - sla_hours

    resolver structure:
    {
        "ProductA": {"agent": "Agent_1", "sla": 24},
        "ProductB": {"agent": "Agent_3", "sla": 48}
    }
    """

    def get_agent(product):
        entry = resolver.get(product, {})
        return entry.get("agent")

    def get_sla(product):
        entry = resolver.get(product, {})
        return entry.get("sla")

    aggregated_df["assigned_agent"] = aggregated_df["product_name"].apply(get_agent)
    aggregated_df["sla_hours"] = aggregated_df["product_name"].apply(get_sla)

    return aggregated_df

# ---------------------------------------------------------
# SLA metrics: hours_to_close + breach flag
# ---------------------------------------------------------
def compute_sla_metrics(df):
    # Ensure datetime types
    df["ticket_opened_date"] = pd.to_datetime(df["ticket_opened_date"], errors="coerce").dt.tz_localize(None)
    df["ticket_closed_date"] = pd.to_datetime(df["ticket_closed_date"], errors="coerce").dt.tz_localize(None)

    def business_hours(start, end):
        if pd.isna(start) or pd.isna(end):
            return np.nan

        # If closed before opened, return NaN
        if end < start:
            return np.nan

        # Generate business days between start and end
        bdays = pd.bdate_range(start.date(), end.date(), freq="C")

        if len(bdays) == 0:
            return 0

        # Full business days (excluding first and last)
        full_days = max(len(bdays) - 2, 0)

        # Hours on first business day
        first_day_end = pd.Timestamp.combine(bdays[0], pd.Timestamp.max.time())
        first_hours = (min(end, first_day_end) - start).total_seconds() / 3600

        # Hours on last business day
        if len(bdays) > 1:
            last_day_start = pd.Timestamp.combine(bdays[-1], pd.Timestamp.min.time())
            last_hours = (end - max(start, last_day_start)).total_seconds() / 3600
        else:
            last_hours = 0

        # Total business hours
        return max(first_hours, 0) + max(full_days * 24, 0) + max(last_hours, 0)

    # Compute business hours to close
    df["hours_to_close"] = df.apply(
        lambda row: business_hours(row["ticket_opened_date"], row["ticket_closed_date"]),
        axis=1
    )

    # SLA breach flag
    df["sla_breached"] = df["hours_to_close"] > df["sla_hours"]

    return df

def conversation_to_text(conversation):
    lines = []
    for msg in conversation:
        sender = msg.get("message_from", "unknown")
        time = msg.get("msg_datetime", "")
        content = msg.get("msg_content", "")
        lines.append(f"[{time}] {sender}: {content}")
    return "\n".join(lines)

async def apply_llm_sentiment_async(df, client):
    transcripts = df["transcript"].tolist()
    total = len(transcripts)
    batch_size = 25
    results = []
    progress = st.progress(0)

    for i in range(0, total, batch_size):
        batch = transcripts[i:i+batch_size]

        tasks = [llm_sentiment_analysis(t, client) for t in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        progress.progress(min((i + batch_size) / total, 1.0))

    df["sentiment_label"] = [r["sentiment_label"] for r in results]
    df["sentiment_rationale"] = [r["sentiment_rationale"] for r in results]
    df["sentiment_recommendation"] = [r["sentiment_recommendation"] for r in results]
    df["customer_request"] = [r["customer_request"] for r in results]

    return df
# ---------------------------------------------------------
# Agent-level sentiment aggregation
# ---------------------------------------------------------
def compute_agent_sentiment_summary_llm(df):
    return df.groupby("assigned_agent").agg(
        tickets=("ticket_id", "count"),
        positive_share=("sentiment_label", lambda x: (x == "positive").mean()),
        neutral_share=("sentiment_label", lambda x: (x == "neutral").mean()),
        negative_share=("sentiment_label", lambda x: (x == "negative").mean()),
        sample_recommendation=("sentiment_recommendation", lambda x: x.iloc[0])
    ).reset_index()    

def build_ticket_summary(df):
    summary = {}

    # Basic counts
    summary["total_tickets"] = len(df)
    summary["total_open"] = (df["status"].str.lower() != "closed").sum()
    summary["total_closed"] = (df["status"].str.lower() == "closed").sum()
    #summary["total_open"] = df["ticket_closed_date"].isna().sum()
    #summary["total_closed"] = df["ticket_closed_date"].notna().sum()

    # Percentages
    summary["pct_open"] = (
        summary["total_open"] / summary["total_tickets"] * 100
        if summary["total_tickets"] > 0 else 0
    )
    summary["pct_closed"] = (
        summary["total_closed"] / summary["total_tickets"] * 100
        if summary["total_tickets"] > 0 else 0
    )

    # SLA
    if "sla_breached" in df.columns:
        closed = df[df["status"].str.lower() == "closed"]

        summary["resolved_within_sla"] = (closed["sla_breached"] == False).sum()
        summary["resolved_beyond_sla"] = (closed["sla_breached"] == True).sum()

        #summary["resolved_within_sla"] = (df["sla_breached"] == False).sum()
        #summary["resolved_beyond_sla"] = (df["sla_breached"] == True).sum()

        total_resolved = summary["resolved_within_sla"] + summary["resolved_beyond_sla"]
        summary["pct_meeting_sla"] = (
            summary["resolved_within_sla"] / total_resolved * 100
            if total_resolved > 0 else 0
        )
    else:
        summary["resolved_within_sla"] = 0
        summary["resolved_beyond_sla"] = 0
        summary["pct_meeting_sla"] = 0

    # Resolution time
    if "hours_to_close" in df.columns:
        resolved = df.dropna(subset=["ticket_closed_date"])
        summary["avg_resolution_hours"] = resolved["hours_to_close"].mean()
        summary["median_resolution_hours"] = resolved["hours_to_close"].median()
    else:
        summary["avg_resolution_hours"] = None
        summary["median_resolution_hours"] = None

    # Sentiment breakdown (closed tickets only)
    if "sentiment_label" in df.columns:
        closed = df[df["status"].str.lower() == "closed"]
        summary["sentiment_counts"] = closed["sentiment_label"].value_counts().to_dict()
    else:
        summary["sentiment_counts"] = {}

    # Trend: last 7 days vs previous 7 days
    closed = df[df["status"].str.lower() == "closed"]
    if "ticket_closed_date" in df.columns:
        closed["ticket_closed_date"] = pd.to_datetime(df["ticket_closed_date"])
        last_7 = closed[closed["ticket_closed_date"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
        prev_7 = closed[
            (closed["ticket_closed_date"] < (pd.Timestamp.now() - pd.Timedelta(days=7))) &
            (closed["ticket_closed_date"] >= (pd.Timestamp.now() - pd.Timedelta(days=14)))
        ]
        summary["last_7_days"] = len(last_7)
        summary["prev_7_days"] = len(prev_7)
        summary["trend_delta"] = summary["last_7_days"] - summary["prev_7_days"]
    else:
        summary["last_7_days"] = 0
        summary["prev_7_days"] = 0
        summary["trend_delta"] = 0

    return summary

def compute_top_requests_by_product(df):
    if "product_name" not in df.columns or "customer_request" not in df.columns:
        return {}

    results = {}

    for product, group in df.groupby("product_name"):
        top3 = (
            group["customer_request"]
            .value_counts()
            .head(3)
            .to_dict()
        )
        results[product] = top3

    return results

def extract_json(raw):
    if not raw or not isinstance(raw, str):
        return None

    # Remove markdown fences like ```json ... ```
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z0-9]*", "", raw)   # remove ```json or ```anything
    raw = re.sub(r"```$", "", raw)               # remove closing ```
    raw = raw.strip()

    # Extract the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None
    
def fmt(value, decimals=1):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"
# ---------------------------------------------------------
# MAIN STREAMLIT APP
# ---------------------------------------------------------
def main():
    navbar.navbar()
    st.title("Ticket Analytics Dashboard")
   
    # Admin check (optional, mirror your other pages)
    if "role" not in st.session_state or st.session_state["role"] != "admin":
        st.error("Access denied. Admins only.")
        st.button("Go to Login Page", on_click=st.switch_page("login.py"))
        st.stop()

    # Storage setup
    gcs = GCSStorage(credentials_key="gcp")
    # Cloud config from secrets
    gcs_bucket = st.secrets["gcp"]["bucket"]
    gcs_path = st.secrets["gcp"]["input_path"]

    s3 = S3Storage(
        aws_access_key=st.secrets["aws"]["access_key"],
        aws_secret_key=st.secrets["aws"]["secret_key"],
        region=st.secrets["aws"].get("region", "ap-south-1"),
    )

    bucket = st.secrets["aws"]["bucket"]
    resolver_path = st.secrets["aws"]["queue_resolver_path"]
    analytics_path = st.secrets["aws"]["analytics_path"]

    # Load data
    df_raw = load_csv_from_gcp(gcs, gcs_bucket, gcs_path)

    st.write("Number of records fetched from CSV")
    st.write(len(df_raw))
    if df_raw.empty:
        st.stop()

    df = normalize_ticket_dataframe(df_raw)
    if df.empty:
        st.stop()

    ticket_agg = s3.read_parquet(bucket, analytics_path) 
    resolver = load_queue_resolver(s3, bucket, resolver_path)
    
    st.write(f"Loaded {len(ticket_agg)} existing aggregated tickets from analytics.")
    new_ticket_agg = preprocess_ticket_messages(df)
    if len(ticket_agg) > 0:
        existing_ticket_ids = set(ticket_agg["ticket_id"].tolist())
        new_tickets = new_ticket_agg[~new_ticket_agg["ticket_id"].isin(existing_ticket_ids)]
        ticket_agg = pd.concat([ticket_agg, new_tickets], ignore_index=True)
    else:
        ticket_agg = new_ticket_agg

    # Enrich with agent + SLA
    ticket_agg = enrich_with_agent_and_sla(ticket_agg, resolver)
    # SLA metrics
    ticket_agg = compute_sla_metrics(ticket_agg)
    # Build transcript
    ticket_agg["transcript"] = ticket_agg["conversation"].apply(conversation_to_text)

    #Filter: only process tickets that are NOT closed OR missing sentiment
    if len(ticket_agg) > 0 and "sentiment_label" in ticket_agg.columns:
        st.info(f"Total tickets to analyze for sentiment: {len(ticket_agg)}")
        to_process = ticket_agg[ticket_agg["sentiment_label"].isna()].copy()

        # Tickets to skip
        skipped = ticket_agg[ticket_agg["sentiment_label"].notna()]
    else:
        to_process = ticket_agg.copy()  
        skipped = pd.DataFrame(columns=ticket_agg.columns)

    #Agent sentiment summary
    if not to_process.empty:
        st.info(f"Processing {len(to_process)} tickets for LLM sentiment analysis...")
        to_process = asyncio.run(apply_llm_sentiment_async(to_process, client))

    required_cols = [
        "sentiment_label",
        "sentiment_rationale",
        "sentiment_recommendation"]

    for col in required_cols:
        if col not in skipped.columns:
            skipped[col] = None

    ticket_agg = pd.concat([to_process, skipped], ignore_index=True)
    save_analytics_to_s3(s3, bucket, analytics_path, ticket_agg)

    summary = build_ticket_summary(ticket_agg)
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Ticket Summary",
        "SLA Overview",
        "Agent Performance",
        "Product Insights",
        "Ticket Trends",
        "Sentiment by Agent",
    ])
    # -----------------------------------------------------
    # TAB 1: Ticket Summary
    # -----------------------------------------------------
    with tab1:
        st.subheader("📊 Ticket Summary")
        # Row 1 — Ticket volume
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tickets", summary["total_tickets"])
        col2.metric("Open Tickets", summary["total_open"], f"{summary['pct_open']:.1f}% open")
        col3.metric("Closed Tickets", summary["total_closed"], f"{summary['pct_closed']:.1f}% closed")

        # Row 2 — SLA performance
        col4, col5, col6 = st.columns(3)
        col4.metric("Resolved Within SLA", summary["resolved_within_sla"])
        col5.metric("Resolved Beyond SLA", summary["resolved_beyond_sla"])
        col6.metric("% Meeting SLA", f"{summary['pct_meeting_sla']:.1f}%")

        # Row 3 — Resolution time
        col7, col8 = st.columns(2)
        col7.metric("Avg Resolution Time (hrs)", fmt(summary["avg_resolution_hours"]))
        col8.metric("Median Resolution Time (hrs)", fmt(summary["median_resolution_hours"]))
        # Row 4 — Ticket trend
        col9, col10 = st.columns(2)
        col9.metric("Tickets (Last 7 Days)", summary["last_7_days"])
        col10.metric("Δ vs Previous 7 Days", summary["trend_delta"])

        # Row 5 — Sentiment breakdown
        st.subheader("😊 Sentiment Breakdown (Closed Tickets)")
        sent = summary["sentiment_counts"]

        colA, colB, colC = st.columns(3)
        colA.metric("Positive", sent.get("positive", 0))
        colB.metric("Neutral", sent.get("neutral", 0))
        colC.metric("Negative", sent.get("negative", 0))

    # -----------------------------------------------------
    # TAB 2: SLA Overview
    # -----------------------------------------------------
    with tab2:
        st.subheader("SLA Compliance Overview")

        total_tickets = len(df)
        breaches = ticket_agg["sla_breached"].sum()
        compliance = 1 - (breaches / total_tickets) if total_tickets > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tickets", total_tickets)
        col2.metric("SLA Breaches", int(breaches))
        col3.metric("Compliance Rate", f"{compliance:.1%}")

        st.markdown("### Product-wise SLA Breach Rate")
        prod_sla = ticket_agg.groupby("product_name")["sla_breached"].mean().reset_index()
        st.bar_chart(prod_sla, x="product_name", y="sla_breached")

    # -----------------------------------------------------
    # TAB 3: Agent Performance
    # -----------------------------------------------------
    with tab3:
        st.subheader("Agent Performance Dashboard")

        agent_perf = ticket_agg.groupby("assigned_agent").agg(
            tickets_handled=("ticket_id", "count"),
            avg_resolution_hours=("hours_to_close", "mean"),
            median_resolution_hours=("hours_to_close", "median"),
            sla_breach_rate=("sla_breached", "mean"),
        ).reset_index()

        st.dataframe(agent_perf)

        st.markdown("### Tickets Handled per Agent")
        st.bar_chart(agent_perf, x="assigned_agent", y="tickets_handled")

        st.markdown("### SLA Breach Rate per Agent")
        st.bar_chart(agent_perf, x="assigned_agent", y="sla_breach_rate")

    # -----------------------------------------------------
    # TAB 4: Product Insights
    # -----------------------------------------------------
    with tab4:
        st.subheader("Product Insights")

        st.markdown("### Ticket Volume by Product")
        prod_volume = ticket_agg.groupby("product_name").size().reset_index(name="tickets")
        st.bar_chart(prod_volume, x="product_name", y="tickets")

        st.markdown("### SLA Breach Heatmap (Product × Agent)")
        ticket_agg["sla_breached"] = ticket_agg["sla_breached"].astype(float)

        heatmap_df = ticket_agg.pivot_table(
            index="product_name",
            columns="assigned_agent",
            values="sla_breached",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_df, annot=True, cmap="Reds", fmt=".2f", ax=ax)
        st.pyplot(fig)

        st.subheader("📌 Top 3 Customer Requests by Product")
        top_requests = compute_top_requests_by_product(ticket_agg)

        for product, requests in top_requests.items():
            st.markdown(f"### 🛍️ {product}")
            if not requests:
                st.write("No customer request data available.")
                continue

            for req, count in requests.items():
                st.write(f"- **{req}** — {count} requests")
            st.markdown("---")

    # -----------------------------------------------------
    # TAB 5: Ticket Trends
    # -----------------------------------------------------
    with tab5:
        st.subheader("Ticket Trends")

        st.markdown("### Daily Ticket Trend")
        daily = ticket_agg.groupby(ticket_agg["ticket_opened_date"].dt.date).size().reset_index(name="tickets")
        daily = daily.rename(columns={"ticket_opened_date": "date"})
        st.line_chart(daily, x="date", y="tickets")

        st.markdown("### Weekly Ticket Trend")
        # Weekly trend (safe handling of NaT)
        ticket_agg["week"] = ticket_agg["ticket_opened_date"].dt.to_period("W")
        ticket_agg["week"] = ticket_agg["week"].apply(lambda r: r.start_time if not pd.isna(r) else None)

        weekly = ticket_agg.dropna(subset=["week"]).groupby("week").size().reset_index(name="tickets")

        st.line_chart(weekly, x="week", y="tickets")

        st.markdown("### Monthly Ticket Trend")
        # Convert to monthly period, then to timestamp (start of month)
        ticket_agg["month"] = (
                ticket_agg["ticket_opened_date"]
                .dt.to_period("M")
                .dt.to_timestamp())

        monthly = (
            ticket_agg
            .dropna(subset=["month"])
            .groupby("month")
            .size()
            .reset_index(name="tickets"))

        monthly = monthly.set_index("month")
        st.line_chart(monthly)

        #st.line_chart(monthly, x="month", y="tickets")
        
        st.markdown("### Yearly Ticket Trend")
        ticket_agg["year"] = ticket_agg["ticket_opened_date"].dt.year

        yearly = (
            ticket_agg
            .dropna(subset=["year"])
            .groupby("year")
            .size()
            .reset_index(name="tickets")
        )

        st.line_chart(yearly, x="year", y="tickets")
    # -----------------------------------------------------
    # TAB 6: Sentiment by Agent (with drill-down)
    # -----------------------------------------------------
    with tab6:
        st.subheader("Sentiment Analysis by Agent")

        st.markdown("### Agent Sentiment Summary")
        summary_df = compute_agent_sentiment_summary_llm(ticket_agg)
        st.dataframe(summary_df)

        st.markdown("### Drill into Tickets for a Selected Agent")
        if not summary_df.empty:
            selected_agent = st.selectbox(
                "Select an agent",
                options=summary_df["assigned_agent"].tolist(),
            )

            agent_tickets = ticket_agg[ticket_agg["assigned_agent"] == selected_agent].copy()

            st.dataframe(
                agent_tickets[
                    [
                        "ticket_id",
                        "product_name",
                        "status",
                        "sentiment_label",
                        "sentiment_rationale"
                    ]
                ]
            )
        else:
            st.info("No agent sentiment data available.")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()