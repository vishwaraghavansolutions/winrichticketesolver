import streamlit as st
import pandas as pd
from datetime import datetime
from utils.gcs_storage import GCSStorage
from utils.s3storage import S3Storage

# ---------------------------------------------------------
# SAFE TIMESTAMP FORMATTER
# ---------------------------------------------------------
def safe_format(ts):
    if ts is None or pd.isna(ts):
        return "Unknown"
    try:
        return ts.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Unknown"

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Ticketing Console", layout="wide")

# ---------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------
st.markdown(
    """
<style>

/* LIGHT MODE */
:root {
    --card-bg: #fafafa;
    --card-border: #e6e6e6;

    --text-primary: #1a1a1a;
    --text-secondary: #222;

    --badge-positive-bg: #e6f4ea;
    --badge-positive-text: #137333;

    --badge-neutral-bg: #e8eaed;
    --badge-neutral-text: #2a2a2a;

    --badge-negative-bg: #fce8e6;
    --badge-negative-text: #c5221f;

    --badge-sla-low-bg: #e6f4ea;
    --badge-sla-low-text: #137333;

    --badge-sla-medium-bg: #fff4ce;
    --badge-sla-medium-text: #8a6d1d;

    --badge-sla-high-bg: #fce8e6;
    --badge-sla-high-text: #c5221f;

    --bubble-customer-bg: #f1f1f1;
    --bubble-customer-text: #000;

    --bubble-agent-bg: #e0e7ef;
    --bubble-agent-text: #000;

    --bubble-meta-text: #444;

    --section-tag-bg: #d4e4ff;      /* B2 blue accent */
    --section-tag-border: #9bb8ff;
    --section-tag-text: #174ea6;
}

/* DARK MODE */
@media (prefers-color-scheme: dark) {
    :root {
        --card-bg: #1e1e1e;
        --card-border: #333;

        --text-primary: #ffffff;
        --text-secondary: #f0f0f0;

        --badge-positive-bg: #1b3a28;
        --badge-positive-text: #a6f5b8;

        --badge-neutral-bg: #2a2a2a;
        --badge-neutral-text: #f0f0f0;

        --badge-negative-bg: #3a1e1e;
        --badge-negative-text: #ffb3b3;

        --badge-sla-low-bg: #1b3a28;
        --badge-sla-low-text: #a6f5b8;

        --badge-sla-medium-bg: #3a321a;
        --badge-sla-medium-text: #ffe9a6;

        --badge-sla-high-bg: #3a1e1e;
        --badge-sla-high-text: #ffb3b3;

        --bubble-customer-bg: #2a2a2a;
        --bubble-customer-text: #ffffff;

        --bubble-agent-bg: #3a4750;
        --bubble-agent-text: #ffffff;

        --bubble-meta-text: #dcdcdc;

        --section-tag-bg: #294a7a;
        --section-tag-border: #4f7dd8;
        --section-tag-text: #e3ecff;
    }
}

/* LAYOUT + CARDS */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.card {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}

@media (prefers-color-scheme: dark) {
    .card {
        box-shadow: 0 1px 3px rgba(0,0,0,0.6);
    }
    .card:hover {
        box-shadow: 0 2px 6px rgba(0,0,0,0.75);
    }
}

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
    color: var(--text-primary);
}

/* Remove default header bars */
h1, h2, h3, h4, h5, h6 {
    border-top: none !important;
    margin-top: 0.2rem !important;
    padding-top: 0 !important;
}

/* BADGES */
.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.25rem;
}

.badge-sentiment-positive {
    background-color: var(--badge-positive-bg);
    color: var(--badge-positive-text);
}

.badge-sentiment-neutral {
    background-color: var(--badge-neutral-bg);
    color: var(--badge-neutral-text);
}

.badge-sentiment-negative {
    background-color: var(--badge-negative-bg);
    color: var(--badge-negative-text);
}

.badge-sla-low {
    background-color: var(--badge-sla-low-bg);
    color: var(--badge-sla-low-text);
}

.badge-sla-medium {
    background-color: var(--badge-sla-medium-bg);
    color: var(--badge-sla-medium-text);
}

.badge-sla-high {
    background-color: var(--badge-sla-high-bg);
    color: var(--badge-sla-high-text);
}

.sla-pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* CHAT BUBBLES */
.chat-container {
    margin-top: 0.5rem;
}

.chat-bubble {
    max-width: 80%;
    padding: 0.6rem 0.8rem;
    border-radius: 12px;
    margin-bottom: 0.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    font-size: 0.9rem;
}

.chat-bubble-customer {
    background-color: var(--bubble-customer-bg);
    color: var(--bubble-customer-text);
    border-top-left-radius: 2px;
    margin-right: auto;
}

.chat-bubble-agent {
    background-color: var(--bubble-agent-bg);
    color: var(--bubble-agent-text);
    border-top-right-radius: 2px;
    margin-left: auto;
}

.chat-meta {
    font-size: 0.7rem;
    color: var(--bubble-meta-text);
    margin-top: 0.2rem;
    text-align: right;
}

/* SECTION TAGS */
.section-tag {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: var(--section-tag-bg);
    color: var(--section-tag-text);
    border: 1px solid var(--section-tag-border);
    margin-bottom: 0.4rem;
}

.section-block {
    margin-top: 0.4rem;
    margin-bottom: 0.6rem;
}

.section-block p {
    margin: 0.1rem 0;
    color: var(--text-secondary);
}

/* TABLES */
.details-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.5rem;
}

.details-table th {
    width: 180px;
    text-align: left;
    padding: 0.6rem;
    background-color: var(--section-tag-bg);
    color: var(--section-tag-text);
    border: 1px solid var(--section-tag-border);
    border-radius: 6px;
    font-weight: 600;
    vertical-align: top;
}

.details-table td {
    padding: 0.6rem;
    border: 1px solid var(--card-border);
    vertical-align: top;
}

/* TICKET INFO BOX */
.ticket-info-box {
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# CLOUD CONFIG + LOAD DATA
# ---------------------------------------------------------
gcs = GCSStorage(credentials_key="gcp")
s3 = S3Storage(
    aws_access_key=st.secrets["aws"]["access_key"],
    aws_secret_key=st.secrets["aws"]["secret_key"],
    region=st.secrets["aws"].get("region", "ap-south-1"),
)

gcs_bucket = st.secrets["gcp"]["bucket"]
gcs_path = st.secrets["gcp"]["input_path"]

s3_bucket = st.secrets["aws"]["bucket"]
s3_path = st.secrets["aws"]["output_path"]

# Lifecycle CSV from GCS
try:
    lifecycle_df = gcs.read_csv(gcs_bucket, gcs_path)
except Exception as e:
    st.error(f"Failed to load lifecycle CSV from GCS: {e}")
    st.stop()

if "posted_date" not in lifecycle_df.columns:
    st.error("Lifecycle CSV must contain a 'posted_date' column.")
    st.stop()

lifecycle_df["posted_date"] = pd.to_datetime(lifecycle_df["posted_date"], errors="coerce")

# Parquet from S3 (aggregated / enriched data)
try:
    df = s3.read_parquet(s3_bucket, s3_path)
except Exception as e:
    st.error(f"Failed to load S3 parquet: {e}")
    st.stop()

required_fields = [
    "ticket_id",
    "customer_name",
    "nature_of_customer_request",
    "sentiment",
    "sla_risk",
    "hours_to_sla",
    "status",
    "msg_content",
    "number_of_interactions",
    "sla_deadline",
    "posted_date",
    "closed_date",
    "recommended_response",
]
for col in required_fields:
    if col not in df.columns:
        df[col] = None

if not pd.api.types.is_numeric_dtype(df["hours_to_sla"]):
    df["hours_to_sla"] = pd.to_numeric(df["hours_to_sla"], errors="coerce")

df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")

# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------
st.title("🎫 Ticketing Console")

# ---------------------------------------------------------
# FILTERS
# ---------------------------------------------------------
with st.expander("Filters", expanded=False):
    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 2])

    with col_f1:
        status_filter = st.multiselect(
            "Status",
            options=sorted(df["status"].dropna().unique().tolist()),
            default=sorted(df["status"].dropna().unique().tolist()),
        )

    with col_f2:
        sentiment_filter = st.multiselect(
            "Sentiment",
            options=sorted(df["sentiment"].dropna().unique().tolist()),
            default=sorted(df["sentiment"].dropna().unique().tolist()),
        )

    with col_f3:
        sla_risk_filter = st.multiselect(
            "SLA risk",
            options=sorted(df["sla_risk"].dropna().unique().tolist()),
            default=sorted(df["sla_risk"].dropna().unique().tolist()),
        )

    with col_f4:
        search_query = st.text_input(
            "Search",
            placeholder="Search customer, request summary, or message...",
        )

filtered_df = df[
    df["status"].isin(status_filter)
    & df["sentiment"].isin(sentiment_filter)
    & df["sla_risk"].isin(sla_risk_filter)
]

if search_query:
    q = search_query.lower()
    filtered_df = filtered_df[
        filtered_df["customer_name"].fillna("").str.lower().str.contains(q)
        | filtered_df["nature_of_customer_request"].fillna("").str.lower().str.contains(q)
        | filtered_df["msg_content"].fillna("").str.lower().str.contains(q)
    ]

if filtered_df.empty:
    st.warning("No tickets match the filters.")
    st.stop()

# ---------------------------------------------------------
# TICKET AGGREGATOR (ONE ROW PER TICKET)
# ---------------------------------------------------------
ticket_agg = (
    filtered_df.sort_values("posted_date")
    .groupby("ticket_id")
    .agg(
        {
            "customer_name": "first",
            "nature_of_customer_request": "first",
            "sentiment": "last",
            "sla_risk": "last",
            "hours_to_sla": "last",
            "status": "last",
            "msg_content": "last",
            "recommended_response": "last",
            "posted_date": "min",
            "closed_date": "max",
            "number_of_interactions": "max",
        }
    )
    .reset_index()
)

ticket_agg = ticket_agg.sort_values("posted_date").reset_index(drop=True)

if "ticket_index" not in st.session_state:
    st.session_state.ticket_index = 0

st.session_state.ticket_index = max(
    0, min(st.session_state.ticket_index, len(ticket_agg) - 1)
)

# ---------------------------------------------------------
# NAVIGATION (BY TICKET)
# ---------------------------------------------------------
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 3])

with nav_col1:
    if st.button("⬅️ Previous") and st.session_state.ticket_index > 0:
        st.session_state.ticket_index -= 1

with nav_col2:
    if st.button("Next ➡️") and st.session_state.ticket_index < len(ticket_agg) - 1:
        st.session_state.ticket_index += 1

with nav_col3:
    st.write(f"Ticket {st.session_state.ticket_index + 1} of {len(ticket_agg)}")

ticket = ticket_agg.iloc[st.session_state.ticket_index]
ticket_id = ticket["ticket_id"]

ticket_rows = (
    filtered_df[filtered_df["ticket_id"] == ticket_id]
    .sort_values("posted_date")
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# LIFECYCLE: Opened + Days Open from CSV
# ---------------------------------------------------------
csv_rows = lifecycle_df[lifecycle_df["ticket_id"] == ticket_id].sort_values("posted_date")

if not csv_rows.empty:
    first_csv_ts = csv_rows.iloc[0]["posted_date"]
else:
    first_csv_ts = None

if pd.notna(first_csv_ts):
    opened_duration = datetime.now() - first_csv_ts
    opened_days = opened_duration.days
else:
    opened_duration = None
    opened_days = "N/A"

# ---------------------------------------------------------
# TICKET DETAIL CARD
# ---------------------------------------------------------
st.markdown("<div class='card ticket-wrapper'>", unsafe_allow_html=True)

# SLA pill
hours_left = ticket["hours_to_sla"]
sla_text = ""
if pd.notna(hours_left):
    try:
        hours_left = float(hours_left)
        sla_text = f"{int(hours_left)}h left" if hours_left < 24 else f"{int(hours_left//24)}d left"
    except Exception:
        sla_text = ""

if sla_text:
    sla_class = {
        "low": "badge-sla-low",
        "medium": "badge-sla-medium",
        "high": "badge-sla-high",
    }.get((ticket["sla_risk"] or "").lower(), "badge-sla-medium")

    st.markdown(
        f"<span class='sla-pill {sla_class}'>{sla_text}</span>",
        unsafe_allow_html=True,
    )

# Ticket ID
st.markdown(f"### #{ticket_id}", unsafe_allow_html=True)

# First and latest messages (from parquet-filtered rows)
first_row = ticket_rows.iloc[0]
first_msg = first_row["msg_content"] or "No message content available."
first_ts = first_row["posted_date"]

latest_row = ticket_rows.iloc[-1]
latest_msg = latest_row["msg_content"] or "No message content available."
latest_ts = latest_row["posted_date"]

# ---------------------------------------------------------
# TABLE 1: CUSTOMER ASK ONLY
# ---------------------------------------------------------
st.markdown("<div class='section-block'>", unsafe_allow_html=True)

table_html = f"""
<table class="details-table">
    <tr>
        <th>Customer Ask</th>
        <td>
            <div class="chat-bubble chat-bubble-customer">
                <div>{first_msg}</div>
                <div class="chat-meta">Customer • {safe_format(first_ts)}</div>
            </div>
        </td>
    </tr>
</table>
"""

st.markdown(table_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TICKET INFO GRID (NO TICKET AGE, SENTIMENT + SLA RISK INCLUDED)
# ---------------------------------------------------------
st.markdown("<div class='section-block'>", unsafe_allow_html=True)
st.markdown("<span class='section-tag'>Ticket Info</span>", unsafe_allow_html=True)

st.markdown("<div class='ticket-info-box'>", unsafe_allow_html=True)

sentiment = ticket["sentiment"] or "Unknown"
sla_risk = ticket["sla_risk"] or "Unknown"
interactions = (
    int(ticket["number_of_interactions"])
    if pd.notna(ticket["number_of_interactions"])
    else len(ticket_rows)
)

# Row 1: Customer | Status | SLA Risk | Sentiment
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**Customer**")
    st.markdown(ticket["customer_name"] or "Unknown")

with c2:
    st.markdown("**Status**")
    st.markdown(ticket["status"] or "Unknown")

with c3:
    st.markdown("**SLA Risk**")
    st.markdown(sla_risk)

with c4:
    st.markdown("**Sentiment**")
    st.markdown(sentiment)

# Row 2: Opened | Days open | Interactions | (empty)
c5, c6, c7, c8 = st.columns(4)

with c5:
    st.markdown("**Opened**")
    st.markdown(safe_format(first_csv_ts))

with c6:
    st.markdown("**Days open**")
    st.markdown(opened_days)

with c7:
    st.markdown("**Interactions**")
    st.markdown(interactions)

with c8:
    st.markdown("")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Badges at bottom
sentiment_lower = (ticket["sentiment"] or "").lower()
sla_risk_lower = (ticket["sla_risk"] or "").lower()

sentiment_class = {
    "positive": "badge-sentiment-positive",
    "neutral": "badge-sentiment-neutral",
    "negative": "badge-sentiment-negative",
}.get(sentiment_lower, "badge-sentiment-neutral")

sla_badge_class = {
    "low": "badge-sla-low",
    "medium": "badge-sla-medium",
    "high": "badge-sla-high",
}.get(sla_risk_lower, "badge-sla-medium")

st.markdown(
    f"""
    <div style="margin-top:0.5rem;">
        <span class="badge {sentiment_class}">Sentiment: {ticket['sentiment'] or 'unknown'}</span>
        <span class="badge {sla_badge_class}">SLA risk: {ticket['sla_risk'] or 'unknown'}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TIMELINE VIEW (ONE COLLAPSIBLE SECTION, MESSAGES ALWAYS OPEN)
# ---------------------------------------------------------
with st.expander("Timeline", expanded=False):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    timeline_rows = lifecycle_df[lifecycle_df["ticket_id"] == ticket_id].sort_values("posted_date")

    for _, row in timeline_rows.iterrows():
        msg = row.get("msg_content", "") or ""
        ts = row["posted_date"]
        ts_str = safe_format(ts)

        st.markdown(
            f"""
            <div class="chat-bubble chat-bubble-customer">
                <div>{msg}</div>
                <div class="chat-meta">{ts_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# ADD NOTE (RESPONSE SECTION SHELL)
# ---------------------------------------------------------
st.markdown("#### Add note")
st.markdown("<div class='card'>", unsafe_allow_html=True)

if ticket["recommended_response"]:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="chat-bubble chat-bubble-agent">
            <div>{ticket["recommended_response"]}</div>
            <div class="chat-meta">AI assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Use AI recommended response"):
        st.session_state.reply_text = ticket["recommended_response"]
else:
    st.info("No AI recommended response available for this ticket.")

reply_text = st.text_area(
    "Message",
    height=150,
    key="reply_text",
)

send_col, _ = st.columns([1, 4])
with send_col:
    if st.button("Send"):
        st.success("Message added (mock).")

st.markdown("</div>", unsafe_allow_html=True)