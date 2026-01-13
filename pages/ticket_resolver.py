import pytz
import streamlit as st
import pandas as pd
from datetime import datetime, time
from utils.gcs_storage import GCSStorage
from utils.s3storage import S3Storage
from utils.TicketResolver import TicketResolver
from dateutil import parser
import re
import utils.agentnavbar as agentnavbar
import utils.navbar as navbar


# ---------------------------------------------------------
# LOAD EXTERNAL CSS FILES
# ---------------------------------------------------------
def load_css_files(paths):
    for path in paths:
        try:
            with open(path, "r") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to load CSS file {path}: {e}")

def load_lifecycle_csv():
    df = gcs.read_csv(gcs_bucket, gcs_path)
    # --- Normalize column names ---
    df.columns = df.columns.str.strip()

    return df

def safe_parse_ts(value):
    if value is None:
        return None

    if isinstance(value, pd.Series):
        if len(value) > 0:
            value = value.iloc[0]
            firstvalue = str(value).strip()
            # If already in ISO format with timezone, skip parsing
            iso_tz_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"
            if re.match(iso_tz_pattern, firstvalue):
                return parser.parse(firstvalue)
            
    # Force to string
    s = str(value).strip()
    #st.write(f"Parsing timestamp: -->'{s}'")

    # Remove hidden characters
    for bad in ["\ufeff", "\u200b", "\xa0", "\r", "\n"]:
        s = s.replace(bad, "")

    # Handle empty or null-like values
    if s in ["", "None", "nan", "NaN", "NULL"]:
        return None

    # Remove anything after "(" (e.g., "(India Standard Time)")
    if "(" in s:
        s = s.split("(")[0].strip()
        #st.write(f"Stripped parentheses: -->'{s}'")

    # Always fix GMT offsets (with or without colon)
    s = re.sub(r"GMT([+-]\d{2}):?(\d{2})", r"GMT\1:\2", s)
    s = re.sub(r"^\d+\s+", "", s)

    # Parse using dateutil
    try:
        parsed = parser.parse(s)
        return parsed
    except Exception as e:
        st.write(f"Failed to parse: {e}")
        return None
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

# Load CSS
load_css_files([
    "assets/theme.css",
    "assets/light.css",
    "assets/dark.css",
    "assets/responsive.css",
])

if "role" not in st.session_state or st.session_state["role"] != "agent" and st.session_state["role"] != "admin":   
    agentnavbar.agentnavbar
    st.error("Access denied. Agents only.")
    st.button("Go to Login Page", on_click=st.switch_page("login.py"))
    st.stop()

if st.session_state.get("role") == "admin":
    navbar.navbar()
else:
    agentnavbar.agentnavbar()

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
csv_path = st.secrets["aws"]["csv_path"]

# Load lifecycle CSV
try:
    lifecycle_df = load_lifecycle_csv()
    lifecycle_df["ticket_id"] = lifecycle_df["ticket_id"].astype(str).str.strip()

except Exception as e:
    st.error(f"Failed to load lifecycle CSV from GCS: {e}")
    st.stop()

# Load parquet
try:
    df = s3.read_parquet(s3_bucket, s3_path)
except Exception as e:
    st.error(f"Failed to load S3 parquet: {e}")
    st.stop()

# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------
st.title("🎫 Ticketing Console")
st.subheader("Welcome " + st.session_state.get("agent_name", "Agent"))

agent_id = st.session_state.get("agent_id")
st.write(f"Agent ID: {agent_id}")
resolver_path = st.secrets["aws"]["queue_resolver_path"]
resolver = s3.read_json(s3_bucket, resolver_path)

# Filter products assigned to this agent
assigned_products = [
    product for product, assigned_agent in resolver.items()
    if assigned_agent == agent_id
]

if not assigned_products:
    st.info("No products have been assigned to you.")
else:
    product_list = ", ".join(assigned_products)
    st.markdown(f"**Tickets from the Products are assigned to you: [** {product_list} ]")
# ---------------------------------------------------------
# FILTERS
# ---------------------------------------------------------
with st.expander("Filters", expanded=False):
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        sentiment_filter = st.multiselect(
            "Sentiment",
            sorted(df["sentiment"].dropna().unique()),
            sorted(df["sentiment"].dropna().unique()),
        )

    with col2:
        sla_filter = st.multiselect(
            "SLA Risk",
            sorted(df["sla_risk"].dropna().unique()),
            sorted(df["sla_risk"].dropna().unique()),
        )

    with col3:
        search_query = st.text_input("Search", placeholder="Search customer or message...")

filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["product_name"].isin(assigned_products)]
filtered_df = filtered_df[filtered_df["status"] != "closed"]

#if status_filter:
#    filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]

if sentiment_filter:
    filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_filter)]

if sla_filter:
    filtered_df = filtered_df[filtered_df["sla_risk"].isin(sla_filter)]

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

if "processed_tickets" not in st.session_state:
    st.session_state["processed_tickets"] = set()
else:
    filtered_df = filtered_df[~filtered_df["ticket_id"].isin(st.session_state["processed_tickets"])]
    if filtered_df.empty:
        st.info("All tickets have been processed.")
        st.stop()

# ---------------------------------------------------------
# AGGREGATE TICKETS
# ---------------------------------------------------------
ticket_agg = (
    filtered_df.sort_values("dt_posted_date")
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
            "dt_posted_date": "min",
            "dt_closed_date": "max",
            "number_of_interactions": "max",
        }
    )
    .reset_index()
)

ticket_agg = ticket_agg.sort_values("dt_posted_date").reset_index(drop=True)

# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------
if "ticket_index" not in st.session_state:
    st.session_state.ticket_index = 0

col_prev, col_next, col_info = st.columns([1, 1, 3])

with col_prev:
    if st.button("⬅️ Previous") and st.session_state.ticket_index > 0:
        st.session_state.ticket_index -= 1

with col_next:
    if st.button("Next ➡️") and st.session_state.ticket_index < len(ticket_agg) - 1:
        st.session_state.ticket_index += 1

with col_info:
    st.write(f"Ticket {st.session_state.ticket_index + 1} of {len(ticket_agg)}")

ticket = ticket_agg.iloc[st.session_state.ticket_index]
ticket_id = ticket["ticket_id"]

ticket_rows = (
    filtered_df[filtered_df["ticket_id"] == ticket_id]
    .sort_values("posted_date")
    .reset_index(drop=True)
)

# ---------------------------------------------------------
# LIFECYCLE TIMESTAMP + DAYS OPEN
# ---------------------------------------------------------
csv_rows = lifecycle_df[lifecycle_df["ticket_id"] == ticket_id].sort_values("posted_date")
conv_posted_date = safe_parse_ts(csv_rows.iloc[0]["posted_date"]) if not csv_rows.empty else None
first_csv_ts = conv_posted_date.replace(tzinfo=None) if not csv_rows.empty else None
opened_days = (datetime.now() - first_csv_ts).days if pd.notna(first_csv_ts) else "N/A"

# ---------------------------------------------------------
# CUSTOMER ASK
# ---------------------------------------------------------
st.markdown(f"### #{ticket_id}", unsafe_allow_html=True)
#st.markdown("<div class='card'>", unsafe_allow_html=True)

first_row = ticket_rows.iloc[0]
first_msg = first_row["msg_content"] or "No message"
first_ts = first_row["posted_date"]

st.markdown(
    f"""
    <table class="details-table">
        <tr>
            <th>Customer Ask</th>
            <td>
                <div class="chat-bubble chat-bubble-customer">
                    <div>{first_msg}</div>
                </div>
            </td>
        </tr>
    </table>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr></hr>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TICKET INFO
# ---------------------------------------------------------
#st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<span class='section-tag'>Ticket Info</span>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("**Customer**")
    st.markdown(ticket["customer_name"] or "Unknown")

with c2:
    st.markdown("**Status**")
    st.markdown(ticket["status"] or "Unknown")

with c3:
    st.markdown("**SLA Risk**")
    st.markdown(ticket["sla_risk"] or "Unknown")

with c4:
    st.markdown("**Sentiment**")
    st.markdown(ticket["sentiment"] or "Unknown")

c5, c6, c7, c8 = st.columns(4)

with c5:
    st.markdown("**Opened**")
    st.markdown(safe_format(first_csv_ts))

with c6:
    st.markdown("**Days Open**")
    st.markdown(opened_days)

with c7:
    st.markdown("**Interactions**")
    st.markdown(ticket["number_of_interactions"] or len(ticket_rows))

with c8:
    st.write("")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# TIMELINE (FROM LIFECYCLE CSV)
# ---------------------------------------------------------
with st.expander("Timeline", expanded=False):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for _, row in csv_rows.iterrows():

        msg = row.get("msg_content", "") or ""
        ts = str(row["posted_date"])
        msg_from = row.get("message_from", "customer").lower()
        message_text = msg_from + " : " + ts

        st.markdown(
            f"""
            <div class="chat-bubble chat-bubble-customer">
                <div>{msg}</div>
                <div class="chat-meta" style="text-align: right; font-family: Arial; color: #670;"><small>{message_text}</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# RESPONSE SECTION
# ---------------------------------------------------------
resolver = TicketResolver(gcs, s3)
st.markdown("### Add Comment")
st.write("AI Recommended Response: ", ticket["recommended_response"])
default_value = ""
if st.session_state.get("Use_AI_Recommendation", False):
    if "comment_text" not in st.session_state or not st.session_state["comment_text"]:
        default_value = " "
    else:
        default_value = st.session_state["comment_text"]
    st.session_state["Use_AI_Recommendation"] = False

with st.form("comment_form"):
    comment_text = st.text_area(
        "Write your comment",
        height=120,
        value=default_value,
        key="comment_text_area"
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        post_and_resolve = st.form_submit_button("Post Comment & Resolve", type="primary")

    with col2:
        ai_recommend = st.form_submit_button("Post AI Recommended Text", type="secondary")

    with col3:
        post_only = st.form_submit_button("Post Comment & Keep open")


# -------------------------
# Now handle the actions
# -------------------------

if ai_recommend:
    st.session_state["Use_AI_Recommendation"] = True
    lifecycle_df = resolver.append_entry(lifecycle_df, ticket_id, ticket["recommended_response"], status="open")
    cols_to_drop = ["dt_msg_datetime", "dt_posted_date", "dt_closed_date"]
    lifecycle_df = lifecycle_df.drop(columns=[c for c in cols_to_drop if c in lifecycle_df.columns])
    st.write("Updated CSV:")
    st.write(lifecycle_df.tail(5))
    gcs.write_csv(lifecycle_df, gcs_bucket, gcs_path, overwrite=True)
    st.success("Comment posted")
    st.write(lifecycle_df.tail(5))
elif post_only:
    st.write("Posting comment only")
    st.write(f"Comment text:'{comment_text}'")
    if comment_text.strip():
        lifecycle_df = resolver.append_entry(lifecycle_df, ticket_id, comment_text, status="open")
        st.write("Updated CSV:")
        st.write(lifecycle_df.tail(5))
        gcs.write_csv(lifecycle_df, gcs_bucket, gcs_path, overwrite=True)
        s3.write_csv(lifecycle_df, s3_bucket, csv_path, overwrite=True)
        st.session_state["comment_text"] = comment_text
        st.success("Comment posted")
        st.write(lifecycle_df.tail(5))
    else:
        st.warning("Please enter a comment before posting")

elif post_and_resolve:
    if not comment_text.strip():
        st.warning("Please enter a comment before resolving")
    else:
        lifecycle_df = resolver.append_entry(lifecycle_df, ticket_id, comment_text, status="closed")
        cols_to_drop = ["dt_msg_datetime", "dt_posted_date", "dt_closed_date"]
        lifecycle_df = lifecycle_df.drop(columns=[c for c in cols_to_drop if c in lifecycle_df.columns])
        lifecycle_df.loc[lifecycle_df["ticket_id"] == ticket_id, "status"] = "closed"
        filtered_df.loc[filtered_df["ticket_id"] == ticket_id, "status"] = "closed"

        st.write("Updated CSV:")
        st.write(lifecycle_df.tail(5))
        gcs.write_csv(lifecycle_df, gcs_bucket, gcs_path, overwrite=True)
        s3.write_csv(lifecycle_df, s3_bucket, csv_path, overwrite=True)
        
        df = df[~df["ticket_id"].isin(filtered_df["ticket_id"])]
        # Step 2: append the updated rows
        df = pd.concat([df, filtered_df], ignore_index=True)

        s3.write_parquet(df, s3_bucket, s3_path, overwrite=True)

        st.session_state["processed_tickets"].add(ticket_id)
        ticket_agg = ticket_agg[ticket_agg["ticket_id"] != ticket_id].reset_index(drop=True)
        st.session_state.ticket_index = max(0, st.session_state.ticket_index - 1)
        st.session_state["comment_text"] = comment_text
        st.success("Ticket resolved")
        st.rerun()

logout_col = st.columns([6, 1])[1]

with logout_col:
    if st.button("🚪 Logout", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Redirect to login page
        st.switch_page("login.py")