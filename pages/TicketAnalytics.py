import streamlit as st
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt
from utils.s3storage import S3Storage
from utils.gcs_storage import GCSStorage
import utils.navbar as navbar
import utils.agentnavbar as agentnavbar
import pandas as pd
import numpy as np
import json
from openai import AsyncOpenAI
from st_aggrid import AgGrid, GridOptionsBuilder
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
    df = df.copy()
    
    # Parse posted_date
    df["posted_date"] = df["posted_date"].astype(str).str.replace(r"\s*\(.*\)$", "", regex=True)
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    
    # Parse closed_date
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
    
    # Parse msg_datetime with multiple format attempts
    if "msg_datetime" in df.columns:
        def parse_msg_datetime(dt_str):
            """Try multiple formats to parse datetime"""
            if pd.isna(dt_str) or str(dt_str).strip() in ['', 'None', 'nan', 'NaN']:
                return pd.NaT
            
            dt_str = str(dt_str).strip()
            
            # Try specific formats
            formats = [
                "%d-%m-%Y %H:%M",      # 05-11-2025 13:22
                "%d-%m-%Y %H:%M:%S",   # 05-11-2025 13:22:30
                "%d/%m/%Y %H:%M",      # 05/11/2025 13:22
                "%Y-%m-%d %H:%M:%S",   # 2025-11-05 13:22:30
                "%d-%m-%Y",            # 05-11-2025
            ]
            
            for fmt in formats:
                try:
                    return pd.to_datetime(dt_str, format=fmt)
                except:
                    continue
            
            # Try automatic parsing as last resort
            try:
                return pd.to_datetime(dt_str, dayfirst=True)
            except:
                return pd.NaT
        
        # Apply parsing
        df["msg_datetime"] = df["msg_datetime"].apply(parse_msg_datetime)
        
        # Report results
        success_rate = (df["msg_datetime"].notna().sum() / len(df)) * 100
        
        if success_rate < 90:
            failed_samples = df[df["msg_datetime"].isna()]["msg_datetime"].head(5)
            st.warning(f"⚠️ Some dates failed to parse. Sample failures: {failed_samples.to_list()}")
    else:
        st.error("❌ msg_datetime column not found in DataFrame!")
        df["msg_datetime"] = pd.NaT
    
    # Sort using msg_datetime, fallback to posted_date
    df["_sort_key"] = df["msg_datetime"].fillna(df["posted_date"])
    df = df.sort_values(["ticket_id", "_sort_key"])
    
    # Build conversation with proper datetime handling
    def build_conversation(group):
        conversations = []
        for _, row in group.iterrows():
            # Use msg_datetime if available, otherwise use posted_date
            display_datetime = row["msg_datetime"] if pd.notna(row["msg_datetime"]) else row["posted_date"]
            
            conversations.append({
                "msg_content": row["msg_content"],
                "message_from": row["message_from"],
                "msg_datetime": display_datetime,  # This will now have a value
                "status": row["status"],
                "posted_date": row["posted_date"],
                "closed_date": row["closed_date"],
            })
        return conversations
    
    # Aggregate
    aggregated = df.groupby("ticket_id", observed=True).agg(
        customer_id=("customer_id", "first"),
        customer_name=("customer_name", "first"),
        product_name=("product_name", "first"),
        status=("status", "first"),
        ticket_opened_date=("posted_date", "min"),
        ticket_closed_date=("closed_date", "max"),
    )
    
    aggregated["conversation"] = df.groupby("ticket_id", observed=True).apply(
        build_conversation, 
        include_groups=False
    )
    
    aggregated = aggregated.reset_index()
    
    # Clean up temporary column if it exists
    if "_sort_key" in df.columns:
        df.drop(columns=["_sort_key"], inplace=True)
    
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
        closed = df[df["ticket_closed_date"].notna()].copy()
        closed["ticket_closed_date"] = pd.to_datetime(closed["ticket_closed_date"])
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

def build_grid_options(df, display_cols):
    gb = GridOptionsBuilder.from_dataframe(df[display_cols])
    gb.configure_default_column(resizable=True, flex=1)
    gb.configure_selection('single')
    return gb.build()

def identify_tickets_needing_analysis(
    ticket_agg: pd.DataFrame,
    s3, s3_bucket: str, analytics_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare current tickets with S3 analytics and identify which need sentiment analysis.
    Returns: (tickets_to_process, tickets_to_skip)
    """
    
    required_cols = ["sentiment_label", "sentiment_rationale", "sentiment_recommendation", "customer_request"]
    
    # Load existing analytics from S3
    try:
        existing_agg = s3.read_parquet(s3_bucket, analytics_path)
        #st.info(f"Loaded {len(existing_agg)} existing tickets from S3 analytics.")
    except FileNotFoundError:
        st.info("No existing analytics found in S3. Will process all tickets.")
        existing_agg = pd.DataFrame()
    except Exception as e:
        st.warning(f"Failed to load S3 analytics: {str(e)}. Processing all tickets.")
        existing_agg = pd.DataFrame()
    
    # Initialize columns if they don't exist
    for col in required_cols:
        if col not in ticket_agg.columns:
            ticket_agg[col] = None
    
    if existing_agg.empty:
        # No existing data - process all tickets
        to_process = ticket_agg.copy()
        skipped = pd.DataFrame(columns=ticket_agg.columns)
        st.info(f"Processing all {len(to_process)} tickets (no existing analytics).")
    else:
        # Ensure required columns exist in existing data
        for col in required_cols:
            if col not in existing_agg.columns:
                existing_agg[col] = None
        
        # Create a hashable representation of ticket content for comparison
        def create_content_hash(row):
            """Create a hash of the ticket content fields to detect changes"""
            # Adjust these fields based on what constitutes a "change" in your data
            content_fields = ['ticket_id', 'message', 'subject', 'status']  
            content = '|'.join([str(row.get(field, '')) for field in content_fields if field in row.index])
            return hash(content)
        
        # Add content hash to both dataframes
        ticket_agg['_content_hash'] = ticket_agg.apply(create_content_hash, axis=1)
        existing_agg['_content_hash'] = existing_agg.apply(create_content_hash, axis=1)
        
        # Create lookup for existing tickets
        existing_lookup = existing_agg.set_index('ticket_id')[[
            '_content_hash', 'sentiment_label', 'sentiment_rationale', 'sentiment_recommendation', 'customer_request'
        ]].to_dict('index')
        
        # Identify tickets that need processing
        tickets_to_process_mask = []
        
        for idx, row in ticket_agg.iterrows():
            ticket_id = row['ticket_id']
            
            if ticket_id not in existing_lookup:
                # New ticket - needs processing
                tickets_to_process_mask.append(True)
            else:
                existing_ticket = existing_lookup[ticket_id]
                
                # Check if content has changed
                content_changed = row['_content_hash'] != existing_ticket['_content_hash']
                
                # Check if sentiment fields are missing
                sentiment_missing = (
                    pd.isna(existing_ticket.get('sentiment_label')) or
                    pd.isna(existing_ticket.get('sentiment_rationale')) or
                    pd.isna(existing_ticket.get('sentiment_recommendation'))
                )
                
                needs_processing = content_changed or sentiment_missing
                tickets_to_process_mask.append(needs_processing)
                
                # If not processing, carry forward existing sentiment data
                if not needs_processing:
                    ticket_agg.at[idx, 'sentiment_label'] = existing_ticket.get('sentiment_label')
                    ticket_agg.at[idx, 'sentiment_rationale'] = existing_ticket.get('sentiment_rationale')
                    ticket_agg.at[idx, 'sentiment_recommendation'] = existing_ticket.get('sentiment_recommendation')
                    ticket_agg.at[idx, 'customer_request'] = existing_ticket.get('customer_request')
        
        to_process = ticket_agg[tickets_to_process_mask].copy()
        skipped = ticket_agg[~pd.Series(tickets_to_process_mask)].copy()
        
        # Remove temporary hash column
        to_process = to_process.drop(columns=['_content_hash'])
        skipped = skipped.drop(columns=['_content_hash'])
        ticket_agg = ticket_agg.drop(columns=['_content_hash'])
        
        # st.success(f"Found {len(to_process)} tickets needing analysis:")
        # st.write(f"  - New tickets: {len(to_process[~to_process['ticket_id'].isin(existing_lookup.keys())])}")
        # st.write(f"  - Changed tickets: {len(to_process[to_process['ticket_id'].isin(existing_lookup.keys())])}")
        # st.write(f"  - Skipping {len(skipped)} unchanged tickets with existing sentiment.")
    
    return to_process, skipped, ticket_agg

def sla_stats_by_product_and_request(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean and median SLA hours grouped by:
    - product_name
    - customer_request (nature of request)
    """

    # Ensure required columns exist
    required = ["product_name", "customer_request", "hours_to_close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Group and aggregate
    stats = (
        df.groupby(["product_name", "customer_request"])
          .agg(
              mean_sla_hours=("hours_to_close", "mean"),
              median_sla_hours=("hours_to_close", "median"),
              ticket_count=("hours_to_close", "count")
          )
          .reset_index()
          .sort_values(["product_name", "customer_request"])
    )

    # Group and aggregate
    pstats = (
        df.groupby(["product_name"])
          .agg(
              mean_sla_hours=("hours_to_close", "mean"),
              median_sla_hours=("hours_to_close", "median"),
              ticket_count=("hours_to_close", "count")
          )
          .reset_index()
          .sort_values(["product_name"])
    )

    return stats, pstats
# ---------------------------------------------------------
# MAIN STREAMLIT APP
# ---------------------------------------------------------
def main():

    st.title("Ticket Analytics Dashboard")
   
    # Admin check (optional, mirror your other pages)
    if "role" not in st.session_state:
        st.write("Session expired. Please log in again.")
        st.button("Go to Login Page", on_click=st.switch_page("login.py"))
        st.stop()

    if st.session_state["role"] == "admin":
        st.success(f"Logged in as Admin ") 
        navbar.navbar()
    else:
        agentnavbar.agentnavbar()
        st.success(f"Logged in as Agent: {st.session_state['agent_id']}")
 
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

    # Load raw data from GCP
    try:
        df_raw = load_csv_from_gcp(gcs, gcs_bucket, gcs_path)
        st.write(f"Number of records fetched from CSV: {len(df_raw)}")
        if df_raw.empty:
            st.warning("No raw ticket data found in GCS.")
            st.stop()            
    except Exception as e:
        st.error(f"Failed to load CSV from GCP: {str(e)}")
        st.stop()

   # Normalize ticket data
    try:
        df = normalize_ticket_dataframe(df_raw)
        if df.empty:
            st.warning("Normalized dataframe is empty after processing.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to normalize tickets: {str(e)}")
        st.stop()
    
    # Load resolver (still needed for processing)
    try:
        resolver = load_queue_resolver(s3, bucket, resolver_path)
    except Exception as e:
        st.warning(f"Failed to load resolver: {str(e)}. Proceeding without resolver.")
        resolver = {}

    # Preprocess and create fresh analytics
    try:
        ticket_agg = preprocess_ticket_messages(df)        
    
    except Exception as e:
        st.error(f"Failed to preprocess tickets: {str(e)}")
        st.stop()

    # Enrich with agent + SLA
    ticket_agg = enrich_with_agent_and_sla(ticket_agg, resolver)
    # SLA metrics
    ticket_agg = compute_sla_metrics(ticket_agg)
    # Build transcript
    ticket_agg["transcript"] = ticket_agg["conversation"].apply(conversation_to_text)

    # Usage in your pipeline
    st.write("## Sentiment Analysis")

    # Compare with existing data and identify tickets needing analysis
    to_process, skipped, ticket_agg = identify_tickets_needing_analysis(
        ticket_agg, s3, bucket, analytics_path
    )

    # Process tickets that need sentiment analysis
    if not to_process.empty:
        st.info(f"Processing {len(to_process)} tickets for LLM sentiment analysis...")
        to_process = asyncio.run(apply_llm_sentiment_async(to_process, client))
        
        # Merge processed tickets back with skipped ones
        frames = []
        for df in [to_process, skipped]:
            if not df.empty and not df.isna().all().all():
                frames.append(df)
        final_ticket_agg = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()  
        st.success(f"Sentiment analysis complete. Total tickets: {len(final_ticket_agg)}")
    else:
        st.info("No tickets need sentiment analysis. All tickets are up to date.")
        final_ticket_agg = ticket_agg

    # Save updated analytics back to S3
    try:
        save_analytics_to_s3(s3, bucket, analytics_path, final_ticket_agg)
    except Exception as e:
        st.error(f"Failed to save analytics: {str(e)}")

    # ---------------------------------------------------------
    # ROLE-BASED FILTERING WITH SINGLE SELECT
    # ---------------------------------------------------------
    if st.session_state["role"] == "admin":
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Get unique agents
            all_agents = sorted(ticket_agg["assigned_agent"].dropna().unique().tolist())
            agent_options = ["All"] + all_agents
            
            selected_agent = st.selectbox(
                "Select Agent",
                options=agent_options,
                index=0,
                help="Select 'All' for consolidated report or choose a specific agent"
            )
        
        with col2:
            st.metric("Total Agents", len(all_agents))
        
        with col3:
            if st.button("🔄 Reset View"):
                st.rerun()
        
        # Apply filter based on selection
        if selected_agent == "All":
            # Show consolidated report for all agents
            st.success(f"📊 Viewing consolidated report for all {len(all_agents)} agents")
            view_mode = "consolidated"
        else:
            # Show specific agent's data
            ticket_agg = ticket_agg[ticket_agg["assigned_agent"] == selected_agent]
            st.info(f"👤 Viewing data for agent: **{selected_agent}**")
            view_mode = "individual"

    elif st.session_state["role"] == "agent":
        # Agent can only view their own tickets
        agent_name = st.session_state["agent_id"]
        ticket_agg = ticket_agg[ticket_agg["assigned_agent"] == agent_name]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"👤 Logged in as: **{agent_name}**")
        with col2:
            st.caption("Agent View")        
 
    else:
        st.error("⚠️ Invalid role. Access denied.")
        st.stop()
      
    # # Agent-level filtering, if logged in as agent
    # if st.session_state["role"] == "agent":
    #     agent_name = st.session_state["agent_id"]
    #     ticket_agg = ticket_agg[ticket_agg["assigned_agent"] == agent_name]

    summary = build_ticket_summary(ticket_agg)
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Ticket Summary",
        "SLA Overview",
        "Agent Performance",
        "Product Insights",
        "Ticket Trends",
        "Sentiment by Agent",
        "Aging Tickets",
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

        total_tickets = len(ticket_agg)
        breaches = ticket_agg["sla_breached"].sum()
        compliance = 1 - (breaches / total_tickets) if total_tickets > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tickets", total_tickets)
        col2.metric("SLA Breaches", int(breaches))
        col3.metric("Compliance Rate", f"{compliance:.1%}")

        st.markdown("### Product-wise SLA Breach Rate")
        prod_sla = ticket_agg.groupby("product_name")["sla_breached"].mean().reset_index()
        st.bar_chart(prod_sla, x="product_name", y="sla_breached")

        st.write("### SLA Metrics by Product and Customer Request")
        sla_stats, prod_stats = sla_stats_by_product_and_request(ticket_agg)
        st.dataframe(sla_stats)

        st.write("### SLA Metrics by Product")
        st.dataframe(prod_stats)
        
    # -----------------------------------------------------
    # TAB 3: Agent Performance
    # -----------------------------------------------------
    with tab3:
        st.subheader("Agent Performance Dashboard")

        # Enhanced agent performance metrics
        agent_perf = ticket_agg.groupby("assigned_agent").agg(
            total_tickets=("ticket_id", "count"),
            open_tickets=("status", lambda x: (x != "closed").sum()),
            closed_tickets=("status", lambda x: (x == "closed").sum()),
            avg_resolution_hours=("hours_to_close", "mean"),
            median_resolution_hours=("hours_to_close", "median"),
            sla_breach_rate=("sla_breached", "mean"),
        ).reset_index()

        st.dataframe(agent_perf)

        st.markdown("### Tickets Handled per Agent")
        st.bar_chart(agent_perf, x="assigned_agent", y="total_tickets")

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
        # Use pd.to_numeric which handles type conversion better:
        ticket_agg["sla_breached"] = pd.to_numeric(
            ticket_agg["sla_breached"], 
            errors='coerce',
            downcast='float'
        )

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
        # ============================================
        # Daily Ticket Trend
        # ============================================
        st.subheader("📈 Ticket Trends")

        # Daily Trend
        st.markdown("### Daily Ticket Trend")
        daily = (
            ticket_agg
            .assign(date=lambda df: df["ticket_opened_date"].dt.date)
            .groupby("date")
            .size()
            .reset_index(name="tickets")
        )
        st.line_chart(daily, x="date", y="tickets")

        # Weekly Trend
        st.markdown("### Weekly Ticket Trend")
        weekly = (
            ticket_agg
            .assign(
                week=lambda df: df["ticket_opened_date"].dt.to_period("W"),
                week_start=lambda df: df["week"].apply(lambda r: r.start_time if pd.notna(r) else None)
            )
            .dropna(subset=["week_start"])
            .groupby("week_start")
            .size()
            .reset_index(name="tickets")
        )
        st.line_chart(weekly, x="week_start", y="tickets")

        # Monthly Trend
        st.markdown("### Monthly Ticket Trend")
        monthly = (
            ticket_agg
            .assign(month=lambda df: df["ticket_opened_date"].dt.to_period("M").dt.to_timestamp())
            .dropna(subset=["month"])
            .groupby("month")
            .size()
            .reset_index(name="tickets")
            .set_index("month")
        )
        st.line_chart(monthly)

        # Yearly Trend
        st.markdown("### Yearly Ticket Trend")
        yearly = (
            ticket_agg
            .assign(year=lambda df: df["ticket_opened_date"].dt.year.astype("Int64"))
            .dropna(subset=["year"])
            .groupby("year", as_index=False)
            .size()
            .rename(columns={"size": "tickets"})
            .sort_values("year")
        )
        st.bar_chart(yearly, x="year", y="tickets")
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
    # -----------------------------------------------------
    # TAB 7: Ticket Aging metrics
    # -----------------------------------------------------
    with tab7:
        # Filter only open tickets
        open_tickets = ticket_agg[ticket_agg["status"] == "open"].copy()

        # Compute ticket age in days
        open_tickets["ticket_age_days"] = (
            pd.Timestamp.now().normalize() - open_tickets["ticket_opened_date"].dt.normalize()
        ).dt.days

        bins = [0, 30, 60, 90, 180, float("inf")]
        labels = ["0–30 days", "30–60 days", "60–90 days", "90–180 days", "gt 180 days"]

        open_tickets["age_bucket"] = pd.cut(
            open_tickets["ticket_age_days"],
            bins=bins,
            labels=labels,
            right=False
        )

        # Summary table
        age_summary = (
            open_tickets.groupby("age_bucket", observed=True)
            .size()
            .reset_index(name="open_tickets")
        )

        st.markdown("### Open Tickets by Age Bucket")
        st.write(f"Total open tickets {len(open_tickets)}")
        st.table(age_summary)

        open_tickets = open_tickets.sort_values("ticket_age_days", ascending=False)
        display_cols = [
            "ticket_id",
            "customer_id",
            "customer_name",
            "product_name",
            "ticket_opened_date",
            "ticket_age_days",
            "status",
            "sentiment_label",
            "assigned_agent",
            "sentiment_rationale",
            "sentiment_recommendation"
        ]

        st.subheader("Details of Open Tickets by Age")
        # ---------------------------------------------------------
        # ADVANCED FILTERS
        # ---------------------------------------------------------
        with st.expander("🔍 Filters", expanded=True):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            #with filter_col1:
                # Multi-select for status
                # status_options = sorted(open_tickets['status'].unique().tolist())
                # selected_statuses = st.multiselect(
                #     "Status (select multiple)",
                #     options=status_options,
                #     default=status_options
                # )
            
            with filter_col1:
                # Multi-select for sentiment
                sentiment_values = open_tickets['sentiment_label'].dropna().unique().tolist()
                sentiment_options = sorted([str(s) for s in sentiment_values])
                selected_sentiments = st.multiselect(
                    "Sentiment (select multiple)",
                    options=sentiment_options,
                    default=sentiment_options
                )
            
            with filter_col2:
                # Multi-select for product
                if 'product_name' in open_tickets.columns:
                    product_values = open_tickets['product_name'].dropna().unique().tolist()
                    product_options = sorted([str(p) for p in product_values])
                    selected_products = st.multiselect(
                        "Product (select multiple)",
                        options=product_options,
                        default=product_options
                    )
                else:
                    selected_products = []
            
            # Additional filters row
            filter_col3, filter_col4, filter_col5 = st.columns(3)
            
            with filter_col3:
                # Agent filter (if exists)
                if 'assigned_agent' in open_tickets.columns:
                    agent_options = ['All'] + sorted(open_tickets['assigned_agent'].dropna().unique().tolist())
                    selected_agent = st.selectbox("Assigned Agent", options=agent_options, index=0)
                else:
                    selected_agent = 'All'
            
            with filter_col4:
                # Priority filter (if exists)
                if 'priority' in open_tickets.columns:
                    priority_options = ['All'] + sorted(open_tickets['priority'].dropna().unique().tolist())
                    selected_priority = st.selectbox("Priority", options=priority_options, index=0)
                else:
                    selected_priority = 'All'
            
            with filter_col5:
                # Clear filters button
                if st.button("🔄 Clear All Filters", use_container_width=True):
                    st.rerun()

        # ---------------------------------------------------------
        # APPLY FILTERS
        # ---------------------------------------------------------
        filtered_tickets = open_tickets.copy()

        # Apply multi-select filters
        #if selected_statuses:
        #    filtered_tickets = filtered_tickets[filtered_tickets['status'].isin(selected_statuses)]

        if selected_sentiments:
            filtered_tickets = filtered_tickets[filtered_tickets['sentiment_label'].isin(selected_sentiments)]

        if selected_products and 'product_name' in filtered_tickets.columns:
            filtered_tickets = filtered_tickets[filtered_tickets['product_name'].isin(selected_products)]


        # Apply single-select filters
        if selected_agent != 'All' and 'assigned_agent' in filtered_tickets.columns:
            filtered_tickets = filtered_tickets[filtered_tickets['assigned_agent'] == selected_agent]

        if selected_priority != 'All' and 'priority' in filtered_tickets.columns:
            filtered_tickets = filtered_tickets[filtered_tickets['priority'] == selected_priority]

        # Show filter summary
        col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
        with col_info1:
            st.metric("Filtered Tickets", len(filtered_tickets), delta=len(filtered_tickets) - len(open_tickets))
        with col_info2:
            if len(filtered_tickets) > 0:
                avg_age = filtered_tickets.get('ticket_age_days', pd.Series([0])).mean()
                st.metric("Avg Age (days)", f"{avg_age:.1f}")
        with col_info3:
            if st.button("📊 Export Filtered Data"):
                csv = filtered_tickets.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"filtered_tickets_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        st.divider()
        # Rest of the AgGrid code remains the same...
        if not filtered_tickets.empty:
            # Prepare grid
            grid_options = build_grid_options(filtered_tickets, display_cols)
            df_view = filtered_tickets[display_cols].copy().reset_index(drop=True)

            grid_response = AgGrid(
                df_view,
                gridOptions=grid_options,
                enable_enterprise_modules=False,
                update_on=['rowSelected', 'cellValueChanged'],  
                height=400,
                fit_columns_on_grid_load=True,
                theme='streamlit'
            )

            # Handle selection
            selected_rows = grid_response.get("selected_rows")

            if selected_rows is not None:
                if not isinstance(selected_rows, pd.DataFrame):
                    selected_df = pd.DataFrame(selected_rows)
                else:
                    selected_df = selected_rows
                
                if not selected_df.empty:
                    st.divider()
                    
                    # Get ticket details
                    ticket = selected_df.iloc[0]
                    ticket_id = ticket.get("ticket_id")
                    
                    # Display header with ticket ID
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader(f"Ticket #{ticket_id}")
                    with col2:
                        if st.button("🔄 Refresh"):
                            st.rerun()
                    
                    # Sentiment information in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment_label = ticket.get('sentiment_label', 'N/A')
                        st.markdown(
                            f"<div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>"
                            f"<p style='color:#d62728; margin: 0;'><b>Sentiment Label</b><br/>{sentiment_label}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        sentiment_rationale = ticket.get('sentiment_rationale', 'N/A')
                        st.markdown(
                            f"<div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>"
                            f"<p style='color:#1f77b4; margin: 0;'><b>Rationale</b><br/>{sentiment_rationale[:50]}...</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        sentiment_recommendation = ticket.get('sentiment_recommendation', 'N/A')
                        st.markdown(
                            f"<div style='padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>"
                            f"<p style='color:#2ca02c; margin: 0;'><b>Recommendation</b><br/>{sentiment_recommendation[:50]}...</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    
                    # Full sentiment details in expander
                    with st.expander("📝 Full Sentiment Analysis", expanded=False):
                        st.markdown(f"**Rationale:** {sentiment_rationale}")
                        st.markdown(f"**Recommendation:** {sentiment_recommendation}")
                    
                    # Timeline
                    with st.expander("📅 Timeline", expanded=True):
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                        
                        ticket_data = ticket_agg[ticket_agg["ticket_id"] == ticket_id]

                        if not ticket_data.empty:
                            conversation = ticket_data.iloc[0].get("conversation", [])
                            
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                            
                            if conversation and isinstance(conversation, list):
                                st.write(f"Total messages in timeline: {len(conversation)}")
                                
                                for idx, message_dict in enumerate(conversation):
                                    msg_content = message_dict.get("msg_content", "")
                                    message_from = message_dict.get("message_from", "customer")
                                    posted_date = message_dict.get("msg_datetime", "")
                                    status = message_dict.get("status", "")
                                    
                                    if not msg_content or not str(msg_content).strip():
                                        continue
                                    
                                    # Format timestamp
                                    if posted_date:
                                        try:
                                            if isinstance(posted_date, pd.Timestamp):
                                                ts = posted_date.strftime("%b %d, %Y at %I:%M %p")
                                            else:
                                                dt = pd.to_datetime(posted_date)
                                                ts = dt.strftime("%b %d, %Y at %I:%M %p")
                                        except:
                                            ts = str(posted_date)
                                    else:
                                        ts = "Unknown time"
                                    
                                    msg_from = str(message_from).lower().strip()
                                    
                                    # Determine styling
                                    if msg_from in ['agent', 'support', 'system', 'admin']:
                                        bubble_class = "chat-bubble-agent"
                                        from_label = "🎧 " + msg_from.title()
                                        bg_color = "#e8f5e9"
                                        border_color = "#4caf50"
                                    else:
                                        bubble_class = "chat-bubble-customer"
                                        from_label = "👤 Customer"
                                        bg_color = "#e3f2fd"
                                        border_color = "#2196f3"
                                    
                                    # Add status badge if status changed
                                    status_badge = ""
                                    if status:
                                        status_colors = {
                                            'open': '#ff9800',
                                            'in_progress': '#2196f3',
                                            'pending': '#ffc107',
                                            'resolved': '#4caf50',
                                            'closed': '#9e9e9e'
                                        }
                                        status_color = status_colors.get(status.lower(), '#666')
                                        status_badge = f'<span style="background-color: {status_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; margin-left: 10px;">Status: {status}</span>'
                                    
                                    message_text = f"{from_label} • {ts} {status_badge}"
                                    
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background-color: {bg_color};
                                            padding: 12px;
                                            border-radius: 8px;
                                            margin: 8px 0;
                                            border-left: 4px solid {border_color};
                                        ">
                                            <div style="white-space: pre-wrap; margin-bottom: 8px; color: #333;">
                                                {msg_content}
                                            </div>
                                            <div style="text-align: right; font-family: Arial; color: #666; font-size: 0.85em;">
                                                <small>{message_text}</small>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.info(f"No conversation available for ticket #{ticket_id}")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("👆 Click on a row in the table above to view ticket details")
            else:
                st.info("👆 Click on a row in the table above to view ticket details")
        else:
            st.warning("⚠️ No tickets match the selected filters. Please adjust your criteria.")

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()