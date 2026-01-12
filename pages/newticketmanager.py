import json
import asyncio
from typing import List, Dict

import streamlit as st
import pandas as pd

from openai import AsyncOpenAI
from utils.gcs_storage import GCSStorage
from utils.s3storage import S3Storage

client = AsyncOpenAI()

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

BATCH_SIZE = 10


# ---------- Domain helpers ----------

def filter_out_closed_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ticket_id" not in df.columns or "status" not in df.columns:
        raise KeyError("Expected columns 'ticket_id' and 'status' in the CSV")

    df["status"] = df["status"].astype(str).str.lower()

    has_closed = (
        df.groupby("ticket_id")["status"]
          .apply(lambda s: (s == "closed").any())
    )

    valid_ids = has_closed[~has_closed].index
    return df[df["ticket_id"].isin(valid_ids)]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder for your real preprocessing logic
    return df


# ---------- LLM helpers ----------

async def classify_request_batch(messages: List[str]) -> List[Dict]:
    """
    Takes a list of message strings and returns a list of dicts:
    {
        "summary": ...,
        "recommended_response": ...,
        "sentiment": ...,
        "sla_risk": ...
    }
    """
    prompt = f"""
You are a customer support assistant.

For each customer message below, produce a JSON object with:
- "summary": one-sentence summary of the customer's main request.
- "recommended_response": a concise, polite response an agent can send.
- "sentiment": one of ["positive", "neutral", "negative"] based on tone.
- "sla_risk": one of ["low", "medium", "high"] based on urgency, frustration, or severity.

Return a JSON list, one item per message, in the same order.

Messages:
{json.dumps(messages, indent=2)}
"""

    resp = await client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        else:
            # If model returns a single object, wrap it
            return [parsed] * len(messages)
    except Exception:
        # fallback: return generic values for all messages
        return [
            {
                "summary": "Unable to classify",
                "recommended_response": "Thank you for your message. We are reviewing your request.",
                "sentiment": "neutral",
                "sla_risk": "medium",
            }
            for _ in messages
        ]


async def classify_all_messages_async(
    messages: List[str],
    progress_mode: str,
    status_placeholder,
    progress_bar,
) -> List[Dict]:

    total = len(messages)
    if total == 0:
        return []

    # Build batches
    batches = []
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = messages[start:end]
        batches.append((start, end, batch))

    # Create tasks (no metadata needed)
    tasks = [classify_request_batch(batch) for _, _, batch in batches]

    results = [None] * total
    completed = 0

    # Run all tasks in parallel
    batch_outputs = await asyncio.gather(*tasks)

    # Process results in order
    for (start, end, _), batch_result in zip(batches, batch_outputs):
        for i, r in enumerate(batch_result):
            if start + i < total:
                results[start + i] = r

        completed += (end - start)
        progress_bar.progress(completed / total)
        status_placeholder.write(
            f"Processed {completed}/{total} messages (batch size {BATCH_SIZE})…"
        )

    # Fill missing slots
    for i in range(total):
        if results[i] is None:
            results[i] = {
                "summary": "Unable to classify",
                "recommended_response": "Thank you for your message. We are reviewing your request.",
                "sentiment": "neutral",
                "sla_risk": "medium",
            }

    return results

def classify_all_messages(messages: List[str], progress_mode: str, status, progress):
    # Wrapper to run async in Streamlit
    return asyncio.run(
        classify_all_messages_async(messages, progress_mode, status, progress)
    )


# ---------- Queue builder ----------

def build_queue(df: pd.DataFrame, progress_mode: str) -> pd.DataFrame:
    df = df.copy()

    df["nature_of_customer_request"] = None
    df["recommended_response"] = None
    df["sentiment"] = None
    df["sla_risk"] = None

    # Progress UI
    progress = st.progress(0)
    status = st.empty()

    # Decide what we are iterating over based on progress_mode
    if progress_mode == "Per ticket (A)":
        # One message per row, but label by ticket
        messages = df["msg_content"].fillna("").tolist()
        labels = df["ticket_id"].tolist()
    elif progress_mode == "Per message row (B)":
        messages = df["msg_content"].fillna("").tolist()
        labels = list(range(1, len(messages) + 1))
    else:  # "Per unique ticket (C)"
        # For unique ticket mode, we classify only the first row per ticket
        unique_ids = df["ticket_id"].unique().tolist()
        messages = []
        labels = []
        index_map = []  # map from result index to df index

        for ticket_id in unique_ids:
            idx = df[df["ticket_id"] == ticket_id].index[0]
            messages.append(df.at[idx, "msg_content"] or "")
            labels.append(ticket_id)
            index_map.append(idx)

    with st.spinner("Analyzing tickets with AI (batched, parallel)…"):
        results = classify_all_messages(messages, progress_mode, status, progress)

    status.write("Processing complete.")

    # Write results back to df
    if progress_mode in ["Per ticket (A)", "Per message row (B)"]:
        for i, r in enumerate(results):
            df.at[df.index[i], "nature_of_customer_request"] = r.get("summary")
            df.at[df.index[i], "recommended_response"] = r.get("recommended_response")
            df.at[df.index[i], "sentiment"] = r.get("sentiment")
            df.at[df.index[i], "sla_risk"] = r.get("sla_risk")
    else:
        # Unique ticket mode: only first row per ticket gets classified
        for i, r in enumerate(results):
            idx = index_map[i]
            df.at[idx, "nature_of_customer_request"] = r.get("summary")
            df.at[idx, "recommended_response"] = r.get("recommended_response")
            df.at[idx, "sentiment"] = r.get("sentiment")
            df.at[idx, "sla_risk"] = r.get("sla_risk")

    s3 = S3Storage(
        aws_access_key=st.secrets["aws"]["access_key"],
        aws_secret_key=st.secrets["aws"]["secret_key"],
        region=st.secrets["aws"].get("region", "ap-south-1"),
    )
    # Cloud config from secrets
    bucket = st.secrets["aws"]["bucket"]
    resolver_path = st.secrets["aws"]["queue_resolver_path"]
    agents = s3.read_json(bucket, resolver_path)
    st.write(agents)
    df["agent_id"] = df["product_name"].map(agents)
    st.write(df["agent_id"])

    # Ensure all required columns exist
    for col in REQUIRED_QUEUE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    df = df[REQUIRED_QUEUE_COLUMNS]

    if "ticket_id" in df.columns:
        df = df.sort_values("ticket_id")


    return df


# ---------- Streamlit app ----------

def main():
    st.title("📌 Ticket Queue Builder")
    if st.session_state.get("role") != "admin":
        st.error("Access denied. Admins only.")
        st.button("Go to Login Page", on_click=st.switch_page("login.py"))
        st.stop()

    # Storage adapters
    gcs = GCSStorage(credentials_key="gcp")
    s3 = S3Storage(
        aws_access_key=st.secrets["aws"]["access_key"],
        aws_secret_key=st.secrets["aws"]["secret_key"],
        region=st.secrets["aws"].get("region", "ap-south-1"),
    )

    # Cloud config from secrets
    gcs_bucket = st.secrets["gcp"]["bucket"]
    gcs_path = st.secrets["gcp"]["input_path"]

    s3_bucket = st.secrets["aws"]["bucket"]
    s3_path = st.secrets["aws"]["output_path"]

    st.subheader("Cloud Configuration")
    st.info(f"Using GCS bucket: `{gcs_bucket}`")
    st.info(f"Using GCS path: `{gcs_path}`")
    st.info(f"Using S3 bucket: `{s3_bucket}`")
    st.info(f"Using S3 output path: `{s3_path}`")

    df = None

    if st.button("Load, Filter, and Preprocess from GCS"):
        try:
            df_raw = gcs.read_csv(gcs_bucket, gcs_path)
            raw_count = len(df_raw)

            df_filtered = filter_out_closed_tickets(df_raw)
            df_filtered["number_of_interactions"] = (df_filtered.groupby("ticket_id")["msg_content"].transform("count"))
            filtered_count = len(df_filtered)
            unique_tickets = df_filtered["ticket_id"].nunique()

            df_processed = preprocess(df_filtered)
            processed_count = len(df_processed)

            df = df_processed

            st.success("Loaded, filtered, and preprocessed CSV from GCS")

            # Stats card
            st.subheader("📊 Data Summary")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Rows in CSV", raw_count)
            with col2:
                st.metric("Rows after filtering", filtered_count)
            with col3:
                st.metric("Unique tickets", unique_tickets)
            with col4:
                st.metric("Rows after preprocessing", processed_count)

            st.subheader("Filtered Data Preview")
            st.dataframe(df.head(), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading or processing file: {e}")

    if df is not None:
        st.session_state["filtered_df"] = df
    elif "filtered_df" in st.session_state:
        df = st.session_state["filtered_df"]

    if df is not None:
        st.subheader("LLM Processing Configuration")

        progress_mode = st.radio(
            "Progress mode",
            [
                "Per ticket (A)",
                "Per message row (B)",
                "Per unique ticket (C)",
            ],
            index=0,
        )

        if st.button("Generate Queue with AI Analysis and Save to S3"):
            try:
                queue = build_queue(df, progress_mode)

                st.subheader("LLM‑Enriched Queue Preview")
                st.dataframe(queue, use_container_width=True)

                s3.write_parquet(
                    queue,
                    bucket=s3_bucket,
                    path=s3_path,
                    overwrite=True,
                )
                st.success(f"queue.parquet successfully written to s3://{s3_bucket}/{s3_path}")

            except Exception as e:
                st.error(f"Error generating queue or writing to S3: {e}")


if __name__ == "__main__":
    main()