import asyncio
import pandas as pd
import streamlit as st
import utils.gcs_storage as GCSStorage
import utils.s3storage as S3Storage

class TicketPipeline:
    def __init__(self, client):
        # Storage setup
        self.gcs = GCSStorage(credentials_key="gcp")
        self.gcs_bucket = st.secrets["gcp"]["bucket"]
        self.gcs_path = st.secrets["gcp"]["input_path"]

        self.s3 = S3Storage(
            aws_access_key=st.secrets["aws"]["access_key"],
            aws_secret_key=st.secrets["aws"]["secret_key"],
            region=st.secrets["aws"].get("region", "ap-south-1"),
        )

        self.bucket = st.secrets["aws"]["bucket"]
        self.resolver_path = st.secrets["aws"]["queue_resolver_path"]
        self.analytics_path = st.secrets["aws"]["analytics_path"]

        self.client = client

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


    def load_and_normalize(self):
        df_raw = self.gcs.read_csv(self.gcs_bucket, self.gcs_path)
        if df_raw is None:
            st.error(f"CSV file not found at {self.gcs_path} in bucket {self.gcs_bucket}.")
            return pd.DataFrame()

        df_raw = df_raw.head(120)

        st.write("Number of records fetched from CSV")
        st.write(len(df_raw))

        if df_raw.empty:
            st.stop()

        df = self.normalize_ticket_dataframe(df_raw)
        if df.empty:
            st.stop()

        return df

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
        aggregated = df.groupby("ticket_id").apply(
            lambda g: pd.Series({
                "customer_id": g["customer_id"].iloc[0],
                "customer_name": g["customer_name"].iloc[0],
                "product_name": g["product_name"].iloc[0],
                "status": g["status"].iloc[0],
                # Consolidated conversation
                "conversation": build_conversation(g),

                # Lifecycle fields
                "ticket_opened_date": g["posted_date"].min(),
                "ticket_closed_date": g["closed_date"].max()
            })
        ).reset_index()

        return aggregated

    def load_existing_analytics(self):
        ticket_agg = self.s3.read_parquet(self.bucket, self.analytics_path)
        st.write(f"Loaded {len(ticket_agg)} existing aggregated tickets from analytics.")
        return ticket_agg

    def merge_new_tickets(self, ticket_agg, df):
        new_ticket_agg = self.preprocess_ticket_messages(df)

        if len(ticket_agg) > 0:
            existing_ids = set(ticket_agg["ticket_id"].tolist())
            new_tickets = new_ticket_agg[~new_ticket_agg["ticket_id"].isin(existing_ids)]
            ticket_agg = pd.concat([ticket_agg, new_tickets], ignore_index=True)
        else:
            ticket_agg = new_ticket_agg

        return ticket_agg

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

    def enrich(self, ticket_agg):
        resolver = self.load_queue_resolver(self.s3, self.bucket, self.resolver_path)
        ticket_agg = self.enrich_with_agent_and_sla(ticket_agg, resolver)
        ticket_agg = self.compute_sla_metrics(ticket_agg)
        ticket_agg["transcript"] = ticket_agg["conversation"].apply(self.conversation_to_text)
        return ticket_agg

    def process_sentiment(self, ticket_agg):
        if len(ticket_agg) > 0 and "sentiment_label" in ticket_agg.columns:
            st.info(f"Total tickets to analyze for sentiment: {len(ticket_agg)}")

            to_process = ticket_agg[
                ticket_agg["ticket_closed_date"].isna() |
                ticket_agg["sentiment_label"].isna()
            ].copy()

            skipped = ticket_agg[
                ticket_agg["ticket_closed_date"].notna() &
                ticket_agg["sentiment_label"].notna()
            ]
        else:
            to_process = ticket_agg.copy()
            skipped = pd.DataFrame(columns=ticket_agg.columns)

        if not to_process.empty:
            st.info(f"Processing {len(to_process)} tickets for LLM sentiment analysis...")
            to_process = asyncio.run(self.apply_llm_sentiment_async(to_process, self.client))

        required_cols = [
            "sentiment_label",
            "sentiment_rationale",
            "sentiment_recommendation"
        ]

        for col in required_cols:
            if col not in skipped.columns:
                skipped[col] = None

        return pd.concat([to_process, skipped], ignore_index=True)

    def conversation_to_text(conversation):
        lines = []
        for msg in conversation:
            sender = msg.get("message_from", "unknown")
            time = msg.get("msg_datetime", "")
            content = msg.get("msg_content", "")
            lines.append(f"[{time}] {sender}: {content}")
        return "\n".join(lines)

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

    def save(self, ticket_agg):
        self.save_analytics_to_s3(self.s3, self.bucket, self.analytics_path, ticket_agg)

    def run(self):
        df = self.load_and_normalize()
        ticket_agg = self.load_existing_analytics()
        ticket_agg = self.merge_new_tickets(ticket_agg, df)
        ticket_agg = self.enrich(ticket_agg)
        ticket_agg = self.process_sentiment(ticket_agg)

        st.write(ticket_agg.head(20))
        self.save(ticket_agg)

        return ticket_agg