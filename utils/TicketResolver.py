import pandas as pd
from datetime import datetime
import pytz

class TicketResolver:
    def __init__(self, s3_storage=None, gcp_storage=None):
        """
        Pass whichever storage backends you want:
        - Only S3
        - Only GCP
        - Both S3 and GCP
        """
        self.s3 = s3_storage
        self.gcp = gcp_storage
        self.ist = pytz.timezone("Asia/Kolkata")

    # ---------------------------------------------------------
    # Helper: get metadata from first lifecycle row
    # ---------------------------------------------------------
    def get_ticket_metadata(self, lifecycle_df, ticket_id):
        rows = lifecycle_df[lifecycle_df["ticket_id"] == ticket_id]

        if rows.empty:
            return None, None, None

        first = rows.iloc[0]

        return (
            first["customer_id"],
            first["customer_name"],
            first["product_name"]
        )

    # ---------------------------------------------------------
    # 1) Append a new lifecycle entry
    # ---------------------------------------------------------
    def append_entry(self, lifecycle_df, ticket_id, comment_text, status="open"):
        customer_id, customer_name, product_name = self.get_ticket_metadata(
            lifecycle_df, ticket_id
        )

        now = datetime.now(self.ist)
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        formatted_posted_date = now.isoformat(sep=" ", timespec="milliseconds")

        closed_date = formatted_posted_date if status == "closed" else ""

        new_row = pd.DataFrame([{
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "product_name": product_name,
            "message_from": "agent",
            "msg_content": comment_text,
            "msg_datetime": formatted_now,
            "status": status,
            "posted_date": str(formatted_posted_date),
            "closed_date": str(closed_date)
        }])

        lifecycle_df = pd.concat([lifecycle_df, new_row], ignore_index=True)

        return lifecycle_df

    # ---------------------------------------------------------
    # 2) Mark all rows for this ticket_id as closed
    # ---------------------------------------------------------
    def close_ticket(self, lifecycle_df, ticket_id):
        now = datetime.now(self.ist)
        formatted_posted_date = now.isoformat(sep=" ", timespec="milliseconds")

        lifecycle_df.loc[
            lifecycle_df["ticket_id"] == ticket_id, "status"
        ] = "closed"

        lifecycle_df.loc[
            lifecycle_df["ticket_id"] == ticket_id, "closed_date"
        ] = formatted_posted_date

        return lifecycle_df
