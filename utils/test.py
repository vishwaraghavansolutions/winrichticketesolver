from google.cloud import storage
import utils.gcs_storage as GCSStorage
import streamlit as st

# client = storage.Client()
# b = client.bucket("winrich")
# print(b.exists())

# blobs = b.list_blobs(prefix="Datawarehouse/MutualFunds/2026/01/15/mutualfunds.csv")

# for b in blobs:
#     print(b.name)


def load_csv_from_gcp(storage, bucket, path):
    raw = storage.read_csv(bucket, path)
    if raw is None:
        st.error(f"CSV file not found at {path} in bucket {bucket}.")
        return pd.DataFrame()
    return raw

gcs = GCSStorage(credentials_key="gcp")
# Cloud config from secrets
gcs_bucket = st.secrets["gcp"]["bucket"]
gcs_path = st.secrets["gcp"]["mf_input_path"]

# Load data
df_raw = load_csv_from_gcp(gcs, gcs_bucket, gcs_path)
st.write(df_raw)
