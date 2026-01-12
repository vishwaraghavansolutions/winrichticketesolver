import streamlit as st
import pandas as pd
from utils.gcs_storage import GCSStorage
from utils.s3storage import S3Storage

def load_parquet_data():
    s3 = S3Storage(
        aws_access_key=st.secrets["aws"]["access_key"],
        aws_secret_key=st.secrets["aws"]["secret_key"],
        region=st.secrets["aws"].get("region", "ap-south-1"),
    )

    bucket = st.secrets["aws"]["bucket"]
    path = st.secrets["aws"]["output_path"]

    df = s3.read_parquet(bucket, path)
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    return df


def load_lifecycle_csv():
    gcs = GCSStorage(credentials_key="gcp")
    bucket = st.secrets["gcp"]["bucket"]
    path = st.secrets["gcp"]["input_path"]

    df = gcs.read_csv(bucket, path)
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    return df