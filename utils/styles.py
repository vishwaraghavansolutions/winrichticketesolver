import streamlit as st

def inject_global_css():
    st.markdown(
        """
        <style>
        /* your full CSS from earlier goes here */
        </style>
        """,
        unsafe_allow_html=True,
    )