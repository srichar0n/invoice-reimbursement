import streamlit as st

st.set_page_config(page_title="Invoice Reimbursement System", layout="wide")
st.title("Welcome to the Invoice Reimbursement System")

st.write(
    """
    Use the sidebar to navigate between:
    - **Chatbot**: Ask questions about invoice reimbursements.
    - **Reimbursement**: Upload and analyze invoices.
    """
) 