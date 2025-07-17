"""
Chatbot page for the Invoice Reimbursement System multipage app.
"""
import streamlit as st
from vector_store import search_invoices, extract_metadata_filters
from utils import answer_query_with_gemini

st.title("Invoice Reimbursement Chatbot")  # Page title

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input with chat input
if prompt := st.chat_input("Type your query about invoice reimbursements..."):
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Process the user's query and generate assistant response
    with st.spinner("Searching and generating answer..."):
        # Extract metadata filters from the query
        filters = extract_metadata_filters(prompt)
        # Perform filtered similarity search
        results = search_invoices(prompt, top_k=15, filters=filters)
        if not results:
            assistant_reply = "No matching invoices found. Please try a different query."
        else:
            assistant_reply = answer_query_with_gemini(prompt, results)
    # Display assistant's reply and add to chat history
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
    st.session_state["messages"].append({"role": "assistant", "content": assistant_reply}) 