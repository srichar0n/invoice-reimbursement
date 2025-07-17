"""
Reimbursement page for the Invoice Reimbursement System multipage app.
"""
import streamlit as st
from utils import extract_pdf_text, extract_invoices_from_zip, analyze_invoice_with_gemini
from vector_store import add_invoice_analysis_to_vector_db
import json
from datetime import date
import re

st.title("Invoice Reimbursement Application")  # Page title

policy_pdf = st.file_uploader("Upload HR Reimbursement Policy (PDF)", type=["pdf"])
invoices_zip = st.file_uploader("Upload Employee Invoices (ZIP of PDFs)", type=["zip"])
employee_name = st.text_input("Employee Name")

if st.button("Submit"):
    if not policy_pdf or not invoices_zip or not employee_name:
        st.error("Please provide all inputs: policy PDF, invoices ZIP, and employee name.")
    else:
        # Parse policy PDF
        policy_text = extract_pdf_text(policy_pdf)
        if policy_text.startswith("[ERROR"):
            st.error(policy_text)
            st.stop()

        # Parse ZIP and extract invoice PDFs
        invoice_texts = extract_invoices_from_zip(invoices_zip)
        if any(fname.startswith("[ZIP ERROR]") for fname in invoice_texts):
            for fname, msg in invoice_texts.items():
                st.error(f"{fname}: {msg}")
            st.stop()

        st.subheader(f"LLM Analysis for {len(invoice_texts)} Invoice(s)")
        analyses = []
        for fname, text in invoice_texts.items():
            if text.startswith("[ERROR"):
                st.error(f"{fname}: {text}")
                continue
            with st.spinner(f"Analyzing {fname} with Gemini..."):
                analysis = analyze_invoice_with_gemini(policy_text, text, employee_name)
            if analysis.startswith("[ERROR"):
                st.error(f"{fname}: {analysis}")
            else:
                st.markdown(f"**{fname}:**")
                st.code(analysis, language="json")
                try:
                    # Remove markdown code block formatting if present
                    cleaned = re.sub(r"^```(?:json)?|```$", "", analysis, flags=re.MULTILINE).strip()
                    analysis_json = json.loads(cleaned)
                    analyses.append({
                        "invoice_id": fname,
                        "status": analysis_json.get("status", ""),
                        "reason": analysis_json.get("reason", ""),
                        "employee_name": analysis_json.get("employee_name", employee_name),
                        "date": str(date.today()),
                        "full_text": text
                    })
                except Exception as e:
                    st.error(f"Failed to parse LLM output for {fname}: {e}")
        if analyses:
            add_invoice_analysis_to_vector_db(analyses)
            st.success(f"Uploaded {len(analyses)} analysis results to the vector database.") 