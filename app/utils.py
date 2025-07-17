"""
Utility functions for the Invoice Reimbursement System.
Includes PDF/ZIP extraction, LLM analysis, and answer generation.
"""
import zipfile
from PyPDF2 import PdfReader  # For PDF text extraction
from google import generativeai as genai
import logging
import re
import json
from datetime import date
import io
import os


key = os.getenv("GEMINI_API_KEY")

def extract_pdf_text(file):
    """
    Extracts all text from a PDF file or bytes.
    Returns the extracted text, or an error message if extraction fails.
    """
    try:
        # If file is bytes, wrap in BytesIO for PyPDF2
        if isinstance(file, bytes):
            file = io.BytesIO(file)
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return f"[ERROR: Could not extract text from PDF: {e}]"

def extract_invoices_from_zip(zip_file):
    """
    Extracts all PDFs from a ZIP file-like object or bytes and returns a dict of filename to text.
    Returns a dict with filename as key and extracted text or error message as value.
    """
    import io
    invoice_texts = {}
    try:
        # If zip_file is bytes, wrap in BytesIO
        if isinstance(zip_file, bytes):
            zip_file = io.BytesIO(zip_file)
        with zipfile.ZipFile(zip_file) as z:
            for filename in z.namelist():
                if filename.lower().endswith(".pdf"):
                    try:
                        with z.open(filename) as pdf_file:
                            text = extract_pdf_text(pdf_file)
                            invoice_texts[filename] = text
                    except Exception as e:
                        logging.error(f"Failed to extract {filename} from ZIP: {e}")
                        invoice_texts[filename] = f"[ERROR: Could not extract PDF: {e}]"
    except Exception as e:
        logging.error(f"Failed to open ZIP file: {e}")
        return {"[ZIP ERROR]": f"Could not open ZIP file: {e}"}
    return invoice_texts

def analyze_invoice_with_gemini(policy_text, invoice_text, employee_name, api_key= "Enter your gemini api key here"):
    """
    Uses Gemini LLM to analyze an invoice against a policy and return the analysis.
    Returns a JSON string with status, reason, and employee name.
    """
    try:
        if api_key:
            genai.configure(api_key=api_key)
        # Construct prompt for LLM
        prompt = f"""
        You are an expert HR reimbursement analyst. Given the following HR reimbursement policy and an employee's invoice, analyze the invoice and determine the reimbursement status. Use these categories: Fully Reimbursed, Partially Reimbursed, Declined. For each, provide a clear, detailed reason based on the policy. Return your answer in this JSON format:
        {{
            "status": "<Fully Reimbursed|Partially Reimbursed|Declined>",
            "reason": "<detailed reason>",
            "employee_name": "{employee_name}"
        }}

        HR Policy:
        {policy_text}

        Invoice:
        {invoice_text}
        """
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini LLM analysis failed: {e}")
        return f"[ERROR: Gemini LLM analysis failed: {e}]"

def answer_query_with_gemini(search_query, results, api_key= "Enter your gemini api key here" ):
    """
    Uses Gemini LLM to answer a user query based on the context and similarity search results.
    Args:
        search_query (str): The user's search query.
        results (list): List of (score, meta) tuples from similarity search.
        api_key (str, optional): Gemini API key. If None, uses default config.
    Returns:
        str: Markdown-formatted answer from Gemini.
    """
    try:
        if api_key:
            genai.configure(api_key=api_key)
        # Prepare context from results for the LLM prompt
        context = "\n\n".join([
            f"Invoice ID: {meta.get('invoice_id', 'N/A')}\nStatus: {meta.get('status', 'N/A')}\nReason: {meta.get('reason', 'N/A')}\nEmployee: {meta.get('employee_name', 'N/A')}\nDate: {meta.get('date', 'N/A')}"
            for _, meta in results
        ])
        rag_prompt = f"""
You are an intelligent assistant for invoice reimbursement queries. Given the following user query and a set of relevant invoice analyses, provide a clear, structured, and helpful answer in markdown format. If possible, summarize the findings, highlight any patterns, and answer the user's question directly.

User Query:
{search_query}

Relevant Invoice Analyses:
{context}
"""
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(rag_prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini LLM answer generation failed: {e}")
        return f"[ERROR: Gemini LLM answer generation failed: {e}]" 