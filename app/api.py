"""
API module for the Invoice Reimbursement System.
Provides endpoints for invoice analysis and chatbot interaction.
"""
from fastapi import FastAPI, File, UploadFile, Form, Request  # FastAPI core imports
from fastapi.responses import JSONResponse  # For custom JSON responses
from fastapi.middleware.cors import CORSMiddleware  # To enable CORS for frontend/backend communication
from utils import extract_pdf_text, extract_invoices_from_zip, analyze_invoice_with_gemini, answer_query_with_gemini  # Utility functions
from vector_store import add_invoice_analysis_to_vector_db, search_invoices, extract_metadata_filters  # Vector DB and search
from datetime import date  # For timestamping analyses
import json
import re
from pydantic import BaseModel  # For request validation
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing

app = FastAPI()

# Enable CORS for all origins (for development/testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatbotRequest(BaseModel):
    """Request model for chatbot endpoint."""
    message: str

def process_invoice(policy_text, fname, text, employee_name):
    """Helper function to analyze a single invoice (for threading)."""
    if text.startswith("[ERROR"):
        return {"invoice_id": fname, "error": text}
    analysis = analyze_invoice_with_gemini(policy_text, text, employee_name)
    if analysis.startswith("[ERROR"):
        return {"invoice_id": fname, "error": analysis}
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", analysis, flags=re.MULTILINE).strip()
        analysis_json = json.loads(cleaned)
        return {
            "invoice_id": fname,
            "status": analysis_json.get("status", ""),
            "reason": analysis_json.get("reason", ""),
            "employee_name": analysis_json.get("employee_name", employee_name),
            "date": str(date.today()),
            "full_text": text
        }
    except Exception as e:
        return {"invoice_id": fname, "error": f"Failed to parse LLM output: {e}"}

@app.post("/analyze_invoices")
async def analyze_invoices(
    policy_pdf: UploadFile = File(...),
    invoices_zip: UploadFile = File(...),
    employee_name: str = Form(...)
):
    """
    Analyze a batch of invoice PDFs against a policy PDF for a given employee.
    Stores successful analyses in the vector database.
    Uses multi-threading for efficient batch processing.
    """
    try:
        # Parse and extract text from the uploaded policy PDF
        policy_bytes = await policy_pdf.read()
        policy_text = extract_pdf_text(policy_bytes)
        if policy_text.startswith("[ERROR"):
            return JSONResponse(status_code=400, content={"success": False, "error": policy_text})

        # Parse ZIP and extract invoice PDFs as text
        zip_bytes = await invoices_zip.read()
        invoice_texts = extract_invoices_from_zip(zip_bytes)
        if any(fname.startswith("[ZIP ERROR]") for fname in invoice_texts):
            return JSONResponse(status_code=400, content={"success": False, "error": invoice_texts})

        analyses = []
        # Use ThreadPoolExecutor to process invoices in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_invoice, policy_text, fname, text, employee_name)
                for fname, text in invoice_texts.items()
            ]
            for future in as_completed(futures):
                analyses.append(future.result())

        # Store only successful analyses in the vector DB
        successful_analyses = [a for a in analyses if not a.get("error")]
        if successful_analyses:
            add_invoice_analysis_to_vector_db(successful_analyses)

        return {"success": True, "analyses": analyses}
    except Exception as e:
        # Catch-all error handler
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/chatbot")
async def chatbot_endpoint(request: ChatbotRequest):
    """
    Chatbot endpoint for answering invoice reimbursement queries.
    Combines metadata filtering and semantic search for relevant answers.
    """
    message = request.message
    if not message:
        return JSONResponse(status_code=400, content={"success": False, "error": "No message provided."})
    # Extract metadata filters from the query
    filters = extract_metadata_filters(message)
    # Perform filtered similarity search
    results = search_invoices(message, top_k=15, filters=filters)
    if not results:
        reply = "No matching invoices found. Please try a different query."
    else:
        reply = answer_query_with_gemini(message, results)
    return {"success": True, "response": reply} 