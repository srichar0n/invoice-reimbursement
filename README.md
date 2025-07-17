# Invoice Reimbursement System

## Project Overview
The Invoice Reimbursement System is a smart, end-to-end platform designed to automate and simplify employee invoice reimbursements. It leverages advanced language models and vector search to:
- Analyze invoices against HR reimbursement policies using LLMs (Gemini)
- Store and search invoice analyses efficiently using embeddings and FAISS
- Provide a conversational chatbot interface for querying invoice data
- Support both API and Streamlit-based user interactions
- Efficiently process large batches of invoices using multi-threading for fast analysis

This project is ideal for HR teams, finance departments, or anyone looking to modernize and automate reimbursement workflows.

---

## Directory Structure

```
reimbursement/
  app/
    api.py                # FastAPI backend
    streamlit_app.py      # Streamlit multipage app entry point
    utils.py              # Utility functions (PDF, ZIP, LLM, etc.)
    vector_store.py       # Vector DB and search logic
    pages/
      1_Chatbot.py        # Streamlit Chatbot page
      2_Reimbursement.py  # Streamlit Reimbursement upload/analysis page
  requirements.txt        # Python dependencies
  README.md               # This file
```

---

## Installation Instructions

**Important:** You must provide your own Gemini API key to use the LLM features. Set it as an environment variable (`GEMINI_API_KEY`) for best security.

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd reimbursement
   ```

2. **Set up a virtual environment (recommended):**
   ```sh
   python -m venv myvenv
   # On Windows:
   myvenv\Scripts\activate
   # On Mac/Linux:
   source myvenv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set your Gemini API key:**
   ```sh
   # Replace with your actual key
   set GEMINI_API_KEY=your-gemini-api-key   # On Windows
   export GEMINI_API_KEY=your-gemini-api-key # On Mac/Linux
   ```

---

## How to Run

### 1. Streamlit Multipage App (Recommended)
This is the main user interface. It provides:
- **Chatbot**: Ask questions about invoice reimbursements.
- **Reimbursement**: Upload HR policy and invoices for analysis.

**Start the app:**
```sh
streamlit run app/streamlit_app.py
```
- Use the sidebar to switch between Chatbot and Reimbursement pages.

### 2. FastAPI Backend 
If you want to use the API directly (for integration or testing):
```sh
fastapi dev app/api.py
```
- API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

**You can test all FastAPI endpoints interactively using Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs).**
Just start the FastAPI server and open this URL in your browser. Swagger UI lets you try out requests, upload files, and see responses directly from your browser—no extra tools needed!

---

## Usage Guide

### Streamlit App
- **Chatbot Page**: Enter natural language queries (e.g., "Show declined invoices for John").
- **Reimbursement Page**: Upload an HR policy PDF, a ZIP of invoice PDFs, and enter the employee name. The system analyzes each invoice and stores the results in the vector database.

### API Endpoints
- **POST /analyze_invoices**
  - Upload HR policy PDF, ZIP of invoice PDFs, and employee name.
  - Returns JSON with analysis results for each invoice.
- **POST /chatbot**
  - Send a message/query about invoice reimbursements.
  - Returns a structured answer using LLM and vector search.
- **Interactive API docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Technical Details

- **Libraries Used:**
  - Streamlit: Multipage UI
  - FastAPI: API backend
  - PyPDF2: PDF text extraction
  - google-generativeai: Gemini LLM integration
  - sentence-transformers: Embedding model (all-MiniLM-L6-v2)
  - faiss-cpu: Vector similarity search
  - numpy: Array operations
  - pydantic: Data validation
  - python-multipart: File uploads
  - concurrent.futures: Multi-threaded batch processing

- **LLM & Embedding Model Choices:**
  - **LLM:** Google Gemini (via `google-generativeai`)
  - **Embeddings:** all-MiniLM-L6-v2 (via `sentence-transformers`)

- **Vector Store Integration:**
  - Invoice analyses are embedded and stored in a FAISS index for efficient similarity search.
  - Metadata (invoice_id, status, reason, employee_name, date) is stored alongside embeddings for filtering.

- **Architecture:**
  - Users upload policies and invoices via Streamlit or API; analyses are stored in the vector DB.
  - Chatbot queries use both metadata filtering and semantic search for accurate answers.
  - Batch processing of invoices is parallelized using multi-threading for high throughput.

---

## Prompt Design

- **Invoice Analysis Prompt:**
  - The LLM is prompted as an expert HR reimbursement analyst, given the HR policy and invoice text, and asked to return a structured JSON with status, reason, and employee name.
- **Chatbot Prompt:**
  - The LLM is prompted as an intelligent assistant, given the user query and relevant invoice analyses, and asked to provide a clear, structured, and helpful answer in markdown format.
- **Metadata Extraction:**
  - Simple keyword matching is used to extract status, employee name, and date from user queries for efficient filtering.

---

## Challenges & Solutions
- **Extracting metadata from natural language:**
  - Regex and keyword matching for common patterns (like "declined", "for John", or dates). Fast and simple, but could be improved with LLM-based extraction.
- **Combining metadata filtering with semantic search:**
  - Filtering metadata before running FAISS similarity search makes the system faster and more relevant.
- **Parsing LLM output reliably:**
  - Enforced strict JSON output format in prompts and added error handling for parsing failures.
- **Efficient batch processing:**
  - Multi-threading is used to process many invoices in parallel, making the system scalable for real-world, high-volume use cases.

---





 