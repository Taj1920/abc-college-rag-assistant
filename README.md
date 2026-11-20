# SmartCampus AI Assistant
An AI-powered college assistant that answers student queries using college documents through Retrieval-Augmented Generation (RAG).

## Setup and Run Instructions

Follow these steps to set up and run the SmartCampus Assistant locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Taj1920/abc-college-rag-assistant.git
cd abc-college-rag-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create your local `.env` file using the example configuration:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```

Open `.env` and configure the required values:

```env
APP_NAME="SmartCampus Assistant"
DEBUG=true

OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini

EMBEDDING_MODEL=text-embedding-3-small

VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=storage/chroma_db
```

Replace `your_openai_api_key_here` with your valid OpenAI API key.

**Important:** Never commit your `.env` file or expose your API key on GitHub.

### 5. Run the FastAPI Application

Start the development server from the project root:

```bash
uvicorn app.main:app --reload
```

The application will start at:

```text
http://127.0.0.1:8000
```

### 6. Verify the Application

Check the health endpoint:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "message": "SmartCampus Assistant is running"
}
```

### 7. Explore API Documentation

Open the interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

You can explore and test the available API endpoints from Swagger UI.

---

## Development Notes

- Run commands from the project root directory.
- Keep API keys and other secrets inside `.env`.
- Update `requirements.txt` whenever you add new dependencies.
- Store generated ChromaDB data inside `storage/chroma_db/`.
- Run automated tests using:

```bash
pytest
```

## Current Development Status

SmartCampus Assistant is under active development. Some features and API endpoints may not be available yet.


# SmartCampus Assistant — Sample Knowledge Base

This folder contains fictional ABC College documents for testing document loading, text extraction, metadata extraction, chunking, embeddings, indexing and RAG retrieval.

## Files
- `01_student_handbook.pdf` — multi-page student handbook
- `02_admissions_guide_2026_2027.pdf` — multi-page admissions guide
- `03_examination_regulations.docx` — examination rules and assessment procedures
- `04_hostel_handbook.docx` — hostel fees, schedule, rules and safety
- `05_fee_schedule_2026_2027.csv` — structured fee records
- `06_library_notice.html` — HTML campus notice
- `07_student_services_faq.md` — Markdown FAQ

## Important
All college details, email addresses, dates, rules and fees are synthetic sample data. They are not official ABC College policies and must not be represented as real-world facts.

## Suggested ingestion tests
1. Load each file type with the relevant loader.
2. Print document count and metadata.
3. Inspect extracted text from the first and last pages.
4. Confirm page metadata is retained where the loader provides it.
5. Compare how tables are extracted from PDF, DOCX and CSV.
6. Ask questions that require combining information from different documents.


## Document Ingestion

SmartCampus Assistant uses **Docling** to load documents and extract their content.

### Supported Features

- Automatically discovers supported documents inside the `data/raw/` directory.
- Recursively scans subdirectories for files.
- Uses Docling for document loading and text extraction.
- Exports extracted content in Markdown format.
- Handles individual file-loading failures without stopping the entire ingestion process.
- Detects exact duplicate documents using SHA-256 content hashing.
- Normalizes whitespace before duplicate detection.
- Skips empty documents.
- Logs document loading and duplicate detection activities.

### Supported File Types

File extensions are discovered dynamically using Docling's `FormatToExtensions` mapping. Actual support depends on the installed Docling version and optional dependencies.

### How to Run

Place your documents inside:

```text
data/raw/
```

Run the loader:

```bash
python -m app.ingestion.loader
```

### Testing

Run ingestion tests:

```bash
pytest tests/test_ingestion.py -v
```

The ingestion test suite currently covers file discovery, recursive scanning, duplicate detection, empty documents, document loading, and error handling.