# SmartCampus Assistant

An AI-powered college assistant. The project uses Retrieval-Augmented Generation (RAG) to answer student questions from college documents.

> **Note:** ABC College information used in this project is fictional sample data. It must not be presented as official information about a real institution.

## Project Status

The project is under active development.

### Completed
- FastAPI application setup and health endpoint
- Document discovery and extraction with Docling
- Markdown export from Docling
- Exact duplicate-document detection using SHA-256 content hashes
- Document cleaning and normalization
- Saving and loading cleaned documents in JSONL format
- Document chunking with LangChain RecursiveCharacterTextSplitter
- Text embeddings with Hugging Face Sentence Transformers
- Persistent vector storage and indexing with ChromaDB
- Unit tests for document loading, cleaning, chunking, and vector indexing

### In Progress / Planned
- Real ChromaDB integration checks and retrieval
- Retriever and metadata filtering
- Reranking
- Prompt augmentation and LLM response generation
- Chat and document-management API endpoints
- RAG evaluation and end-to-end tests

## Architecture

### Current ingestion pipeline

~~~text
Documents in data/raw/
        |
        v
DoclingLoader (Markdown export)
        |
        v
Exact duplicate detection (SHA-256)
        |
        v
DocumentCleaner
        |
        v
DocumentStorage (JSONL)
        |
        v
DocumentChunker
        |
        v
TextEmbedder (Hugging Face)
        |
        v
VectorStore (ChromaDB)
~~~

Docling extracts document content into Markdown. The application then cleans and stores the extracted documents, splits them into chunks, generates embeddings, and indexes the chunks in ChromaDB.

### Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Document extraction | Docling |
| RAG framework | LangChain |
| Text splitting | LangChain Text Splitters |
| Embeddings | Hugging Face Sentence Transformers |
| Vector database | ChromaDB |
| Configuration | Pydantic Settings |
| Testing | pytest |

## Project Structure

~~~text
abc-college-rag-assistant/
├── app/
│   ├── main.py
│   ├── api/
│   ├── core/
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── security.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   ├── storage.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   └── indexer.py
│   ├── retrieval/
│   ├── generation/
│   ├── services/
│   └── schemas/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── storage/
│   └── chroma_db/
├── tests/
├── .env.example
├── requirements.txt
└── README.md
~~~

The structure will grow as retrieval, generation, and API features are implemented. Local source documents, processed data, and the persisted ChromaDB are runtime data and should not be committed to Git.

## Setup and Run Instructions

### 1. Clone the repository

~~~bash
git clone https://github.com/Taj1920/abc-college-rag-assistant.git
cd abc-college-rag-assistant
~~~

### 2. Create and activate a virtual environment

~~~bash
python -m venv .venv
~~~

**Windows PowerShell:**
~~~powershell
.venv\Scripts\Activate.ps1
~~~

**Windows Command Prompt:**
~~~cmd
.venv\Scripts\activate.bat
~~~

**Linux / macOS:**
~~~bash
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

The first embedding-model initialization may download the configured Sentence Transformers model.

### 4. Configure environment variables

Create a local .env file from the example:

**Windows PowerShell:**
~~~powershell
Copy-Item .env.example .env
~~~

**Linux / macOS:**
~~~bash
cp .env.example .env
~~~

The current ingestion pipeline uses a local Hugging Face embedding model. Configure the LLM settings when you implement and run the generation component.

Example configuration:

~~~env
APP_NAME="SmartCampus Assistant"
DEBUG=true

LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=storage/chroma_db
~~~

Never commit .env or expose API keys. Keep .env.example free of real secrets.

### 5. Add source documents

Place the documents you want to ingest under:

~~~text
data/raw/
~~~

The loader recursively discovers supported file types using Docling's FormatToExtensions mapping. Actual format support can depend on the installed Docling version and optional dependencies.

### 6. Run the ingestion components

The ingestion components are currently implemented as separate modules. The indexer module includes a development entry point that loads the previously saved cleaned documents, chunks them, initializes the embedding model, and indexes the chunks:

~~~bash
python -m app.ingestion.indexer
~~~

This command expects cleaned documents to already exist at:

~~~text
data/processed/cleaned_documents.jsonl
~~~

The loader, cleaner, and storage stages should be run before indexing. A single end-to-end ingestion orchestration script is planned.

ChromaDB persists its data in:

~~~text
storage/chroma_db/
~~~

### 7. Run the FastAPI application

~~~bash
uvicorn app.main:app --reload
~~~

Open the local application at:

~~~text
http://127.0.0.1:8000
~~~

Check the health endpoint:

~~~text
http://127.0.0.1:8000/health
~~~

Explore the interactive API documentation:

~~~text
http://127.0.0.1:8000/docs
~~~

Only endpoints that have been implemented will be available.

## Testing

Run all tests from the project root:

~~~bash
pytest -v
~~~

Or run individual test modules:

~~~bash
pytest tests/test_ingestion.py -v
pytest tests/test_chunker.py -v
pytest tests/test_indexer.py -v
~~~

The indexer unit-test suite currently contains 11 tests covering initialization, deterministic ID generation, duplicate removal within an indexing batch, empty input, successful indexing, and error handling. These tests mock ChromaDB; a separate integration test against a real temporary ChromaDB collection is still planned.

## Important Implementation Notes

- The loader's SHA-256 hash detects exact duplicate extracted document content after whitespace normalization.
- The indexer generates deterministic IDs from a chunk's source and normalized content. Re-indexing unchanged chunks therefore reuses their IDs.
- Duplicate chunks with the same generated ID within one indexing batch are submitted only once.
- If a source document changes, its changed chunks receive new IDs. The current indexer does not automatically delete obsolete chunks from an earlier version of that source.
- The current Docling Markdown export does not provide page numbers in the metadata observed in this project. Page-level citations require additional page-provenance handling.
- If you change the embedding model, rebuild the ChromaDB collection so all stored vectors use the same embedding space.

## Development Notes

- Run commands from the repository root.
- Keep credentials and secrets in .env.
- Update requirements.txt when adding dependencies.
- Do not commit source documents, processed JSONL files, or local ChromaDB data.
- This project is a learning and portfolio project; features marked as planned are not yet implemented.
