# SmartCampus Assistant

SmartCampus Assistant is a Retrieval-Augmented Generation (RAG) application that answers student questions using information extracted from college documents. It is being developed as a modular Python backend with FastAPI, LangChain, Hugging Face embeddings, ChromaDB, a Cross-Encoder reranker, and an LLM accessed through OpenRouter.

> **Demo-data notice:** ABC College and all college information used in this project are fictional sample data. They must not be treated as official information about a real institution.

## Project status

The project is under active development. The current backend supports document ingestion, vector indexing, retrieval, reranking, LLM-based answer generation, and initial REST APIs.

### Implemented

- FastAPI application with a lifespan-managed initialization of the RAG and ingestion services.
- Health-check endpoint.
- Document discovery from `data/raw/` using Docling-supported file extensions.
- Document extraction with Docling using Markdown export.
- Exact duplicate-document detection using SHA-256 hashes of whitespace-normalized extracted content.
- Text cleaning while preserving document metadata.
- Persistence of cleaned documents as JSON Lines (JSONL).
- Recursive character-based chunking with configurable chunk size and overlap.
- Hugging Face Sentence Transformers embeddings.
- Persistent ChromaDB vector storage.
- Deterministic chunk IDs based on source path and normalized chunk content.
- Semantic similarity retrieval from ChromaDB.
- Cross-Encoder reranking of retrieved documents.
- Context-grounded prompt construction and LLM response generation.
- OpenRouter integration through LangChain's `ChatOpenAI` interface, currently configured for GPT-4o mini.
- Ingestion and retrieval service layers.
- Pydantic request and response schemas for the chat and ingestion APIs.
- Unit tests, API tests, ChromaDB integration testing, and a RAG pipeline integration test.
- A Locust user scenario for initial API load testing.

### Planned / in progress

- Document upload API and automatic ingestion after upload.
- Incremental processing of only new or modified documents.
- Document deletion and removal of obsolete vectors.
- Ingestion job status tracking and failure recovery.
- Student-facing chat UI and admin dashboard.
- Authentication and role-based authorization.
- RAG quality evaluation (retrieval relevance and answer quality).
- Containerization and deployment.
- Production monitoring and further performance testing.

## Architecture

### Document ingestion pipeline

Documents are currently placed manually under `data/raw/`. The ingestion service runs the following pipeline:

```text
Documents in data/raw/
        |
        v
DataLoader (Docling / Markdown export)
        |
        v
Duplicate document detection (SHA-256)
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
VectorStore (ChromaDB + Hugging Face embeddings)
```

The embedding function is supplied to ChromaDB. ChromaDB uses it to embed document chunks during indexing and user queries during similarity search; the service does not need to manually pass precomputed vectors to ChromaDB.

### Retrieval and answer generation

```text
Student question
       |
       v
Retriever
       |
       v
Query embedding + ChromaDB similarity search
       |
       v
Top-k retrieved documents
       |
       v
Cross-Encoder reranker
       |
       v
Top reranked documents
       |
       v
RAG prompt (question + retrieved context)
       |
       v
OpenRouter (GPT-4o mini)
       |
       v
Answer + source information
```

The RAG prompt instructs the model to answer using the retrieved college-document context, avoid inventing missing facts, and treat retrieved documents as reference material rather than instructions.

### API flow

The API endpoints are currently defined in `app/main.py`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check that the application services have initialized |
| POST | `/api/v1/chat` | Retrieve relevant content, rerank it, and generate an answer |
| POST | `/api/v1/documents/ingest` | Run the ingestion pipeline over documents in `data/raw/` |

The chat endpoint accepts a JSON request:

```json
{
  "question": "What is the examination fee for 2026-2027?"
}
```

Optional retrieval parameters can be supplied as query parameters:

```text
/api/v1/chat?retrieve_k=10&rerank_k=3
```

The intended chat response schema is:

```json
{
  "answer": "The examination fee for 2026-2027 is Rs. 2000.",
  "sources": [
    {
      "source": "data/raw/examination_fee.pdf",
      "score": 0.91
    }
  ]
}
```

The ingestion endpoint returns processing counts:

```json
{
  "documents_loaded": 5,
  "documents_cleaned": 5,
  "chunks_created": 42,
  "chunks_indexed": 42
}
```

**Current ingestion behavior:** The endpoint scans and processes the documents already present in `data/raw/`. It is not yet an upload endpoint and does not automatically detect filesystem changes. The endpoint currently runs the ingestion pipeline during the request; background job processing is planned.

## Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| API framework | FastAPI |
| Data validation and settings | Pydantic, Pydantic Settings |
| Document extraction | Docling |
| RAG framework and prompts | LangChain |
| Text splitting | LangChain Text Splitters |
| Embeddings | Hugging Face Sentence Transformers |
| Vector database | ChromaDB |
| Reranking | Sentence Transformers Cross-Encoder |
| LLM provider | OpenRouter |
| LLM integration | LangChain `ChatOpenAI` compatible client |
| Testing | pytest, FastAPI TestClient, HTTPX |
| Load testing | Locust |

## Repository structure

The tree below reflects the files currently present in the repository. Some planned areas (such as separate API route modules, a frontend, and background workers) have not been added yet.

```text
abc-college-rag-assistant/
├── app/
│   ├── main.py
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
│   │   ├── retriever.py
│   │   └── reranker.py
│   ├── services/
│   │   ├── ingestion_service.py
│   │   ├── retrieval_service.py
│   │   └── rag_service.py
│   └── schemas/
│       ├── chat.py
│       └── ingest.py
├── data/
│   ├── raw/
│   └── processed/
├── storage/
│   └── chroma_db/          # Created at runtime; ignored by Git
├── tests/
│   ├── test_ingestion.py
│   ├── test_cleaner.py
│   ├── test_chunker.py
│   ├── test_indexer.py
│   ├── test_chroma_integration.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_retrieval_service.py
│   ├── test_rag_service.py
│   ├── test_ingestion_service.py
│   ├── test_chat_api.py
│   └── test_rag_integration.py
├── locustfile.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Notes:

- `app/core/security.py` is currently a placeholder; authentication and authorization have not been implemented.
- API endpoints currently live in `app/main.py`; there are no separate `app/api/routes/` modules yet.
- The `data/raw/`, `data/processed/`, `storage/chroma_db/`, and `logs/` directories contain local runtime data and are excluded from Git (except for tracked placeholder files, if present).

## Setup and run

### 1. Clone the repository

```bash
git clone https://github.com/Taj1920/abc-college-rag-assistant.git
cd abc-college-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Docling, Sentence Transformers, and their model dependencies may take time to install. The first embedding or Cross-Encoder model initialization may download model files.

### 4. Configure environment variables

Create a local `.env` file from the example.

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Configure the values in `.env`:

```env
APP_NAME="SmartCampus Assistant"
DEBUG=true

# The application currently reads this setting as OPENAI_API_KEY.
# Put your OpenRouter API key here; do not commit this file.
OPENAI_API_KEY=your_openrouter_api_key
LLM_MODEL=openai/gpt-4o-mini

# Local Hugging Face embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Persistent ChromaDB directory
CHROMA_PERSIST_DIR=storage/chroma_db

# Optional override; otherwise the application default is used
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2
```

The current `RAGService` configures the OpenRouter-compatible base URL (`https://openrouter.ai/api/v1`) in code and reads the API key through the existing `openai_api_key` setting. The environment variable name is retained for compatibility with the current configuration; the value must be an OpenRouter key.

Never commit `.env`, publish API keys, or include real credentials in screenshots or logs. Keep `.env.example` free of secrets.

### 5. Add sample college documents

Place supported source documents under:

```text
data/raw/
```

The loader searches recursively and uses Docling's supported-format mapping. Actual format support can depend on the installed Docling version and optional dependencies.

### 6. Start the FastAPI application

Run from the repository root:

```bash
uvicorn app.main:app --reload
```

Open:

- Application: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

### 7. Run document ingestion

With documents placed under `data/raw/`, call the ingestion endpoint from Swagger UI or another API client:

```http
POST http://127.0.0.1:8000/api/v1/documents/ingest
```

The service discovers the available documents and runs the current ingestion pipeline. Re-running ingestion for unchanged documents reuses deterministic chunk IDs, avoiding additional ChromaDB records for those same source/content pairs.

### 8. Ask a question

Use Swagger UI or send a request:

```http
POST http://127.0.0.1:8000/api/v1/chat
Content-Type: application/json
```

```json
{
  "question": "What is the examination fee for 2026-2027?"
}
```

The chat endpoint uses the configured retrieval and reranking defaults unless query parameters are provided.

## Testing

Run the test suite from the repository root:

```bash
pytest -v
```

Run a specific test module:

```bash
pytest tests/test_ingestion.py -v
pytest tests/test_cleaner.py -v
pytest tests/test_chunker.py -v
pytest tests/test_indexer.py -v
pytest tests/test_chat_api.py -v
```

The tests cover document discovery and duplicate detection, cleaning, chunking, deterministic vector IDs, ChromaDB indexing, retrieval, reranking, service orchestration, API validation/error handling, and RAG integration.

The ChromaDB integration test uses a temporary persistence directory and a real local embedding model / ChromaDB instance. The RAG integration test uses mocked external LLM and Cross-Encoder calls; it is not a full production test against the live OpenRouter service.

## Initial API load testing

A Locust scenario is provided in `locustfile.py`. Start the FastAPI server in one terminal:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

In another terminal, run:

```bash
locust -f locustfile.py --host http://127.0.0.1:8000
```

Then open http://localhost:8089.

**Cost warning:** The current Locust scenario calls the real chat endpoint. Successful requests can invoke the configured OpenRouter model and may incur usage charges. Use small test runs or a mock LLM for high-volume concurrency testing.

## Duplicate handling and current limitations

### Duplicate handling

- **Document level:** `DataLoader` normalizes whitespace in extracted text and calculates a SHA-256 hash. Exact duplicate extracted documents are skipped; the first occurrence is retained.
- **Chunk level:** `VectorStore` creates a deterministic SHA-256 ID from the source path and normalized chunk content. Re-indexing an unchanged chunk from the same source reuses its ID.
- These mechanisms prevent duplicate records for unchanged source/content pairs during repeated ingestion.

### Known limitations

- When a source document is modified, changed chunks receive new IDs, but chunks from the previous version are not automatically deleted. Source-level synchronization and deletion are planned.
- The loader currently ingests all discovered documents; it does not yet process only the changed file.
- Duplicate documents with identical extracted text are collapsed at load time, so only the first source's metadata is retained.
- Page-level source citations are not currently guaranteed; the observed Docling Markdown export metadata primarily contains the source path.
- Document upload, automatic change detection, background ingestion jobs, job status tracking, and document deletion are not implemented yet.
- Authentication and role-based access control are not implemented.
- RAG quality evaluation and production deployment are still planned.
- The no-retrieved-context response path in `RAGService` should be kept consistent with the `ChatResponse` schema before frontend integration: the current service has a plain-text fallback while the API response model expects an object containing `answer` and `sources`.

## Development notes

- Run Python commands from the repository root.
- Keep API credentials in `.env`; never commit secrets.
- Update `requirements.txt` when adding dependencies.
- Do not commit original college documents, processed JSONL files, local ChromaDB data, logs, or environment files.
- Rebuild the vector collection if the embedding model changes, so all indexed vectors use the same embedding space.
- ABC College content is fictional demonstration data only.
