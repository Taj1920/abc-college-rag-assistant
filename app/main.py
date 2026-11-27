import torch
import asyncio
import shutil
import uuid
import hashlib
from pathlib import Path
from datetime import datetime,timezone
from fastapi import FastAPI,HTTPException,Query,Request,Depends,UploadFile,File,BackgroundTasks
from app.core.config import settings
from app.schemas.chat import ChatRequest,ChatResponse
from app.schemas.ingest import IngestResponse
from app.services.rag_service import RAGService
from app.services.ingestion_service import IngestionService
from app.ingestion.document_registry import DocumentRegistry
from app.core.logger import get_logger
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
from app.ingestion.loader import DataLoader

#----logger----
logger = get_logger(settings.app_name)


#-----ROOT----
ROOT = Path(__file__).parents[1]

#-----lifespan---
@asynccontextmanager
async def lifespan(app:FastAPI):
    # Configure CPU resources
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    logger.info("Initializing SmartCampus Assistant")
    try:
        app.state.rag_service = RAGService()
        logger.info("RAG service initiated successfully!")

        app.state.ingestion_service = IngestionService()
        logger.info("Ingestion service initiated successfully!")

        app.state.registry = DocumentRegistry()
        logger.info("Document Registry initialized successfully!")

        # Maximum two simultaneous RAG operations
        app.state.rag_semaphore = asyncio.Semaphore(2)
        yield

    except Exception:
        logger.exception("Application lifecycle failed")
        raise
    finally:
        logger.info("Shutting down SmartCampus Assistant...")


#----fast api app----
app = FastAPI(title=settings.app_name,
              debug=settings.debug,
              version="0.0.1",
              description="An AI-powered college assistant",
              lifespan=lifespan
              )

#---initialize RAG service----
def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service

#-----initialize ingestion service----
def get_ingestion_service(request:Request) -> IngestionService:
    return request.app.state.ingestion_service

#----initialize document registry----
def get_document_registry(request:Request)->DocumentRegistry:
    return request.app.state.registry

#---initialize rag semaphore----
def get_rag_semaphore(request: Request) -> asyncio.Semaphore:
    return request.app.state.rag_semaphore

@app.get("/health")
def health_check(request:Request):

    if not hasattr(request.app.state, "rag_service"):
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready"
        )
    
    if not hasattr(request.app.state,"ingestion_service"):
        raise HTTPException(
            status_code=503,
            detail="Ingestion service is not ready"
        )
    
    return {
        "status":"healthy",
        "message":"SmartCampus Assistant is running..."
        }

#------------version 1 apis----------------

#---Retrieval api---
@app.post("/api/v1/chat",response_model=ChatResponse)
async def chat(request: ChatRequest,
         rag_service:RAGService = Depends(get_rag_service),
         rag_semaphore: asyncio.Semaphore = Depends(get_rag_semaphore),
         retrieve_k:int=Query(default=10,ge=1,le=100),
         rerank_k:int=Query(default=3,ge=1,le=50)):
    
    if rerank_k>retrieve_k:
        raise HTTPException(
            status_code=422,
            detail = "rerank_k cannot be greater than retrieve_k"
        )

    # Wait maximum 2 seconds for a processing slot
    try:
        await asyncio.wait_for(
            rag_semaphore.acquire(),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Server is busy. Please try again shortly.",
            headers={"Retry-After": "2"}
        )

    
    try:
        response = await run_in_threadpool(rag_service.generate_response,
                                           request.question,
                                           retrieve_k=retrieve_k,
                                           rerank_k=rerank_k)
        return response
    
    except Exception:
        logger.exception("Chat API request failed")

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question right now.")
    finally:
        #release the processing slot
        rag_semaphore.release()


#----Bulk ingestion  api (triggers bulk upload from data/raw/)----
@app.post("/api/v1/documents/ingest",response_model=IngestResponse)
def ingest_documents(ingestion_service:IngestionService = Depends(get_ingestion_service)):
    try:
        response = ingestion_service.ingest_documents()
        return response
    except Exception:
        logger.exception("Document ingestion API failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to run ingestion pipeline."
        )

#---file upload api (upload a file and ingestion triggers in the background for that file)---
@app.post("/api/v1/documents/upload",status_code=202)
async def upload_document(background_tasks:BackgroundTasks,
                          uploaded_file:UploadFile=File(...),
                          registry:DocumentRegistry = Depends(get_document_registry),
                          ingestion_service:IngestionService = Depends(get_ingestion_service)):
    if uploaded_file.size is None or uploaded_file.size==0:
        raise HTTPException(
            status_code=400,
            detail="Cannot upload empty files!"
        )
    extension = Path(uploaded_file.filename).suffix.lower().lstrip(".")
    if extension not in DataLoader().supported_ext:
        raise HTTPException(
            status_code = 415,
            detail="Unsupported file found for Docling Loader"
        ) 
    
    #create file hash
    file_content  = await uploaded_file.read()
    
    file_hash = hashlib.sha256( file_content ).hexdigest()

    await uploaded_file.seek(0)

    #validate existing file
    existing_document_id = registry.get_document_by_hash(file_hash)

    if existing_document_id:
        logger.warning(f"Duplicate document detected: {existing_document_id}")

        raise HTTPException(
            status_code=409,
            detail={
                "message":"The document has already been uploaded.",
                "document_id":existing_document_id
            }
        )

    
    destination_dir = ROOT / "data" / "raw"
    destination_dir.mkdir(parents=True,exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}{Path(uploaded_file.filename).suffix.lower()}"
    destination =  destination_dir / unique_filename

    try:
        #saving file to data directory
        with destination.open("wb") as file:
            shutil.copyfileobj(uploaded_file.file,file)
        logger.info(f"File {uploaded_file.filename} uploaded successfully!")

        #Registering new document
        uploaded_at = datetime.now(timezone.utc).isoformat()
        document_id = registry.register_document(original_filename=uploaded_file.filename,
                                   stored_filename=unique_filename,
                                   file_hash=file_hash,
                                   uploaded_at = uploaded_at
                                   )

    except Exception:
        destination.unlink(missing_ok=True)
        logger.exception("Failed to save uploaded document")
        raise HTTPException(
            status_code=500,
            detail="Unable to save uploaded document"
        )

    #schedule ingestion background task
    background_tasks.add_task(
                            ingestion_service.ingest_file,
                            destination,
                            document_id
                            )
    logger.info(
        f"Background ingestion scheduled for document: {document_id}"
    )

    return {
        "document_id":document_id,
        "filename": uploaded_file.filename,
        "stored_filename": unique_filename,
        "status":"pending",
        "message": "Document uploaded successfully. Ingestion scheduled"
    }
