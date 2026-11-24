import torch
import asyncio
from fastapi import FastAPI,HTTPException,Query,Request,Depends
from app.core.config import settings
from app.schemas.chat import ChatRequest,ChatResponse
from app.schemas.ingest import IngestResponse
from app.services.rag_service import RAGService
from app.services.ingestion_service import IngestionService
from app.core.logger import get_logger
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool

#----logger----
logger = get_logger(settings.app_name)



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


#----ingestion  api ----
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