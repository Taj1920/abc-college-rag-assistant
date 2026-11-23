from app.ingestion.loader import DataLoader
from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.storage import DocumentStorage
from app.ingestion.chunker import DocumentChunker
from app.ingestion.embeddings import TextEmbedder
from app.ingestion.indexer import VectorStore
from app.core.logger import get_logger


#---logger----
logger = get_logger(__name__)

class IngestionService:
    def __init__(self):
        try:
            self.loader = DataLoader()
            self.cleaner = DocumentCleaner()
            self.storage = DocumentStorage()
            self.chunker = DocumentChunker()
            self.embedder = TextEmbedder()

            self.vector_store = VectorStore(
                embedding=self.embedder.embedding_model
            )

            logger.info("Ingestion service initialized successfully!")
        except Exception:
            logger.exception("Failed initializing ingestion service")
            raise

    def ingest_documents(self)->dict:
        try:
            #load all documents
            documents = self.loader.load_all_documents()
            if not documents:
                logger.warning("No documents found for ingestion!")
                return {
                    "documents_loaded": 0,
                    "documents_cleaned": 0,
                    "chunks_created": 0,
                    "chunks_indexed": 0
                }
            #clean documents
            cleaned_docs = self.cleaner.clean_documents(documents)
            if not cleaned_docs:
                logger.warning("No documents available for cleaning.")
                return {
                    "documents_loaded": len(documents),
                    "documents_cleaned": 0,
                    "chunks_created": 0,
                    "chunks_indexed": 0
                }

            #save cleaned documents
            self.storage.save_documents(cleaned_docs)

            #create chunks
            chunks = self.chunker.chunk_documents(cleaned_docs)

            if not chunks:
                logger.warning("No chunks created")
                return {
                    "documents_loaded": len(documents),
                    "documents_cleaned": len(cleaned_docs),
                    "chunks_created": 0,
                    "chunks_indexed": 0
                }
            #index chunks
            indexed_ids = self.vector_store.add_documents(chunks)

            logger.info("Document ingestion completed successfully!")

            return {
                "documents_loaded": len(documents),
                "documents_cleaned": len(cleaned_docs),
                "chunks_created": len(chunks),
                "chunks_indexed": len(indexed_ids)
            }

        except Exception:
            logger.exception("Document ingestion pipeline failed.")
            raise

        

