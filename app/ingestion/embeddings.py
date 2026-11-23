from app.core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.logger import get_logger
from langchain_core.documents import Document


logger = get_logger(__name__)

class TextEmbedder:
    def __init__(self):
        try:
            self.embedding_model= HuggingFaceEmbeddings(
            model_name = settings.embedding_model
            )
            logger.info("Embedding model initialized!")
        except Exception:
            logger.exception(f"Failed to initialize embedding model {settings.embedding_model}")
            raise
    def create_embeddings(self,chunks:list[Document])-> list[list[float]]:
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []
        try:
            logger.info("Creating vector embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embedding_model.embed_documents(texts)
            logger.info(f"Embedding creation completed. Total vectors: {len(vectors)} | Embedding size: {len(vectors[0])}")
            return vectors
        except Exception:
            logger.exception("Failed to create embeddings")
            raise
if __name__=="__main__":
    from app.ingestion.storage import DocumentStorage
    from app.ingestion.chunker import DocumentChunker
    store = DocumentStorage()
    cleaned_docs = store.load_documents()
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(cleaned_docs)
    embedder = TextEmbedder()
    vectors = embedder.create_embeddings(chunks)
    print(vectors[0][:10])


        

