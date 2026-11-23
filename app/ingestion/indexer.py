import hashlib
from app.core.config import settings
from langchain_chroma import Chroma
from pathlib import Path
from app.core.logger import get_logger
from langchain_core.documents import Document

#logger
logger = get_logger(__name__)

#path
ROOT = Path(__file__).parents[2]

class VectorStore:
    def __init__(self,embedding):
        try:
            persist_dir = ROOT / settings.chroma_persist_dir
            persist_dir.mkdir(parents=True,exist_ok=True)
            self.chroma_store = Chroma(collection_name="smartcampus_documents",
                                       persist_directory=str(persist_dir),
                                       embedding_function=embedding,
                                    )
            logger.info("Vector store initialized successfully!")
        except Exception:
            logger.exception("Vector store initialization failed!")
            raise
    @staticmethod
    def generate_document_id(document:Document) ->str:
        source = str(document.metadata.get("source","unknown"))
        content = " ".join(document.page_content.split())
        unique_content = f"{source}|{content}"
        return hashlib.sha256(
            unique_content.encode("utf-8")
        ).hexdigest()
    
    def add_documents(self,documents:list[Document])-> list[str]:
        if not documents:
            logger.warning("No documents provided for indexing.")
            return []
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")

            unique_documents = {}
            for document in documents:
                doc_id = self.generate_document_id(document)
                unique_documents[doc_id] = document

            ids = list(unique_documents.keys())
            docs = list(unique_documents.values())

            stored_ids = self.chroma_store.add_documents(
                documents=docs,
                ids=ids
            )
            logger.info(f"successfully added {len(stored_ids)} documents to vector store")
            return stored_ids
        except Exception:
            logger.exception("Failed to index documents")
            raise

if __name__=="__main__":
    from app.ingestion.storage import DocumentStorage
    from app.ingestion.chunker import DocumentChunker
    from app.ingestion.embeddings import TextEmbedder
    store = DocumentStorage()
    cleaned_docs = store.load_documents()
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(cleaned_docs)
    embedder = TextEmbedder()
    vectorstore = VectorStore(embedder.embedding_model)
    ids = vectorstore.add_documents(chunks)
    print(f"Total records: {vectorstore.chroma_store._collection.count()}")