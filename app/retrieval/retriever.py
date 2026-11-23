from app.core.logger import get_logger
from app.ingestion.embeddings import TextEmbedder
from app.ingestion.indexer import VectorStore
from langchain_core.documents import Document

#logger
logger = get_logger(__name__)

class Retriever:
    def __init__(self):
        try:
            embedder = TextEmbedder()

            self.chroma_store = VectorStore(embedder.embedding_model).chroma_store
            
            logger.info(f"Retriver initialized successfully!")
        except Exception:
            logger.exception(f"Failed to initialize retriever")
            raise

    def retrieve(self,users_query:str,k: int=3)->list[Document]:
        if not users_query or not users_query.strip():
            raise ValueError("Users query should not be empty!")

        if k<=0:
            raise ValueError("k must be greater than zero!")

        try:
            logger.info("Retrieving relevant document...")
            results = self.chroma_store.similarity_search(
                                            query = users_query,
                                            k = k)
            logger.info(f"Retrieved top-{len(results)} relevant documents")
            return results
        except Exception:
            logger.exception("Failed to retrieve relevant documents!")
            raise

if __name__=="__main__":
    retriever = Retriever()
    query = "what is the examination fee?"
    results = retriever.retrieve(query,k=-7)
    for doc in results:
        print("metadata: ",doc.metadata)
        print("content: ",doc.page_content)