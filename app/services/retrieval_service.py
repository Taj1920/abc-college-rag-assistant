from app.core.logger import get_logger
from app.retrieval.retriever import Retriever
from app.retrieval.reranker import Reranker
from langchain_core.documents import Document

#logger
logger = get_logger(__name__)

class RetrievalService:
    def __init__(self):
        try:
            self.retriever = Retriever()
            self.reranker = Reranker()
            logger.info("Retrieval Service initialized successfully!")
        except Exception:
            logger.exception(f"Failed to initialize Retrival service")
            raise

    def retrieve_relevant_documents(self,
                                    users_query:str,
                                    retrieve_k:int=10,
                                    rerank_k:int=3)->list[tuple[Document,float]]:
        if not users_query or not users_query.strip():
            raise ValueError("User query cannot be empty!")
        if retrieve_k<=0:
            raise ValueError("retrive_k must be greater than zero!")
        if rerank_k<=0:
            raise ValueError("rerank_k must  be greater than zero!")

        try:
            logger.info("Starting Retrieval Pipeline")

            retrieved_docs = self.retriever.retrieve(users_query,
                                                     k=retrieve_k)
            if not retrieved_docs:
                logger.warning("No relevant documents retrieved")
                return []
            reranked_docs = self.reranker.re_rank(users_query,
                                                retrieved_docs,
                                                k=rerank_k)
            logger.info(f"Retrieval completed. Final documents: {len(reranked_docs)}")
            return reranked_docs

        except Exception:
            logger.exception("Retrieval pipeline failed!")
            raise
