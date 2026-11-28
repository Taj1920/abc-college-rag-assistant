from app.core.logger import get_logger
from sentence_transformers import CrossEncoder
from app.core.config import settings
from langchain_core.documents import Document

#logger
logger = get_logger(__name__)
class Reranker:
    def __init__(self):
        try:
            self.cross_encoder = CrossEncoder(
                settings.cross_encoder_model
            )
            logger.info(f"cross-encoder {settings.cross_encoder_model} Initialized!")
        except Exception:
            logger.exception("cross-encoder initialization failed!")
            raise

    def re_rank(self,users_query:str,documents:list[Document],
                k:int=5)->list[float]:
        if not users_query or not users_query.strip():
                    raise ValueError("Users query should not be empty!")

        if not documents:
             logger.warning("No documents provided for re-ranking!")
             return []
        if k<=0:
             raise ValueError("K value must be greater than zero!")
        try:
            pairs = [(users_query,doc.page_content) for doc in documents]
            relevance_scores = self.cross_encoder.predict(pairs)
            logger.info(f"Calculated relevance score for {len(documents)} documents!")
            
            #map documents and relevance scores
            scored_documents = zip(documents,relevance_scores)

            #sort scored_docs
            re_ranked_docs = sorted(scored_documents,
                                    key = lambda item: item[1],
                                    reverse=True)
            logger.info(f"Successfully reranked {len(re_ranked_docs)} documents!")
            return re_ranked_docs[:k]
        except Exception:
            logger.exception("Failed to rerank documents")
            raise

