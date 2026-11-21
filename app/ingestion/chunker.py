from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.core.logger import get_logger


#logger 
logger = get_logger(__name__)

class DocumentChunker:
    def __init__(self,chunk_size=200,chunk_overlap=10):
        if chunk_size<=0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap<0 or chunk_overlap>=chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size")

        self.splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n","\n"," ",""],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
        logger.info("chunking initiated...")

    def chunk_documents(self,cleaned_docs: list[Document]) -> list[Document]:
        try:
            chunks = self.splitter.split_documents(cleaned_docs)
            logger.info(f"chunking completed! total chunks: {len(chunks)}")
            return chunks
        except Exception:
            logger.exception(f"chunking failed")
            raise


if __name__=="__main__":
    from app.ingestion.storage import DocumentStorage
    store = DocumentStorage()
    cleaned_docs = store.load_documents()
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(cleaned_docs)
    print(chunks)