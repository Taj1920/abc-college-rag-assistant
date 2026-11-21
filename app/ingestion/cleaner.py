import re
from app.core.logger import get_logger
from langchain_core.documents import Document

#logger
logger = get_logger(__name__)

class DocumentCleaner:

    def clean_document(self, document: Document) -> Document:
        """Clean the text content while preserving document metadata."""

        text = document.page_content

        # Normalize Windows and old Mac line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove null bytes and unwanted control characters
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # Normalize spaces and tabs without removing paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)

        # Remove spaces at the beginning and end of lines
        text = re.sub(r" *\n *", "\n", text)

        # Replace 3 or more consecutive newlines with 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading and trailing whitespace
        text = text.strip()

        return Document(
            page_content=text,
            metadata=document.metadata.copy()
        )

    def clean_documents(self,documents:list[Document]) -> list[Document]:
        cleaned_documents = []
        logger.info("Document cleaning initialized...")
        for document in documents:
            cleaned = self.clean_document(document)
            if not cleaned.page_content:
                logger.warning(f"Skipping empty document: {cleaned.metadata.get('source','unknown')}")
                continue

            cleaned_documents.append(cleaned)

        logger.info(f"Document cleaning completed!. Input: {len(documents)} | Output: {len(cleaned_documents)}")
        return cleaned_documents