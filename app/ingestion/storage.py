import json
from pathlib import Path
from app.core.logger import get_logger
from langchain_core.documents import Document
#paths
ROOT = Path(__file__).parents[2]
#logger
logger = get_logger(__name__)

class DocumentStorage:
    def __init__(self,file_path:Path | None = None):
        self.file_path = file_path or (
            ROOT / "data" / "processed" / "cleaned_documents.jsonl"
        )

    def save_documents(self,documents: list[Document]) -> int:

        self.file_path.parent.mkdir(parents=True,exist_ok=True)
        temp_path = self.file_path.with_suffix(
            self.file_path.suffix + ".tmp"
        )
        try:
            with temp_path.open("w",encoding="utf-8") as file:
                for doc in documents:
                    record = {
                        "page_content":doc.page_content,
                        "metadata":doc.metadata
                    }
                    file.write(json.dumps(record,ensure_ascii=False) + "\n")
            temp_path.replace(self.file_path)
            logger.info(f"Saved {len(documents)} to {self.file_path}")
            return len(documents)
        
        except Exception:
            temp_path.unlink(missing_ok=True)
            logger.exception(f"Failed to save documents")
            raise


