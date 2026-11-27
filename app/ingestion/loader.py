import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader,ExportType
from app.core.logger import get_logger
from docling.datamodel.base_models import FormatToExtensions

#initialize logger
logger = get_logger(__name__)

#root path
ROOT = Path(__file__).parents[2]

class DataLoader:
    def __init__(self):
        self.docs_path = ROOT / "data" / "raw"
        self.supported_ext = {ext.lower() for exts in FormatToExtensions.values() for ext in exts}
        logger.info("Data Loader initialized...")
    
    def discover_files(self) -> list[Path]:
        if not self.docs_path.exists():
            logger.error(f"Data directory doesnot exist {self.docs_path}")
            return []
        
        logger.info("Discovering supported files...")
        supported_files= []
        for file in self.docs_path.rglob("*"):
            if file.is_file() and file.name.lower().rsplit(".")[-1] in self.supported_ext:
                supported_files.append(file)
        logger.info(f"Found {len(supported_files)} supported files !")
        return supported_files
    
    def remove_duplicate_docs(self,documents):
        unique_docs = []
        seen_hashes = set()
        for doc in documents:
            normalized_text = " ".join(doc.page_content.split())
            if not normalized_text:
                logger.warning("Skipping empty document")
                continue
            content_hash = hashlib.sha256(
                normalized_text.encode("utf-8")
            ).hexdigest()

            if content_hash in seen_hashes:
                logger.warning(f"Duplicate docs detected. skipping: {doc.metadata.get('source','unknown')}")
                continue
            
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
        return unique_docs


         
    def load_all_documents(self) -> list[Document]:
        supported_files = self.discover_files()
        all_documents = []
        for file in supported_files:
            try:
                logger.info(f"Loading file: {file}..")
                loader = DoclingLoader(
                    file_path=str(file),
                    export_type=ExportType.MARKDOWN
                    )
                docs = loader.load()
                for doc in docs:
                        try:
                            source_path = file.relative_to(ROOT)
                        except ValueError:
                            source_path = file.relative_to(self.docs_path)
                        doc.metadata["source"] = source_path.as_posix()
                all_documents.extend(docs)
                logger.info(f"Successfully loaded {file.name} | documents: {len(docs)}")
            except Exception:
                logger.exception(f"Failed to load file: {file}")
        logger.info(f"Total documents loaded: {len(all_documents)}")
        unique_docs = self.remove_duplicate_docs(all_documents)
        logger.info(f"unique documents after de duplication: {len(unique_docs)}")
        return unique_docs

    def load_file(self,file_path:Path) -> list[Document]:  
            file_path=Path(file_path)
            if not file_path.exists() or not file_path.is_file():
                raise FileNotFoundError(f"Document not found : {file_path}")

            extension = file_path.suffix.lower().lstrip(".")

            if extension not in self.supported_ext:
                raise ValueError(f"Unsupported file extension: {extension}")

            try:
                logger.info(f"Loading uploaded file: {file_path}..")
                loader = DoclingLoader(
                    file_path=str(file_path),
                    export_type=ExportType.MARKDOWN
                    )
                docs = loader.load()
                for doc in docs:
                    try:
                        source_path = file_path.relative_to(ROOT)
                    except ValueError:
                        source_path = file_path.relative_to(self.docs_path)

                    doc.metadata["source"] = source_path.as_posix()

                unique_docs = self.remove_duplicate_docs(docs)
                logger.info(f"Successfully loaded {file_path.name} | documents: {len(unique_docs)}")
                return unique_docs
            except Exception:
                logger.exception(f"Failed to load file: {file_path}")
                raise 
        


if __name__=="__main__":
    loader = DataLoader()
    docs = loader.load_all_documents()
    print(docs)