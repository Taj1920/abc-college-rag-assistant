
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from langchain_core.documents import Document

import app.services.ingestion_service as ingestion_module
from app.services.ingestion_service import IngestionService


@pytest.fixture
def service(monkeypatch):
    # Avoid initializing real models and databases
    ingestion_service = IngestionService.__new__(IngestionService)

    ingestion_service.loader = MagicMock()
    ingestion_service.cleaner = MagicMock()
    ingestion_service.chunker = MagicMock()
    ingestion_service.vector_store = MagicMock()
    ingestion_service.registry = MagicMock()

    # Mock DocumentStorage
    storage_mock = MagicMock()

    class MockDocumentStorage:
        def __init__(self, file_path=None):
            self.file_path = file_path

        def save_documents(self, documents):
            storage_mock.save_documents(documents)
            return len(documents)

    monkeypatch.setattr(
        ingestion_module,
        "DocumentStorage",
        MockDocumentStorage
    )

    return ingestion_service, storage_mock


def test_ingest_file_success(service):
    ingestion_service, storage_mock = service

    file_path = Path("data/raw/exam_rules.pdf")
    document_id = "DOC_123"

    documents = [
        Document(page_content="Exam rules for students"),
        Document(page_content="Examination fee is Rs. 2000")
    ]

    cleaned_docs = [
        Document(page_content="Exam rules for students"),
        Document(page_content="Examination fee is Rs. 2000")
    ]

    chunks = [
        Document(page_content="Exam rules"),
        Document(page_content="Examination fee is Rs. 2000")
    ]

    ingestion_service.loader.load_file.return_value = documents
    ingestion_service.cleaner.clean_documents.return_value = cleaned_docs
    ingestion_service.chunker.chunk_documents.return_value = chunks
    ingestion_service.vector_store.add_documents.return_value = [
        "id_1", "id_2"
    ]

    result = ingestion_service.ingest_file(file_path, document_id)

    # Verify pipeline execution
    ingestion_service.loader.load_file.assert_called_once_with(file_path)

    ingestion_service.cleaner.clean_documents.assert_called_once_with(
        documents
    )

    storage_mock.save_documents.assert_called_once_with(cleaned_docs)

    ingestion_service.chunker.chunk_documents.assert_called_once_with(
        cleaned_docs
    )

    ingestion_service.vector_store.add_documents.assert_called_once_with(
        chunks
    )

    # Verify registry status updates
    assert ingestion_service.registry.update_document_status.call_count == 2

    calls = ingestion_service.registry.update_document_status.call_args_list

    assert calls[0] == call(
        document_id=document_id,
        status="processing"
    )

    assert calls[1].kwargs["document_id"] == document_id
    assert calls[1].kwargs["status"] == "completed"
    assert calls[1].kwargs["processed_at"] is not None

    # Verify response
    assert result["document_id"] == document_id
    assert result["documents_loaded"] == 2
    assert result["documents_cleaned"] == 2
    assert result["chunks_created"] == 2
    assert result["chunks_indexed"] == 2
    assert result["status"] == "completed"
    assert result["processed_at"] is not None


def test_ingest_file_with_no_extracted_documents(service):
    ingestion_service, storage_mock = service

    ingestion_service.loader.load_file.return_value = []

    with pytest.raises(ValueError, match="No content extracted"):
        ingestion_service.ingest_file(
            Path("data/raw/empty.pdf"),
            "DOC_456"
        )

    ingestion_service.cleaner.clean_documents.assert_not_called()
    storage_mock.save_documents.assert_not_called()
    ingestion_service.vector_store.add_documents.assert_not_called()

    # Failed status should be recorded
    calls = ingestion_service.registry.update_document_status.call_args_list

    assert calls[-1].kwargs["status"] == "failed"


def test_ingest_file_with_no_cleaned_documents(service):
    ingestion_service, storage_mock = service

    documents = [
        Document(page_content="Some extracted content")
    ]

    ingestion_service.loader.load_file.return_value = documents
    ingestion_service.cleaner.clean_documents.return_value = []

    with pytest.raises(ValueError, match="No valid content"):
        ingestion_service.ingest_file(
            Path("data/raw/exam.pdf"),
            "DOC_789"
        )

    storage_mock.save_documents.assert_not_called()
    ingestion_service.chunker.chunk_documents.assert_not_called()
    ingestion_service.vector_store.add_documents.assert_not_called()


def test_ingest_file_with_no_chunks(service):
    ingestion_service, storage_mock = service

    documents = [
        Document(page_content="Some extracted content")
    ]

    ingestion_service.loader.load_file.return_value = documents
    ingestion_service.cleaner.clean_documents.return_value = documents
    ingestion_service.chunker.chunk_documents.return_value = []

    with pytest.raises(ValueError, match="No chunks created"):
        ingestion_service.ingest_file(
            Path("data/raw/exam.pdf"),
            "DOC_101"
        )

    storage_mock.save_documents.assert_called_once_with(documents)
    ingestion_service.vector_store.add_documents.assert_not_called()


def test_ingest_file_when_loader_fails(service):
    ingestion_service, storage_mock = service

    ingestion_service.loader.load_file.side_effect = RuntimeError(
        "Docling extraction failed"
    )

    with pytest.raises(RuntimeError, match="Docling extraction failed"):
        ingestion_service.ingest_file(
            Path("data/raw/exam.pdf"),
            "DOC_202"
        )

    # Verify failed status
    calls = ingestion_service.registry.update_document_status.call_args_list

    assert calls[0].kwargs["status"] == "processing"
    assert calls[1].kwargs["status"] == "failed"

    storage_mock.save_documents.assert_not_called()
    ingestion_service.vector_store.add_documents.assert_not_called()
