
import pytest
from unittest.mock import MagicMock, patch

from app.services.ingestion_service import IngestionService


@pytest.fixture
def mock_components():
    with (
        patch("app.services.ingestion_service.DataLoader") as loader,
        patch("app.services.ingestion_service.DocumentCleaner") as cleaner,
        patch("app.services.ingestion_service.DocumentStorage") as storage,
        patch("app.services.ingestion_service.DocumentChunker") as chunker,
        patch("app.services.ingestion_service.TextEmbedder") as embedder,
        patch("app.services.ingestion_service.VectorStore") as vector_store,
    ):
        # Configure embedding model
        embedder.return_value.embedding_model = MagicMock()

        service = IngestionService()

        yield {
            "service": service,
            "loader": loader.return_value,
            "cleaner": cleaner.return_value,
            "storage": storage.return_value,
            "chunker": chunker.return_value,
            "embedder": embedder.return_value,
            "vector_store": vector_store.return_value,
            "vector_store_class": vector_store,
        }


@pytest.fixture
def sample_documents():
    return [
        MagicMock(page_content="Examination fee is Rs. 2000"),
        MagicMock(page_content="Courses offered by ABC College"),
    ]


@pytest.fixture
def sample_cleaned_documents():
    return [
        MagicMock(page_content="Examination fee is Rs. 2000"),
        MagicMock(page_content="Courses offered by ABC College"),
    ]


@pytest.fixture
def sample_chunks():
    return [
        MagicMock(page_content="Examination fee is Rs. 2000"),
        MagicMock(page_content="Courses offered by ABC College"),
        MagicMock(page_content="Admission details"),
    ]


# --------------------------------------------------
# 1. Initialization tests
# --------------------------------------------------

def test_ingestion_service_initialization(mock_components):
    service = mock_components["service"]

    assert service.loader is not None
    assert service.cleaner is not None
    assert service.storage is not None
    assert service.chunker is not None
    assert service.embedder is not None
    assert service.vector_store is not None


def test_vector_store_receives_embedding_model(mock_components):
    mock_components["vector_store_class"].assert_called_once_with(
        embedding=mock_components["embedder"].embedding_model
    )


# --------------------------------------------------
# 2. Successful ingestion test
# --------------------------------------------------

def test_successful_ingestion(
    mock_components,
    sample_documents,
    sample_cleaned_documents,
    sample_chunks,
):
    service = mock_components["service"]

    mock_components["loader"].load_all_documents.return_value = (
        sample_documents
    )

    mock_components["cleaner"].clean_documents.return_value = (
        sample_cleaned_documents
    )

    mock_components["chunker"].chunk_documents.return_value = (
        sample_chunks
    )

    mock_components["vector_store"].add_documents.return_value = [
        "id1", "id2", "id3"
    ]

    result = service.ingest_documents()

    assert result == {
        "documents_loaded": 2,
        "documents_cleaned": 2,
        "chunks_created": 3,
        "chunks_indexed": 3,
    }

    mock_components["loader"].load_all_documents.assert_called_once()
    mock_components["cleaner"].clean_documents.assert_called_once_with(
        sample_documents
    )
    mock_components["storage"].save_documents.assert_called_once_with(
        sample_cleaned_documents
    )
    mock_components["chunker"].chunk_documents.assert_called_once_with(
        sample_cleaned_documents
    )
    mock_components["vector_store"].add_documents.assert_called_once_with(
        sample_chunks
    )


# --------------------------------------------------
# 3. Empty document tests
# --------------------------------------------------

def test_no_documents_found(mock_components):
    service = mock_components["service"]

    mock_components["loader"].load_all_documents.return_value = []

    result = service.ingest_documents()

    assert result == {
        "documents_loaded": 0,
        "documents_cleaned": 0,
        "chunks_created": 0,
        "chunks_indexed": 0,
    }

    mock_components["cleaner"].clean_documents.assert_not_called()
    mock_components["storage"].save_documents.assert_not_called()
    mock_components["chunker"].chunk_documents.assert_not_called()
    mock_components["vector_store"].add_documents.assert_not_called()


def test_no_cleaned_documents(mock_components, sample_documents):
    service = mock_components["service"]

    mock_components["loader"].load_all_documents.return_value = (
        sample_documents
    )

    mock_components["cleaner"].clean_documents.return_value = []

    result = service.ingest_documents()

    assert result == {
        "documents_loaded": 2,
        "documents_cleaned": 0,
        "chunks_created": 0,
        "chunks_indexed": 0,
    }

    mock_components["storage"].save_documents.assert_not_called()
    mock_components["chunker"].chunk_documents.assert_not_called()


def test_no_chunks_created(
    mock_components,
    sample_documents,
    sample_cleaned_documents,
):
    service = mock_components["service"]

    mock_components["loader"].load_all_documents.return_value = (
        sample_documents
    )

    mock_components["cleaner"].clean_documents.return_value = (
        sample_cleaned_documents
    )

    mock_components["chunker"].chunk_documents.return_value = []

    result = service.ingest_documents()

    assert result == {
        "documents_loaded": 2,
        "documents_cleaned": 2,
        "chunks_created": 0,
        "chunks_indexed": 0,
    }

    mock_components["storage"].save_documents.assert_called_once()
    mock_components["vector_store"].add_documents.assert_not_called()


# --------------------------------------------------
# 4. Failure handling tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "component,method",
    [
        ("loader", "load_all_documents"),
        ("cleaner", "clean_documents"),
        ("storage", "save_documents"),
        ("chunker", "chunk_documents"),
        ("vector_store", "add_documents"),
    ],
)
def test_ingestion_component_failure(
    mock_components,
    sample_documents,
    sample_cleaned_documents,
    sample_chunks,
    component,
    method,
):
    service = mock_components["service"]

    # Configure successful preceding pipeline stages
    mock_components["loader"].load_all_documents.return_value = (
        sample_documents
    )
    mock_components["cleaner"].clean_documents.return_value = (
        sample_cleaned_documents
    )
    mock_components["chunker"].chunk_documents.return_value = (
        sample_chunks
    )
    mock_components["vector_store"].add_documents.return_value = [
        "id1", "id2", "id3"
    ]

    # Make the selected component fail
    target = mock_components[component]
    getattr(target, method).side_effect = RuntimeError(
        f"{component} failed"
    )

    with pytest.raises(RuntimeError, match=f"{component} failed"):
        service.ingest_documents()


# --------------------------------------------------
# 5. Initialization failure test
# --------------------------------------------------

def test_ingestion_service_initialization_failure():
    with patch(
        "app.services.ingestion_service.DataLoader",
        side_effect=RuntimeError("Loader initialization failed"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Loader initialization failed",
        ):
            IngestionService()
