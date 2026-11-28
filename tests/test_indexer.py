
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from pydantic import SecretStr

import app.ingestion.indexer as indexer


@pytest.fixture
def pinecone_mocks(monkeypatch):
    mock_index = MagicMock()

    mock_client = MagicMock()
    mock_client.has_index.return_value = True
    mock_client.Index.return_value = mock_index

    mock_store = MagicMock()
    mock_store.add_documents.return_value = ["id1", "id2"]

    mock_pinecone = MagicMock(return_value=mock_client)
    mock_vector_store = MagicMock(return_value=mock_store)

    # Replace external dependencies
    monkeypatch.setattr(indexer, "Pinecone", mock_pinecone)
    monkeypatch.setattr(indexer, "PineconeVectorStore", mock_vector_store)

    # Use test settings
    monkeypatch.setattr(
        indexer,
        "settings",
        SimpleNamespace(
            pinecone_index_name="test-index",
            pinecone_api_key=SecretStr("test-api-key"),
        ),
    )

    return {
        "client": mock_client,
        "index": mock_index,
        "store": mock_store,
        "pinecone": mock_pinecone,
        "vector_store": mock_vector_store,
    }


@pytest.fixture
def embedding():
    return MagicMock()


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="What is machine learning?",
            metadata={"source": "notes.pdf"},
        ),
        Document(
            page_content="What is deep learning?",
            metadata={"source": "notes.pdf"},
        ),
    ]


def test_vector_store_initialization(pinecone_mocks, embedding):
    store = indexer.VectorStore(embedding)

    pinecone_mocks["pinecone"].assert_called_once_with(
    api_key="test-api-key"
    )

    pinecone_mocks["client"].has_index.assert_called_once_with("test-index")

    pinecone_mocks["client"].Index.assert_called_once_with("test-index")

    pinecone_mocks["vector_store"].assert_called_once_with(
        index=pinecone_mocks["index"],
        embedding=embedding,
    )

    assert store.pinecone_store == pinecone_mocks["store"]


def test_vector_store_creates_index_if_not_exists(
    pinecone_mocks, embedding
):
    pinecone_mocks["client"].has_index.return_value = False

    indexer.VectorStore(embedding)

    pinecone_mocks["client"].create_index.assert_called_once()

    kwargs = pinecone_mocks["client"].create_index.call_args.kwargs

    assert kwargs["name"] == "test-index"
    assert kwargs["dimension"] == 384
    assert kwargs["metric"] == "cosine"


def test_vector_store_does_not_create_existing_index(
    pinecone_mocks, embedding
):
    pinecone_mocks["client"].has_index.return_value = True

    indexer.VectorStore(embedding)

    pinecone_mocks["client"].create_index.assert_not_called()


def test_generate_document_id():
    document = Document(
        page_content="Machine learning is useful",
        metadata={"source": "notes.pdf"},
    )

    doc_id = indexer.VectorStore.generate_document_id(document)

    assert isinstance(doc_id, str)
    assert len(doc_id) == 64


def test_same_document_generates_same_id():
    document = Document(
        page_content="Machine learning",
        metadata={"source": "notes.pdf"},
    )

    id1 = indexer.VectorStore.generate_document_id(document)
    id2 = indexer.VectorStore.generate_document_id(document)

    assert id1 == id2


def test_different_source_generates_different_id():
    doc1 = Document(
        page_content="Machine learning",
        metadata={"source": "notes.pdf"},
    )

    doc2 = Document(
        page_content="Machine learning",
        metadata={"source": "book.pdf"},
    )

    assert (
        indexer.VectorStore.generate_document_id(doc1)
        != indexer.VectorStore.generate_document_id(doc2)
    )


def test_different_content_generates_different_id():
    doc1 = Document(
        page_content="Machine learning",
        metadata={"source": "notes.pdf"},
    )

    doc2 = Document(
        page_content="Deep learning",
        metadata={"source": "notes.pdf"},
    )

    assert (
        indexer.VectorStore.generate_document_id(doc1)
        != indexer.VectorStore.generate_document_id(doc2)
    )


def test_add_empty_documents(pinecone_mocks, embedding):
    store = indexer.VectorStore(embedding)

    result = store.add_documents([])

    assert result == []
    pinecone_mocks["store"].add_documents.assert_not_called()


def test_add_documents_successfully(
    pinecone_mocks, embedding, sample_documents
):
    store = indexer.VectorStore(embedding)

    result = store.add_documents(sample_documents)

    assert result == ["id1", "id2"]

    pinecone_mocks["store"].add_documents.assert_called_once()

    kwargs = pinecone_mocks["store"].add_documents.call_args.kwargs

    assert len(kwargs["documents"]) == 2
    assert len(kwargs["ids"]) == 2


def test_duplicate_documents_are_removed(
    pinecone_mocks, embedding
):
    store = indexer.VectorStore(embedding)

    document = Document(
        page_content="Machine learning",
        metadata={"source": "notes.pdf"},
    )

    result = store.add_documents([document, document])

    assert result == ["id1", "id2"]

    kwargs = pinecone_mocks["store"].add_documents.call_args.kwargs

    assert len(kwargs["documents"]) == 1
    assert len(kwargs["ids"]) == 1


def test_repeated_indexing_uses_same_ids(
    pinecone_mocks, embedding, sample_documents
):
    store = indexer.VectorStore(embedding)

    store.add_documents(sample_documents)
    first_call = pinecone_mocks["store"].add_documents.call_args.kwargs

    store.add_documents(sample_documents)
    second_call = pinecone_mocks["store"].add_documents.call_args.kwargs

    assert first_call["ids"] == second_call["ids"]


def test_add_documents_failure(pinecone_mocks, embedding, sample_documents):
    store = indexer.VectorStore(embedding)

    pinecone_mocks["store"].add_documents.side_effect = RuntimeError(
        "Pinecone indexing failed"
    )

    with pytest.raises(RuntimeError, match="Pinecone indexing failed"):
        store.add_documents(sample_documents)


def test_vector_store_initialization_failure(monkeypatch, embedding):
    def raise_error(**kwargs):
        raise RuntimeError("Pinecone initialization failed")

    monkeypatch.setattr(indexer, "Pinecone", raise_error)

    monkeypatch.setattr(
        indexer,
        "settings",
        SimpleNamespace(
            pinecone_index_name="test-index",
            pinecone_api_key=SecretStr("test-api-key"),
        ),
    )

    with pytest.raises(RuntimeError, match="Pinecone initialization failed"):
        indexer.VectorStore(embedding)
