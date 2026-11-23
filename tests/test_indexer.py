
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import app.ingestion.indexer as indexer
from app.ingestion.indexer import VectorStore


# Sample documents for testing
def create_document(
    content="Examination fee is Rs. 2000",
    source="fees.pdf"
):
    return Document(
        page_content=content,
        metadata={"source": source}
    )


# Fixture: Mock ChromaDB and initialize VectorStore
@pytest.fixture
def vector_store(monkeypatch, tmp_path):

    mock_chroma_store = MagicMock()

    mock_chroma = MagicMock(return_value=mock_chroma_store)

    monkeypatch.setattr(indexer, "Chroma", mock_chroma)
    monkeypatch.setattr(indexer, "ROOT", tmp_path)

    # Avoid modifying the actual project settings
    from types import SimpleNamespace

    monkeypatch.setattr(
        indexer,
        "settings",
        SimpleNamespace(chroma_persist_dir="test_chroma_db")
    )

    store = VectorStore(embedding=MagicMock())

    return store, mock_chroma_store, mock_chroma


# 1. Test VectorStore initialization
def test_vector_store_initialization(vector_store):

    store, mock_db, mock_chroma = vector_store

    assert store.chroma_store is mock_db
    mock_chroma.assert_called_once()

    kwargs = mock_chroma.call_args.kwargs

    assert kwargs["collection_name"] == "smartcampus_documents"
    assert kwargs["embedding_function"] is not None


# 2. Test document ID generation
def test_generate_document_id():

    document = create_document()

    doc_id = VectorStore.generate_document_id(document)

    assert isinstance(doc_id, str)
    assert len(doc_id) == 64


# 3. Same document should generate same ID
def test_same_document_generates_same_id():

    doc1 = create_document()
    doc2 = create_document()

    id1 = VectorStore.generate_document_id(doc1)
    id2 = VectorStore.generate_document_id(doc2)

    assert id1 == id2


# 4. Different source should generate different IDs
def test_different_source_generates_different_id():

    doc1 = create_document(source="fees.pdf")
    doc2 = create_document(source="hostel.pdf")

    id1 = VectorStore.generate_document_id(doc1)
    id2 = VectorStore.generate_document_id(doc2)

    assert id1 != id2


# 5. Different content should generate different IDs
def test_different_content_generates_different_id():

    doc1 = create_document(content="Examination fee is Rs. 2000")
    doc2 = create_document(content="Hostel fee is Rs. 60000")

    id1 = VectorStore.generate_document_id(doc1)
    id2 = VectorStore.generate_document_id(doc2)

    assert id1 != id2


# 6. Empty document list should return empty list
def test_add_empty_documents(vector_store):

    store, mock_db, _ = vector_store

    result = store.add_documents([])

    assert result == []
    mock_db.add_documents.assert_not_called()


# 7. Test successful document indexing
def test_add_documents_successfully(vector_store):

    store, mock_db, _ = vector_store

    documents = [
        create_document("Examination fee is Rs. 2000"),
        create_document("Hostel fee is Rs. 60000")
    ]

    mock_db.add_documents.return_value = ["id1", "id2"]

    result = store.add_documents(documents)

    assert result == ["id1", "id2"]
    mock_db.add_documents.assert_called_once()

    kwargs = mock_db.add_documents.call_args.kwargs

    assert len(kwargs["documents"]) == 2
    assert len(kwargs["ids"]) == 2


# 8. Duplicate chunks should be removed before indexing
def test_duplicate_documents_are_removed(vector_store):

    store, mock_db, _ = vector_store

    doc1 = create_document()
    doc2 = create_document()

    mock_db.add_documents.return_value = ["id1"]

    result = store.add_documents([doc1, doc2])

    kwargs = mock_db.add_documents.call_args.kwargs

    assert len(kwargs["documents"]) == 1
    assert len(kwargs["ids"]) == 1
    assert result == ["id1"]


# 9. Repeated indexing should reuse the same IDs
def test_repeated_indexing_uses_same_ids(vector_store):

    store, mock_db, _ = vector_store

    documents = [
        create_document("Examination fee is Rs. 2000"),
        create_document("Hostel fee is Rs. 60000")
    ]

    mock_db.add_documents.side_effect = [
        ["id1", "id2"],
        ["id1", "id2"]
    ]

    first_result = store.add_documents(documents)
    second_result = store.add_documents(documents)

    first_ids = mock_db.add_documents.call_args_list[0].kwargs["ids"]
    second_ids = mock_db.add_documents.call_args_list[1].kwargs["ids"]

    assert first_result == second_result
    assert first_ids == second_ids
    assert mock_db.add_documents.call_count == 2


# 10. Test ChromaDB indexing failure
def test_add_documents_failure(vector_store):

    store, mock_db, _ = vector_store

    mock_db.add_documents.side_effect = RuntimeError(
        "ChromaDB connection failed"
    )

    documents = [create_document()]

    with pytest.raises(RuntimeError, match="ChromaDB connection failed"):
        store.add_documents(documents)


# 11. Test ChromaDB initialization failure
def test_vector_store_initialization_failure(monkeypatch):

    from types import SimpleNamespace

    monkeypatch.setattr(
        indexer,
        "settings",
        SimpleNamespace(chroma_persist_dir="test_chroma_db")
    )

    monkeypatch.setattr(
        indexer,
        "Chroma",
        MagicMock(side_effect=RuntimeError("Initialization failed"))
    )

    with pytest.raises(RuntimeError, match="Initialization failed"):
        VectorStore(embedding=MagicMock())
