
from langchain_core.documents import Document

from app.core.config import settings
from app.ingestion.embeddings import TextEmbedder
from app.ingestion.indexer import VectorStore


def test_chroma_persistence_and_duplicate_indexing(tmp_path, monkeypatch):
    # Use a temporary directory instead of the actual ChromaDB directory
    monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path)

    # Initialize the actual Hugging Face embedding model
    embedder = TextEmbedder()

    # Initialize real ChromaDB
    vector_store = VectorStore(embedder.embedding_model)

    # Sample documents
    documents = [
        Document(
            page_content="ABC College offers B.Tech, BBA and MBA courses.",
            metadata={"source": "courses.pdf"}
        ),
        Document(
            page_content="The examination fee for 2026-2027 is Rs. 2000.",
            metadata={"source": "fees.pdf"}
        )
    ]

    # First indexing
    first_ids = vector_store.add_documents(documents)

    first_count = len(vector_store.chroma_store.get()["ids"])

    assert len(first_ids) == 2
    assert first_count == 2

    # Second indexing: same documents
    second_ids = vector_store.add_documents(documents)

    second_count = len(vector_store.chroma_store.get()["ids"])

    # IDs should remain the same
    assert first_ids == second_ids

    # Collection count should not increase
    assert second_count == first_count

    # Reopen ChromaDB using the same persistence directory
    reopened_store = VectorStore(embedder.embedding_model)

    reopened_count = len(reopened_store.chroma_store.get()["ids"])

    # Data should still exist after reopening
    assert reopened_count == 2

    # Verify that similarity search can retrieve a stored document
    results = reopened_store.chroma_store.similarity_search(
        "What courses does ABC College offer?",
        k=1
    )

    assert len(results) == 1
    assert "B.Tech" in results[0].page_content
