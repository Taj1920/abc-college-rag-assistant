import os
import time
import uuid

import pytest
from langchain_core.documents import Document

from app.ingestion.embeddings import TextEmbedder
from app.ingestion.indexer import VectorStore


RUN_INTEGRATION = os.getenv("RUN_PINECONE_INTEGRATION", "").strip()

pytestmark = pytest.mark.skipif(
    RUN_INTEGRATION != "1",
    reason=f"Integration test disabled. Current value: {RUN_INTEGRATION!r}"
)


def wait_for_vector_count(index, namespace, expected_count, timeout=60):
    """Wait until Pinecone reports the expected vector count."""

    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout:
        stats = index.describe_index_stats()

        namespace_stats = stats.namespaces.get(namespace)
        actual_count = (
            namespace_stats.vector_count if namespace_stats else 0
        )

        if actual_count == expected_count:
            return actual_count

        time.sleep(2)

    raise AssertionError(
        f"Expected {expected_count} vectors, "
        f"but Pinecone reported {actual_count}."
    )


def test_pinecone_indexing_and_similarity_search():

    namespace = f"pytest-{uuid.uuid4().hex}"

    # Initialize actual embedding model
    embedder = TextEmbedder()

    # Connect to actual Pinecone
    vector_store = VectorStore(embedder.embedding_model)

    # Access the underlying Pinecone index for stats and cleanup
    index = vector_store.pinecone_store.index

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

    try:
        # First indexing
        first_ids = vector_store.add_documents(
            documents,
            namespace=namespace
        )

        assert len(first_ids) == 2

        first_count = wait_for_vector_count(
            index, namespace, expected_count=2
        )

        assert first_count == 2

        # Second indexing: same documents
        second_ids = vector_store.add_documents(
            documents,
            namespace=namespace
        )

        assert first_ids == second_ids

        second_count = wait_for_vector_count(
            index, namespace, expected_count=2
        )

        assert second_count == first_count

        # Reconnect to Pinecone
        reopened_store = VectorStore(embedder.embedding_model)

        # Similarity search
        results = reopened_store.pinecone_store.similarity_search(
            query="What courses does ABC College offer?",
            k=1,
            namespace=namespace
        )

        assert len(results) == 1
        assert "B.Tech" in results[0].page_content

        # Verify metadata
        assert results[0].metadata["source"] == "courses.pdf"

    finally:
        # Delete only this test's namespace
        index.delete(
            delete_all=True,
            namespace=namespace
        )
