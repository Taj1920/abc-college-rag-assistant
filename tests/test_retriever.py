
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from app.retrieval.retriever import Retriever


@pytest.fixture
def retriever():
    with patch("app.retrieval.retriever.TextEmbedder") as mock_embedder, \
         patch("app.retrieval.retriever.VectorStore") as mock_vectorstore:

        mock_embedder.return_value.embedding_model = MagicMock()

        mock_store = MagicMock()
        mock_vectorstore.return_value.pinecone_store = mock_store

        instance = Retriever()

        return instance


def test_retriever_initialization(retriever):
    assert retriever.pinecone_store is not None


def test_successful_retrieval(retriever):
    expected_docs = [
        Document(
            page_content="Examination fee for 2026-2027 is Rs. 2000.",
            metadata={"source": "fees.pdf"}
        ),
        Document(
            page_content="Students must pay examination fees before exams.",
            metadata={"source": "rules.pdf"}
        )
    ]

    retriever.pinecone_store.similarity_search.return_value = expected_docs

    results = retriever.retrieve(
        "What is the examination fee?",
        k=2
    )

    assert len(results) == 2
    assert results == expected_docs

    retriever.pinecone_store.similarity_search.assert_called_once_with(
        query="What is the examination fee?",
        k=2
    )


def test_empty_query_raises_error(retriever):
    with pytest.raises(ValueError, match="Users query should not be empty"):
        retriever.retrieve("")


def test_whitespace_query_raises_error(retriever):
    with pytest.raises(ValueError, match="Users query should not be empty"):
        retriever.retrieve("     ")


def test_invalid_k_raises_error(retriever):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        retriever.retrieve(
            "What is the examination fee?",
            k=0
        )


def test_negative_k_raises_error(retriever):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        retriever.retrieve(
            "What is the examination fee?",
            k=-1
        )


def test_retrieval_failure(retriever):
    retriever.pinecone_store.similarity_search.side_effect = Exception(
        "Pinecone retrieval failed"
    )

    with pytest.raises(Exception, match="Pinecone retrieval failed"):
        retriever.retrieve("What is the examination fee?")
