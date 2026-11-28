
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from app.services.retrieval_service import RetrievalService


# Mock Retriever
@pytest.fixture
def mock_retriever():
    with patch("app.services.retrieval_service.Retriever") as mock:
        yield mock


# Mock Reranker
@pytest.fixture
def mock_reranker():
    with patch("app.services.retrieval_service.Reranker") as mock:
        yield mock


# Initialize RetrievalService
@pytest.fixture
def retrieval_service(mock_retriever, mock_reranker):
    return RetrievalService()


# Sample documents
@pytest.fixture
def sample_documents():
    return [
        Document(page_content="ABC College offers B.Tech courses."),
        Document(page_content="The examination fee is Rs. 2000."),
        Document(page_content="The college library opens at 9 AM."),
    ]


# 1. Test initialization
def test_retrieval_service_initialization(
    retrieval_service,
    mock_retriever,
    mock_reranker
):
    mock_retriever.assert_called_once()
    mock_reranker.assert_called_once()

    assert retrieval_service.retriever is not None
    assert retrieval_service.reranker is not None


# 2. Test successful retrieval pipeline
def test_successful_retrieval(
    retrieval_service,
    sample_documents
):
    retrieval_service.retriever.retrieve.return_value = sample_documents

    expected_results = [
        (sample_documents[1], 0.95),
        (sample_documents[0], 0.80),
        (sample_documents[2], 0.60),
    ]

    retrieval_service.reranker.re_rank.return_value = expected_results

    results = retrieval_service.retrieve_relevant_documents(
        "What is the examination fee?"
    )

    retrieval_service.retriever.retrieve.assert_called_once_with(
        "What is the examination fee?",
        k=10
    )

    retrieval_service.reranker.re_rank.assert_called_once_with(
        "What is the examination fee?",
        sample_documents,
        k=3
    )

    assert results == expected_results


# 3. Test custom retrieve_k and rerank_k
def test_custom_k_values(
    retrieval_service,
    sample_documents
):
    retrieval_service.retriever.retrieve.return_value = sample_documents
    retrieval_service.reranker.re_rank.return_value = []

    retrieval_service.retrieve_relevant_documents(
        "Tell me about B.Tech",
        retrieve_k=20,
        rerank_k=5
    )

    retrieval_service.retriever.retrieve.assert_called_once_with(
        "Tell me about B.Tech",
        k=20
    )

    retrieval_service.reranker.re_rank.assert_called_once_with(
        "Tell me about B.Tech",
        sample_documents,
        k=5
    )


# 4. Test empty query
def test_empty_query(retrieval_service):
    with pytest.raises(ValueError):
        retrieval_service.retrieve_relevant_documents("")


# 5. Test whitespace-only query
def test_whitespace_query(retrieval_service):
    with pytest.raises(ValueError):
        retrieval_service.retrieve_relevant_documents("   ")


# 6. Test invalid retrieve_k
def test_invalid_retrieve_k(retrieval_service):
    with pytest.raises(ValueError):
        retrieval_service.retrieve_relevant_documents(
            "What is the examination fee?",
            retrieve_k=0
        )


# 7. Test negative retrieve_k
def test_negative_retrieve_k(retrieval_service):
    with pytest.raises(ValueError):
        retrieval_service.retrieve_relevant_documents(
            "What is the examination fee?",
            retrieve_k=-1
        )


# 8. Test invalid rerank_k
def test_invalid_rerank_k(retrieval_service):
    with pytest.raises(ValueError):
        retrieval_service.retrieve_relevant_documents(
            "What is the examination fee?",
            rerank_k=0
        )


# 9. Test negative rerank_k
def test_negative_rerank_k(retrieval_service):
    with pytest.raises(ValueError):
        retrieval_service.retrieve_relevant_documents(
            "What is the examination fee?",
            rerank_k=-1
        )


# 10. Test no documents retrieved
def test_no_documents_retrieved(retrieval_service):
    retrieval_service.retriever.retrieve.return_value = []

    results = retrieval_service.retrieve_relevant_documents(
        "What is the examination fee?"
    )

    assert results == []

    retrieval_service.reranker.re_rank.assert_not_called()


# 11. Test retriever failure
def test_retriever_failure(retrieval_service):
    retrieval_service.retriever.retrieve.side_effect = RuntimeError(
    "Pinecone retrieval failed"
    )

    with pytest.raises(RuntimeError):
        retrieval_service.retrieve_relevant_documents(
            "What is the examination fee?"
        )


# 12. Test reranker failure
def test_reranker_failure(
    retrieval_service,
    sample_documents
):
    retrieval_service.retriever.retrieve.return_value = sample_documents

    retrieval_service.reranker.re_rank.side_effect = RuntimeError(
        "CrossEncoder reranking failed"
    )

    with pytest.raises(RuntimeError):
        retrieval_service.retrieve_relevant_documents(
            "What is the examination fee?"
        )


# 13. Test initialization failure
def test_retriever_initialization_failure(mock_retriever, mock_reranker):
    mock_retriever.side_effect = RuntimeError(
        "Retriever initialization failed"
    )

    with pytest.raises(RuntimeError):
        RetrievalService()
