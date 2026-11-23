
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from app.retrieval.reranker import Reranker


@pytest.fixture
def mock_cross_encoder():
    with patch("app.retrieval.reranker.CrossEncoder") as mock:
        yield mock


@pytest.fixture
def reranker(mock_cross_encoder):
    instance = Reranker()
    return instance


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="ABC College offers B.Tech courses."),
        Document(page_content="The examination fee is Rs. 2000."),
        Document(page_content="The college library opens at 9 AM."),
    ]


# 1. Test initialization
def test_reranker_initialization(reranker, mock_cross_encoder):
    mock_cross_encoder.assert_called_once()
    assert reranker.cross_encoder is not None


# 2. Test successful reranking
def test_successful_reranking(reranker, sample_documents):
    reranker.cross_encoder.predict.return_value = [0.3, 0.9, 0.5]

    results = reranker.re_rank(
        "What is the examination fee?",
        sample_documents
    )

    assert len(results) == 3
    assert results[0][0].page_content == "The examination fee is Rs. 2000."
    assert results[0][1] == 0.9
    assert results[1][1] == 0.5
    assert results[2][1] == 0.3


# 3. Test top-k selection
def test_top_k_selection(reranker, sample_documents):
    reranker.cross_encoder.predict.return_value = [0.3, 0.9, 0.5]

    results = reranker.re_rank(
        "What is the examination fee?",
        sample_documents,
        k=2
    )

    assert len(results) == 2
    assert results[0][1] == 0.9
    assert results[1][1] == 0.5


# 4. Test empty query
def test_empty_query(reranker, sample_documents):
    with pytest.raises(ValueError):
        reranker.re_rank("", sample_documents)


# 5. Test whitespace-only query
def test_whitespace_query(reranker, sample_documents):
    with pytest.raises(ValueError):
        reranker.re_rank("   ", sample_documents)


# 6. Test empty documents
def test_empty_documents(reranker):
    results = reranker.re_rank(
        "What is the examination fee?",
        []
    )

    assert results == []
    reranker.cross_encoder.predict.assert_not_called()


# 7. Test invalid k = 0
def test_invalid_k_zero(reranker, sample_documents):
    with pytest.raises(ValueError):
        reranker.re_rank(
            "What is the examination fee?",
            sample_documents,
            k=0
        )


# 8. Test negative k
def test_invalid_negative_k(reranker, sample_documents):
    with pytest.raises(ValueError):
        reranker.re_rank(
            "What is the examination fee?",
            sample_documents,
            k=-1
        )


# 9. Test CrossEncoder prediction failure
def test_prediction_failure(reranker, sample_documents):
    reranker.cross_encoder.predict.side_effect = RuntimeError(
        "Model prediction failed"
    )

    with pytest.raises(RuntimeError):
        reranker.re_rank(
            "What is the examination fee?",
            sample_documents
        )
