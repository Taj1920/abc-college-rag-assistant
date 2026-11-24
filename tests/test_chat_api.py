
import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import (
    app,
    get_rag_service,
    get_rag_semaphore
)
from app.services.rag_service import RAGService


# -------------------------
# Test fixtures
# -------------------------

@pytest.fixture
def mock_rag_service():
    return MagicMock(spec=RAGService)


@pytest.fixture
def rag_semaphore():
    return asyncio.Semaphore(1)



@pytest.fixture
def client(mock_rag_service, rag_semaphore):

    app.dependency_overrides[get_rag_service] = (
        lambda: mock_rag_service
    )

    app.dependency_overrides[get_rag_semaphore] = (
        lambda: rag_semaphore
    )

    # Create TestClient without entering its context manager.
    # This prevents the real application lifespan from running.
    test_client = TestClient(app)

    yield test_client

    test_client.close()
    app.dependency_overrides.clear()



# -------------------------
# Sample response
# -------------------------

SAMPLE_RESPONSE = {
    "answer": "The examination fee is Rs. 2000.",
    "sources": [
        {
            "source": "data/raw/examination.pdf",
            "score": 0.95
        }
    ]
}


# -------------------------
# Health API tests
# -------------------------

def test_health_check(client):
    app.state.rag_service = MagicMock()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# -------------------------
# Chat API tests
# -------------------------

def test_chat_success(
    client,
    mock_rag_service
):
    mock_rag_service.generate_response.return_value = (
        SAMPLE_RESPONSE
    )

    response = client.post(
        "/api/v1/chat",
        json={
            "question": "What is the examination fee?"
        }
    )

    assert response.status_code == 200
    assert response.json() == SAMPLE_RESPONSE

    mock_rag_service.generate_response.assert_called_once_with(
        "What is the examination fee?",
        retrieve_k=10,
        rerank_k=3
    )


def test_chat_with_custom_parameters(
    client,
    mock_rag_service
):
    mock_rag_service.generate_response.return_value = (
        SAMPLE_RESPONSE
    )

    response = client.post(
        "/api/v1/chat?retrieve_k=20&rerank_k=5",
        json={
            "question": "What courses are available?"
        }
    )

    assert response.status_code == 200

    mock_rag_service.generate_response.assert_called_once_with(
        "What courses are available?",
        retrieve_k=20,
        rerank_k=5
    )


# -------------------------
# Request validation tests
# -------------------------

@pytest.mark.parametrize(
    "params",
    [
        {"retrieve_k": 0},
        {"retrieve_k": -1},
        {"rerank_k": 0},
        {"rerank_k": -1}
    ]
)
def test_invalid_retrieval_parameters(client, params):

    response = client.post(
        "/api/v1/chat",
        params=params,
        json={"question": "Tell me about college"}
    )

    assert response.status_code == 422


def test_rerank_k_greater_than_retrieve_k(client):

    response = client.post(
        "/api/v1/chat?retrieve_k=5&rerank_k=10",
        json={"question": "Tell me about college"}
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "rerank_k cannot be greater than retrieve_k"
    )


def test_empty_question(client):

    response = client.post(
        "/api/v1/chat",
        json={"question": ""}
    )

    assert response.status_code == 422


# -------------------------
# Concurrency tests
# -------------------------

def test_semaphore_timeout(client):

    # No processing slots available
    app.dependency_overrides[get_rag_semaphore] = (
        lambda: asyncio.Semaphore(0)
    )

    response = client.post(
        "/api/v1/chat",
        json={"question": "What is the fee?"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Server is busy. Please try again shortly."
    )
    assert response.headers["Retry-After"] == "2"


def test_semaphore_released_after_success(
    client,
    mock_rag_service,
    rag_semaphore
):
    mock_rag_service.generate_response.return_value = (
        SAMPLE_RESPONSE
    )

    first_response = client.post(
        "/api/v1/chat",
        json={"question": "What is the fee?"}
    )

    second_response = client.post(
        "/api/v1/chat",
        json={"question": "What courses are available?"}
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    # The processing slot should be available again.
    assert rag_semaphore._value == 1


def test_semaphore_released_after_exception(
    client,
    mock_rag_service,
    rag_semaphore
):
    mock_rag_service.generate_response.side_effect = [
        RuntimeError("Test failure"),
        SAMPLE_RESPONSE
    ]

    first_response = client.post(
        "/api/v1/chat",
        json={"question": "What is the fee?"}
    )

    second_response = client.post(
        "/api/v1/chat",
        json={"question": "What courses are available?"}
    )

    assert first_response.status_code == 500
    assert second_response.status_code == 200

    assert rag_semaphore._value == 1


# -------------------------
# RAG failure test
# -------------------------

def test_rag_service_failure(
    client,
    mock_rag_service
):
    mock_rag_service.generate_response.side_effect = (
        RuntimeError("RAG service failed")
    )

    response = client.post(
        "/api/v1/chat",
        json={"question": "What is the fee?"}
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Unable to process your question right now."
    )
