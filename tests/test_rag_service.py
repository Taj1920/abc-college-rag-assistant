
import pytest
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.services.rag_service import RAGService


# Mock RetrievalService
@pytest.fixture
def mock_retrieval_service():
    with patch(
        "app.services.rag_service.RetrievalService"
    ) as mock:
        yield mock


# Mock LLM
@pytest.fixture
def mock_llm():
    with patch(
        "app.services.rag_service.ChatOpenAI"
    ) as mock:
        yield mock


# Initialize RAGService
@pytest.fixture
def rag_service(mock_retrieval_service, mock_llm):
    return RAGService()


# Sample documents
@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Examination fee is Rs. 2000.",
            metadata={"source": "exam_fee.pdf"}
        ),
        Document(
            page_content="Fee applies for academic year 2026-2027.",
            metadata={"source": "academic_rules.pdf"}
        ),
    ]


# Sample reranked results
@pytest.fixture
def sample_reranked_results(sample_documents):
    return [
        (sample_documents[0], 0.95),
        (sample_documents[1], 0.85),
    ]


# 1. Test initialization
def test_rag_service_initialization(
    rag_service,
    mock_retrieval_service,
    mock_llm
):
    mock_retrieval_service.assert_called_once()
    mock_llm.assert_called_once()

    assert rag_service.retrieval_service is not None
    assert rag_service.llm is not None


# 2. Test successful prompt building
def test_build_prompt_success(rag_service, sample_documents):

    prompt = rag_service.build_prompt(
        sample_documents,
        "What is the examination fee?"
    )

    messages = prompt.to_messages()

    assert len(messages) == 2
    assert "Examination fee is Rs. 2000." in messages[0].content
    assert "What is the examination fee?" in messages[1].content


# 3. Test empty question
def test_build_prompt_empty_question(rag_service, sample_documents):

    with pytest.raises(ValueError):
        rag_service.build_prompt(sample_documents, "")


# 4. Test whitespace-only question
def test_build_prompt_whitespace_question(
    rag_service,
    sample_documents
):

    with pytest.raises(ValueError):
        rag_service.build_prompt(sample_documents, "   ")


# 5. Test empty documents
def test_build_prompt_empty_documents(rag_service):

    with pytest.raises(ValueError):
        rag_service.build_prompt(
            [],
            "What is the examination fee?"
        )


# 6. Test successful response generation
def test_generate_response_success(
    rag_service,
    sample_reranked_results
):

    rag_service.retrieval_service.retrieve_relevant_documents.return_value = (
        sample_reranked_results
    )

    rag_service.llm.invoke.return_value = AIMessage(
        content="The examination fee is Rs. 2000."
    )

    response = rag_service.generate_response(
        "What is the examination fee?"
    )

    assert response == "The examination fee is Rs. 2000."

    rag_service.retrieval_service.retrieve_relevant_documents.assert_called_once_with(
        "What is the examination fee?",
        10,
        3
    )

    rag_service.llm.invoke.assert_called_once()


# 7. Test custom retrieval parameters
def test_generate_response_custom_k(
    rag_service,
    sample_reranked_results
):

    rag_service.retrieval_service.retrieve_relevant_documents.return_value = (
        sample_reranked_results
    )

    rag_service.llm.invoke.return_value = AIMessage(
        content="Sample answer"
    )

    rag_service.generate_response(
        "Tell me about college fees",
        retrieve_k=20,
        rerank_k=5
    )

    rag_service.retrieval_service.retrieve_relevant_documents.assert_called_once_with(
        "Tell me about college fees",
        20,
        5
    )


# 8. Test empty query
def test_generate_response_empty_query(rag_service):

    with pytest.raises(ValueError):
        rag_service.generate_response("")


# 9. Test whitespace-only query
def test_generate_response_whitespace_query(rag_service):

    with pytest.raises(ValueError):
        rag_service.generate_response("   ")


# 10. Test no retrieved documents
def test_generate_response_no_documents(rag_service):

    rag_service.retrieval_service.retrieve_relevant_documents.return_value = []

    response = rag_service.generate_response(
        "What is the examination fee?"
    )

    assert response == (
        "I don't have enough information to answer your question."
    )

    rag_service.llm.invoke.assert_not_called()


# 11. Test retrieval failure
def test_generate_response_retrieval_failure(rag_service):

    rag_service.retrieval_service.retrieve_relevant_documents.side_effect = (
        RuntimeError("Retrieval failed")
    )

    with pytest.raises(RuntimeError):
        rag_service.generate_response(
            "What is the examination fee?"
        )


# 12. Test LLM failure
def test_generate_response_llm_failure(
    rag_service,
    sample_reranked_results
):

    rag_service.retrieval_service.retrieve_relevant_documents.return_value = (
        sample_reranked_results
    )

    rag_service.llm.invoke.side_effect = RuntimeError(
        "LLM generation failed"
    )

    with pytest.raises(RuntimeError):
        rag_service.generate_response(
            "What is the examination fee?"
        )


# 13. Test initialization failure
def test_retrieval_service_initialization_failure(
    mock_retrieval_service,
    mock_llm
):

    mock_retrieval_service.side_effect = RuntimeError(
        "Retrieval service initialization failed"
    )

    with pytest.raises(RuntimeError):
        RAGService()
