
import pytest
from unittest.mock import patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.core.config import settings
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content=(
                "The examination fee for academic year 2026-2027 "
                "is Rs. 2000. Students must pay the fee before "
                "the examination."
            ),
            metadata={"source": "examination.pdf"}
        ),
        Document(
            page_content=(
                "ABC College offers B.Tech Computer Science, "
                "B.Tech Electronics and MBA courses."
            ),
            metadata={"source": "courses.pdf"}
        ),
        Document(
            page_content=(
                "The college library operates from 9 AM to 5 PM "
                "on weekdays."
            ),
            metadata={"source": "library.pdf"}
        )
    ]


def test_end_to_end_rag_pipeline(
    tmp_path,
    monkeypatch,
    sample_documents
):
    # Use a temporary ChromaDB directory
    monkeypatch.setattr(
        settings,
        "chroma_persist_dir",
        str(tmp_path / "chroma_db")
    )

    # Mock external dependencies
    with (
        patch(
            "app.services.ingestion_service.DataLoader"
        ) as mock_loader,
        patch(
            "app.services.ingestion_service.DocumentStorage"
        ) as mock_storage,
        patch(
            "app.retrieval.reranker.CrossEncoder"
        ) as mock_cross_encoder,
        patch(
            "app.services.rag_service.ChatOpenAI"
        ) as mock_chat_model
    ):

        # -----------------------------------
        # 1. Configure document loader
        # -----------------------------------

        mock_loader.return_value.load_all_documents.return_value = (
            sample_documents
        )

        # -----------------------------------
        # 2. Configure CrossEncoder
        # -----------------------------------

        mock_cross_encoder.return_value.predict.side_effect = (
            lambda pairs: [0.9 - (i * 0.1) for i in range(len(pairs))]
        )

        # -----------------------------------
        # 3. Configure mocked LLM
        # -----------------------------------

        mock_chat_model.return_value.invoke.return_value = AIMessage(
            content=(
                "The examination fee for academic year "
                "2026-2027 is Rs. 2000."
            )
        )

        # -----------------------------------
        # 4. Run ingestion pipeline
        # -----------------------------------

        ingestion_service = IngestionService()

        ingestion_result = ingestion_service.ingest_documents()

        assert ingestion_result["documents_loaded"] == 3
        assert ingestion_result["documents_cleaned"] == 3
        assert ingestion_result["chunks_created"] > 0
        assert ingestion_result["chunks_indexed"] > 0

        # -----------------------------------
        # 5. Run RAG pipeline
        # -----------------------------------

        rag_service = RAGService()

        query = (
            "What is the examination fee for 2026-2027?"
        )

        answer = rag_service.generate_response(
            users_query=query,
            retrieve_k=3,
            rerank_k=2
        )

        # -----------------------------------
        # 6. Validate final response
        # -----------------------------------

        assert isinstance(answer, dict)
        assert "Rs. 2000" in answer["answer"]

        # Verify LLM was called
        mock_chat_model.return_value.invoke.assert_called_once()

        # Verify reranker was used
        mock_cross_encoder.return_value.predict.assert_called_once()

        # Verify prompt contains user question
        prompt_value = (
            mock_chat_model.return_value.invoke.call_args.args[0]
        )

        prompt_text = prompt_value.to_string()

        assert query in prompt_text
        assert "examination fee" in prompt_text.lower()

        print("Ingestion result:", ingestion_result)
        print("Generated answer:", answer)
