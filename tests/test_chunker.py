
import pytest
from langchain_core.documents import Document

from app.ingestion.chunker import DocumentChunker


class TestDocumentChunker:

    def test_empty_document_list(self):
        chunker = DocumentChunker()

        chunks = chunker.chunk_documents([])

        assert chunks == []

    def test_short_document(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=10)

        doc = Document(
            page_content="ABC College offers B.Tech and MBA courses.",
            metadata={"source": "courses.txt"}
        )

        chunks = chunker.chunk_documents([doc])

        assert len(chunks) == 1
        assert chunks[0].page_content == doc.page_content

    def test_large_document_creates_multiple_chunks(self):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

        doc = Document(
            page_content="ABC College offers various courses. " * 20
        )

        chunks = chunker.chunk_documents([doc])

        assert len(chunks) > 1

    def test_chunk_size_limit(self):
        chunk_size = 50
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=10
        )

        doc = Document(
            page_content="ABC College offers various courses. " * 20
        )

        chunks = chunker.chunk_documents([doc])

        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size

    def test_chunk_overlap(self):
        chunker = DocumentChunker(
            chunk_size=10,
            chunk_overlap=3
        )

        # No spaces, so the splitter uses character-level splitting.
        doc = Document(page_content="abcdefghijklmnopqrstuvwxyz")

        chunks = chunker.chunk_documents([doc])

        assert len(chunks) > 1

        # Each subsequent chunk should begin with the
        # final 3 characters of the previous chunk.
        for i in range(len(chunks) - 1):
            previous = chunks[i].page_content
            current = chunks[i + 1].page_content

            assert previous[-3:] == current[:3]

    def test_metadata_preservation(self):
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)

        metadata = {
            "source": "student_handbook.pdf",
            "page": 2
        }

        doc = Document(
            page_content="ABC College provides various facilities for students. " * 5,
            metadata=metadata
        )

        chunks = chunker.chunk_documents([doc])

        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.metadata == metadata

    def test_multiple_documents(self):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

        docs = [
            Document(
                page_content="ABC College offers B.Tech courses.",
                metadata={"source": "courses.txt"}
            ),
            Document(
                page_content="The examination fee is Rs. 2000.",
                metadata={"source": "fees.txt"}
            )
        ]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) == 2
        assert chunks[0].metadata["source"] == "courses.txt"
        assert chunks[1].metadata["source"] == "fees.txt"

    def test_custom_chunk_configuration(self):
        chunker = DocumentChunker(
            chunk_size=100,
            chunk_overlap=20
        )

        assert chunker.splitter._chunk_size == 100
        assert chunker.splitter._chunk_overlap == 20

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=0)

        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=-10)

    def test_invalid_chunk_overlap(self):
        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=-1)

        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=150)
