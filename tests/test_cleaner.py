import pytest
from langchain_core.documents import Document

from app.ingestion.cleaner import DocumentCleaner


@pytest.fixture
def cleaner():
    return DocumentCleaner()


def test_remove_extra_spaces(cleaner):
    doc = Document(page_content="ABC    College   offers B.Tech")

    result = cleaner.clean_document(doc)

    assert result.page_content == "ABC College offers B.Tech"


def test_preserve_paragraphs(cleaner):
    doc = Document(
        page_content="First paragraph.\n\n\n\nSecond paragraph."
    )

    result = cleaner.clean_document(doc)

    assert result.page_content == (
        "First paragraph.\n\nSecond paragraph."
    )


def test_normalize_line_endings(cleaner):
    doc = Document(page_content="Hello\r\nWorld\r\n")

    result = cleaner.clean_document(doc)

    assert result.page_content == "Hello\nWorld"


def test_remove_control_characters(cleaner):
    doc = Document(page_content="ABC\x00 College\x07")

    result = cleaner.clean_document(doc)

    assert result.page_content == "ABC College"


def test_remove_leading_and_trailing_whitespace(cleaner):
    doc = Document(page_content="   ABC College   ")

    result = cleaner.clean_document(doc)

    assert result.page_content == "ABC College"


def test_preserve_metadata(cleaner):
    metadata = {
        "source": "handbook.pdf",
        "page": 1
    }

    doc = Document(
        page_content="ABC College",
        metadata=metadata
    )

    result = cleaner.clean_document(doc)

    assert result.metadata == metadata
    assert result.metadata is not doc.metadata


def test_skip_empty_documents(cleaner):
    docs = [
        Document(page_content="ABC College"),
        Document(page_content="   \n  "),
        Document(page_content="B.Tech course")
    ]

    result = cleaner.clean_documents(docs)

    assert len(result) == 2


def test_clean_multiple_documents(cleaner):
    docs = [
        Document(page_content="  ABC   College  "),
        Document(page_content="  B.Tech   Course  ")
    ]

    result = cleaner.clean_documents(docs)

    assert len(result) == 2
    assert result[0].page_content == "ABC College"
    assert result[1].page_content == "B.Tech Course"


def test_empty_input(cleaner):
    result = cleaner.clean_documents([])

    assert result == []
