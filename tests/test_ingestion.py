from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.ingestion.loader import DataLoader


@pytest.fixture
def data_loader(tmp_path):
    """Create a DataLoader using a temporary directory."""

    loader = DataLoader()
    loader.docs_path = tmp_path

    return loader


# --------------------------------------------------
# 1. FILE DISCOVERY TESTS
# --------------------------------------------------

def test_discover_supported_files(data_loader):

    (data_loader.docs_path / "handbook.pdf").touch()
    (data_loader.docs_path / "admission.docx").touch()
    (data_loader.docs_path / "notes.txt").touch()
    (data_loader.docs_path / "image.xyz").touch()

    files = data_loader.discover_files()

    file_names = {file.name for file in files}

    assert "handbook.pdf" in file_names
    assert "admission.docx" in file_names
    assert "image.xyz" not in file_names


def test_discover_files_recursively(data_loader):

    subfolder = data_loader.docs_path / "academic"
    subfolder.mkdir()

    (subfolder / "calendar.pdf").touch()

    files = data_loader.discover_files()

    assert len(files) == 1
    assert files[0].name == "calendar.pdf"


def test_discover_files_when_directory_is_empty(data_loader):

    files = data_loader.discover_files()

    assert files == []


def test_discover_files_when_directory_does_not_exist(data_loader):

    data_loader.docs_path = data_loader.docs_path / "missing"

    files = data_loader.discover_files()

    assert files == []


def test_discover_files_returns_path_objects(data_loader):

    (data_loader.docs_path / "handbook.pdf").touch()

    files = data_loader.discover_files()

    assert all(isinstance(file, Path) for file in files)


def test_discover_files_returns_sorted_results(data_loader):

    (data_loader.docs_path / "z_handbook.pdf").touch()
    (data_loader.docs_path / "a_handbook.pdf").touch()

    files = data_loader.discover_files()

    assert files == sorted(files)


# --------------------------------------------------
# 2. DUPLICATE DOCUMENT TESTS
# --------------------------------------------------

def test_remove_exact_duplicate_documents(data_loader):

    doc1 = Document(
        page_content="ABC College requires 75% attendance.",
        metadata={"source": "handbook.pdf"}
    )

    doc2 = Document(
        page_content="ABC College requires 75% attendance.",
        metadata={"source": "handbook_copy.pdf"}
    )

    result = data_loader.remove_duplicate_docs([doc1, doc2])

    assert len(result) == 1
    assert result[0].metadata["source"] == "handbook.pdf"


def test_remove_duplicates_with_different_whitespace(data_loader):

    doc1 = Document(
        page_content="ABC College requires 75% attendance."
    )

    doc2 = Document(
        page_content="ABC College requires\n\n75%   attendance."
    )

    result = data_loader.remove_duplicate_docs([doc1, doc2])

    assert len(result) == 1


def test_keep_documents_with_different_content(data_loader):

    doc1 = Document(
        page_content="Attendance requirement is 75%."
    )

    doc2 = Document(
        page_content="Examination fee is Rs. 2000."
    )

    result = data_loader.remove_duplicate_docs([doc1, doc2])

    assert len(result) == 2


def test_empty_document_list(data_loader):

    result = data_loader.remove_duplicate_docs([])

    assert result == []


def test_skip_empty_documents(data_loader):

    doc1 = Document(page_content="")
    doc2 = Document(page_content="   \n  ")

    result = data_loader.remove_duplicate_docs([doc1, doc2])

    assert result == []


def test_keep_only_first_occurrence_of_duplicate(data_loader):

    doc1 = Document(
        page_content="ABC College offers B.Tech."
    )

    doc2 = Document(
        page_content="ABC College offers B.Tech."
    )

    doc3 = Document(
        page_content="ABC College offers MBA."
    )

    result = data_loader.remove_duplicate_docs(
        [doc1, doc2, doc3]
    )

    assert len(result) == 2
    assert result[0].page_content == doc1.page_content
    assert result[1].page_content == doc3.page_content


# --------------------------------------------------
# 3. DOCUMENT LOADING TESTS
# --------------------------------------------------

@patch("app.ingestion.loader.DoclingLoader")
def test_load_all_documents_successfully(
    mock_docling,
    data_loader
):

    (data_loader.docs_path / "handbook.pdf").touch()

    expected_doc = Document(
        page_content="ABC College offers B.Tech, BBA and MBA.",
        metadata={"source": "handbook.pdf"}
    )

    mock_docling.return_value.load.return_value = [expected_doc]

    documents = data_loader.load_all_documents()

    assert len(documents) == 1
    assert isinstance(documents[0], Document)
    assert documents[0].page_content == expected_doc.page_content
    assert documents[0].metadata["source"] == "handbook.pdf"

    mock_docling.assert_called_once()


@patch("app.ingestion.loader.DoclingLoader")
def test_load_all_documents_returns_flat_list(
    mock_docling,
    data_loader
):

    (data_loader.docs_path / "handbook.pdf").touch()
    (data_loader.docs_path / "admission.pdf").touch()

    doc1 = Document(page_content="Student handbook")
    doc2 = Document(page_content="Attendance policy")
    doc3 = Document(page_content="Admission guidelines")

    mock_docling.side_effect = [
        MagicMock(load=MagicMock(return_value=[doc1, doc2])),
        MagicMock(load=MagicMock(return_value=[doc3]))
    ]

    documents = data_loader.load_all_documents()

    assert len(documents) == 3
    assert all(isinstance(doc, Document) for doc in documents)


@patch("app.ingestion.loader.DoclingLoader")
def test_loader_continues_after_file_failure(
    mock_docling,
    data_loader
):

    corrupted_file = data_loader.docs_path / "corrupted.pdf"
    valid_file = data_loader.docs_path / "handbook.pdf"

    corrupted_file.touch()
    valid_file.touch()

    valid_doc = Document(
        page_content="Valid document"
    )

    # Explicit order avoids depending on filesystem order.
    data_loader.discover_files = MagicMock(
        return_value=[corrupted_file, valid_file]
    )

    mock_docling.side_effect = [
        RuntimeError("Unable to process corrupted file"),
        MagicMock(load=MagicMock(return_value=[valid_doc]))
    ]

    documents = data_loader.load_all_documents()

    assert len(documents) == 1
    assert documents[0].page_content == "Valid document"


def test_load_all_documents_when_no_files_exist(data_loader):

    documents = data_loader.load_all_documents()

    assert documents == []


@patch("app.ingestion.loader.DoclingLoader")
def test_load_all_documents_removes_duplicates(
    mock_docling,
    data_loader
):

    (data_loader.docs_path / "handbook.pdf").touch()
    (data_loader.docs_path / "handbook_copy.pdf").touch()

    duplicate_doc1 = Document(
        page_content="ABC College requires 75% attendance."
    )

    duplicate_doc2 = Document(
        page_content="ABC College requires 75% attendance."
    )

    mock_docling.side_effect = [
        MagicMock(load=MagicMock(return_value=[duplicate_doc1])),
        MagicMock(load=MagicMock(return_value=[duplicate_doc2]))
    ]

    documents = data_loader.load_all_documents()

    assert len(documents) == 1
    assert documents[0].page_content == duplicate_doc1.page_content
