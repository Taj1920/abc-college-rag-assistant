
import sqlite3
import pytest

import app.ingestion.document_registry as registry_module
from app.ingestion.document_registry import DocumentRegistry


@pytest.fixture
def registry(tmp_path, monkeypatch):
    # Use a temporary directory instead of the actual project storage
    monkeypatch.setattr(registry_module, "ROOT", tmp_path)

    return DocumentRegistry()


def test_create_table(registry):
    conn = registry.get_db_connection()

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )

        tables = [row[0] for row in cursor.fetchall()]

        assert "documents" in tables

    finally:
        conn.close()


def test_register_document(registry):
    document_id = registry.register_document(
        original_filename="exam_rules.pdf",
        stored_filename="abc123.pdf",
        file_hash="hash_123",
        uploaded_at="2026-11-26T10:30:00+00:00"
    )

    assert document_id.startswith("DOC_")


def test_document_initial_status_is_pending(registry):
    document_id = registry.register_document(
        original_filename="fee_details.pdf",
        stored_filename="xyz123.pdf",
        file_hash="hash_456",
        uploaded_at="2026-11-26T10:30:00+00:00"
    )

    conn = registry.get_db_connection()

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, processed_at FROM documents WHERE document_id=?",
            (document_id,)
        )

        result = cursor.fetchone()

        assert result[0] == "pending"
        assert result[1] is None

    finally:
        conn.close()


def test_get_document_by_hash(registry):
    file_hash = "hash_789"

    document_id = registry.register_document(
        original_filename="exam_schedule.pdf",
        stored_filename="schedule123.pdf",
        file_hash=file_hash,
        uploaded_at="2026-11-26T10:30:00+00:00"
    )

    existing_id = registry.get_document_by_hash(file_hash)

    assert existing_id == document_id


def test_get_document_by_unknown_hash(registry):
    result = registry.get_document_by_hash("unknown_hash")

    assert result is None


def test_get_document_by_empty_hash(registry):
    assert registry.get_document_by_hash("") is None


def test_duplicate_file_hash_is_rejected(registry):
    file_hash = "duplicate_hash"

    registry.register_document(
        original_filename="exam_rules.pdf",
        stored_filename="file1.pdf",
        file_hash=file_hash,
        uploaded_at="2026-11-26T10:30:00+00:00"
    )

    with pytest.raises(sqlite3.IntegrityError):
        registry.register_document(
            original_filename="exam_rules_copy.pdf",
            stored_filename="file2.pdf",
            file_hash=file_hash,
            uploaded_at="2026-11-26T10:35:00+00:00"
        )
