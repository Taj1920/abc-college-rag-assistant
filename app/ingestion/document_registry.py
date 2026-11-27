import sqlite3
import uuid
from pathlib import Path
from app.core.logger import get_logger


#logger
logger = get_logger(__name__)
#path
ROOT = Path(__file__).resolve().parents[2]

class DocumentRegistry:
    def __init__(self):
        self.db_path = ROOT / "storage" / "document_registry.db"
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        self.create_table()


    def get_db_connection(self):
        try:
            conn=sqlite3.connect(self.db_path)
            return conn
        except Exception:
            logger.exception("Failed to connect Document registry DB.")
            raise

    def create_table(self):
        query = """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    processed_at TEXT
                );
                """
        conn = None
        try:
            conn=sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            logger.info("Document registry initialized successfully!")
        except Exception:
            logger.exception("Failed to initialize Document registry.")
            raise
        finally:
            if conn is not None:
                conn.close()

    def get_document_by_hash(self,file_hash:str):
        if not file_hash or not file_hash.strip():
            logger.warning("File hash should not be empty")
            return None

        query = "SELECT document_id FROM documents WHERE file_hash=?"
        conn = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query,(file_hash,))
            result =  cursor.fetchone()
            return result[0] if result else None
        
        except sqlite3.Error:
            logger.exception("Failed to retrieve document from registry.")
            raise

        finally:
            if conn is not None:
                conn.close()

    
    def register_document(self,original_filename:str,
                          stored_filename:str,file_hash:str,
                          uploaded_at:str,processed_at:str | None=None):
        document_id = f"DOC_{uuid.uuid4()}"
        status = "pending"
        query = """
                INSERT INTO documents(document_id,original_filename,
                          stored_filename,file_hash,status,
                          uploaded_at,processed_at) VALUES(?,?,?,?,?,?,?)
                """
        conn = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query,(document_id,original_filename,
                          stored_filename,file_hash,status,
                          uploaded_at,processed_at))
            conn.commit()
            logger.info(f"Document inserted successfully!: {document_id}")
            return document_id
        except Exception:
            logger.exception("Document insertion failed")
            raise
        finally:
            if cursor is not None:
                conn.close()

    def update_document_status(self,document_id:str,status:str,processed_at:str | None=None):
        query="""
            UPDATE  documents
            SET status = ?,processed_at=? 
            WHERE document_id=? 
            """
        
        conn = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute(query,(status,processed_at,document_id))
            
            #check document exists
            if cursor.rowcount==0:
                raise ValueError(f"Document not found: {document_id}")
            
            conn.commit()
            logger.info(f"Document :{document_id}. status updated")
        except Exception:
            if conn is not None:
                conn.rollback()
            logger.exception("Failed to update document status.")
            raise

        finally:
            if conn is not None:
                conn.close()