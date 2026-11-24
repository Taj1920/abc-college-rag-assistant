from pydantic import BaseModel,Field


class IngestResponse(BaseModel):
    documents_loaded: int = Field(default=0)
    documents_cleaned: int = Field(default=0)
    chunks_created: int = Field(default=0)
    chunks_indexed: int = Field(default=0)