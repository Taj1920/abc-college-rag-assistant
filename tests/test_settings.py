from app.core.config import settings

print("App:", settings.app_name)
print("Debug:", settings.debug)
print("LLM:", settings.llm_model)
print("Embedding:", settings.embedding_model)
print("Chroma:", settings.chroma_persist_dir)
print("API key configured:", settings.openai_api_key is not None)