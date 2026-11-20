from fastapi import FastAPI,HTTPException
from app.core.config import settings

app = FastAPI(title=settings.app_name,
              debug=settings.debug,
              version="0.0.1",
              description="An AI-powered college assistant"
              )

@app.get("/health")
def health_check():
    return {
        "status":"healthy",
        "message":"SmartCampus Assistant is running..."
        }