from pydantic import BaseModel,Field


#Request
class ChatRequest(BaseModel):
    question:str = Field(
        ...,
        min_length=1,
        description="Question asked by the student"
    )

#source info
class SourceInfo(BaseModel):
    source:str
    score:float

#Response
class ChatResponse(BaseModel):
    answer : str 
    sources:list[SourceInfo] = Field(default_factory=list)