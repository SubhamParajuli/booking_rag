from enum import Enum
from typing import Optional
from pydantic import BaseModel,EmailStr,Field

class ChunkStrategy(str,Enum):
    RECURSIVE="recursive"
    CHARACTER="character"

class IngestResponse(BaseModel):
    document_id:int
    filename:str
    strategy:ChunkStrategy
    chunks_stored:int

class ChatRequest(BaseModel):
    session_id:str= Field(..., min_length=1)
    query:str=Field(..., min_length=1)

class BookingDetails(BaseModel):
    id:int
    name:str
    email:EmailStr
    date:str
    time:str

class ChatResponse(BaseModel):
    session_id:str
    answer:str
    booking: Optional[BookingDetails]=None