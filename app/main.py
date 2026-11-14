
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, init_db, DocumentRecord
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    ChunkStrategy
)
from app.vector_store import vector_service
from app.graph_agent import graph_rag_service


# Initialize database when application starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="LangChain & LangGraph RAG API",
    lifespan=lifespan
)


# 1. Upload and process documents
@app.post("/api/v1/documents/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    strategy: ChunkStrategy = Form(ChunkStrategy.RECURSIVE),
    db: Session = Depends(get_db)
):

    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    try:
        # Read uploaded file
        content = await file.read()

        # Extract text from PDF, DOCX or TXT
        text = vector_service.extract_text(
            content,
            file.filename
        )

        # Split text into chunks
        documents = vector_service.chunk_text(
            text,
            strategy
        )

        # Store chunks in vector database
        count = vector_service.add_documents(
            documents,
            file.filename
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Save document information in SQL database
    record = DocumentRecord(
        filename=file.filename,
        strategy=strategy.value,
        chunk_count=count
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return IngestResponse(
        document_id=record.id,
        filename=record.filename,
        strategy=strategy,
        chunks_stored=count
    )


# 2. Chat with the RAG agent
@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):

    answer, booking = graph_rag_service.run(
        session_id=payload.session_id,
        query=payload.query,
        db=db
    )

    return ChatResponse(
        session_id=payload.session_id,
        answer=answer,
        booking=booking
    )
