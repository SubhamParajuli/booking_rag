from io import BytesIO
from typing import List,Any,Dict,Optional,Tuple,TypedDict
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,ToolMessage
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from app.config import settings
from app.schemas import ChunkStrategy

class VectorStoreService:
    def __init__(self)->None:
        self.client=QdrantClient(path=settings.qdrant_path)
        self.embeddings=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.vector_size=384
        self._ensure_collection()


        self.vector_store=QdrantVectorStore(
            client=self.client,
            collection_name=settings.collection_name,
            embedding=self.embeddings
        )

        def _ensure_collection(self) -> None:
            collections = [c.name for c in self.client.get_collections().collections]
            if settings.collection_name not in collections:
                self.client.create_collection(
                    collection_name=settings.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )


        @staticmethod
        def extract_text(content:bytes, filename:str)->str:
            lower=filename.lower()
            if lower.endswith(".txt"):
                return content.decode("utf-8",errors="ignore")
            if lower.endswith(".pdf"):
                reader = PdfReader(BytesIO(content))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            raise ValueError("Unsupported file format. Use .pdf or .txt.")

        def chunk_text(self, text: str, strategy: ChunkStrategy) -> List[Document]:
            if strategy == ChunkStrategy.RECURSIVE:
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            else:
                splitter = CharacterTextSplitter(separator="\n", chunk_size=400, chunk_overlap=40)

            return splitter.create_documents([text])

        def add_documents(self, documents: List[Document], filename: str) -> int:
            for doc in documents:
                doc.metadata = {"source": filename}
            self.vector_store.add_documents(documents)
            return len(documents)

        def get_retriever(self, k: int = 3):
            return self.vector_store.as_retriever(search_kwargs={"k": k})


vector_service = VectorStoreService()