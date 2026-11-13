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