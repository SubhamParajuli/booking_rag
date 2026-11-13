import json
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy.orm import Session
from app.config import settings
from app.database import BookingRecord
from app.schemas import BookingDetails
from app.vector_store import vector_service