import json
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session
from app.config import settings
from app.database import BookingRecord
from app.schemas import BookingDetails
from app.vector_store import vector_service

class AgentState(TypedDict):
    messages:List[BaseMessage]
    context:str
    booking_result:Optional[Dict[str,Any]]

class GraphRAGService:
    def __init__(self)->None:
        self.llm=ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0.2
        )
        self.retriever=vector_service.get_retriever(k=3)
        self.memory = RedisSaver.from_conn_string(settings.redis_url)
        self.memory.setup()

    def _save_booking_to_db(self,db:Session,args:Dict[str,str])->str:
        """Helper to persist the booking in SQlite and return a JSON Status."""
        record=BookingRecord(
            name=args["name"],
            email=args["email"],
            date=args["date"],
            time=args["time"],
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return json.dumps({
            "status":"success",
            "booking_id":record.id,
            "name":record.name,
            "email":record.email,
            "date":record.date,
            "time":record.time,
        })

    def run(self,session_id:str,query:str,db:Session)->Tuple[str,Optional[BookingDetails]]:
        """
        Builds and executes the graph on-the-fly, using Redis checkpointing 
        for state and binding the current request's database session (db).
        """

        def retrieve_node(state:AgentState)->Dict[str,Any]:
            """Step 1: Fetch relevant document chunks from Qdrant based on user query."""
            messages=state["messages"]
            last_query=messages[-1].content if messages else query
            docs=self.retriever.invoke(last_query)
            context = "\n---\n".join([doc.page_content for doc in docs]) if docs else "No context found."
            return {"context":context}

        def model_node(state: AgentState) -> Dict[str, Any]:
            """Step 2: Invoke Groq LLM with system prompt, RAG context, and bound booking tool."""
            messages = state["messages"]
            context = state["context"]

            system_prompt = (
                "You are a professional recruiting assistant. Answer strictly using the provided context. "
                "If the user wants to book an interview, collect their name, email, date (YYYY-MM-DD), and time (HH:MM), "
                "then call the `book_interview` tool. Ask politely if any details are missing.\n\n"
                f"Context:\n{context}"
            )

            bound_llm = self.llm.bind_tools([
                {
                    "name": "book_interview",
                    "description": "Save an interview booking once all candidate details are provided.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "date": {"type": "string", "description": "YYYY-MM-DD"},
                            "time": {"type": "string", "description": "HH:MM"},
                        },
                        "required": ["name", "email", "date", "time"],
                    },
                }
            ])

            response = bound_llm.invoke([SystemMessage(content=system_prompt)] + messages)
            return {"messages": [response]}

        def tool_node(state: AgentState) -> Dict[str, Any]:
            """Step 3: Execute the tool if requested by the LLM and save to SQLite."""
            messages = state["messages"]
            last_msg = messages[-1]
            booking_payload = None
            tool_messages = []

            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    if tool_call["name"] == "book_interview":
                        result_json = self._save_booking_to_db(db, tool_call["args"])
                        booking_payload = json.loads(result_json)

                        tool_messages.append(
                            ToolMessage(content=result_json, tool_call_id=tool_call["id"])
                        )

            return {"messages": tool_messages, "booking_result": booking_payload}

        # --- Assemble the LangGraph Workflow ---
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("agent", model_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "agent")

        def route_after_agent(state: AgentState):
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                return "tools"
            return END

        workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        # Compile graph with Redis checkpointer (keyed by session_id/thread_id)
        graph = workflow.compile(checkpointer=self.memory)
        config = {"configurable": {"thread_id": session_id}}

        # Execute graph workflow
        output = graph.invoke({"messages": [HumanMessage(content=query)]}, config=config)

        # Extract final answer text and structured booking details
        final_messages = output["messages"]
        answer = final_messages[-1].content if final_messages else "Done."
        
        booking_raw = output.get("booking_result")
        booking_details: Optional[BookingDetails] = None
        if booking_raw:
            booking_details = BookingDetails(
                id=booking_raw["booking_id"],
                name=booking_raw["name"],
                email=booking_raw["email"],
                date=booking_raw["date"],
                time=booking_raw["time"],
            )

        return answer, booking_details

graph_rag_service = GraphRAGService()