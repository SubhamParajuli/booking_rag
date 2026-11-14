
import json
from typing import TypedDict, Optional, Any,Annotated

from datetime import datetime
from zoneinfo import ZoneInfo

import redis

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph.message import add_messages

from sqlalchemy.orm import Session

from app.config import settings
from app.database import BookingRecord
from app.schemas import BookingDetails
from app.vector_store import vector_service


# 1. Define the state of our agent
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    booking_result: Optional[dict]


class GraphRAGService:

    def __init__(self):

        # Initialize LLM
        self.llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0.2
        )

        # Initialize retriever
        self.retriever = vector_service.get_retriever(k=3)

        # Initialize Redis memory
        self.redis_client = redis.from_url(settings.redis_url)
        self.memory = RedisSaver(redis_client=self.redis_client)
        self.memory.setup()

        # Create LangGraph workflow
        self.graph = self.build_graph()


    # 2. Retrieve relevant documents
    def retrieve(self, state: AgentState):

        question = state["messages"][-1].content

        documents = self.retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in documents]
        )

        if not context:
            context = "No relevant information found."

        return {"context": context}


    # 3. Generate LLM response
    def generate_response(self, state: AgentState):

        today = datetime.now(
            ZoneInfo("Asia/Kathmandu")
        ).strftime("%Y-%m-%d")

        system_prompt = f"""
        You are a professional recruiting assistant.

        Today's date: {today}
        Timezone: Asia/Kathmandu.

        Your responsibilities:

        1. Answer questions using the retrieved context.
        2. Understand natural language.
        3. Remember information from previous messages.
        4. Collect booking details:
        - name
        - email
        - date
        - time

        BOOKING RULES:

        - Extract information from the entire conversation.
        - Convert dates to YYYY-MM-DD.
        - Convert times to 24-hour HH:MM.
        - Understand expressions like tomorrow, next Friday,
        afternoon, 2 PM, etc.
        - Never invent missing information.
        - Ask the user if information is missing or ambiguous.
        - Respect corrections made by the user.
        - Call book_interview only when all details are available.

        Context:
        {state["context"]}
        """

        response = self.llm_with_tools.invoke(
            [SystemMessage(content=system_prompt)] + state["messages"]
        )

        return {"messages": [response]}


    # 4. Save booking in database
    def save_booking(self, args: dict):

        booking = BookingRecord(
            name=args["name"],
            email=args["email"],
            date=args["date"],
            time=args["time"]
        )

        self.db.add(booking)
        self.db.commit()
        self.db.refresh(booking)

        return {
            "status": "success",
            "booking_id": booking.id,
            "name": booking.name,
            "email": booking.email,
            "date": booking.date,
            "time": booking.time
        }


    # 5. Execute tool calls
    def execute_tools(self, state: AgentState):

        last_message = state["messages"][-1]

        tool_messages = []
        booking_result = None

        for tool_call in last_message.tool_calls:

            if tool_call["name"] == "book_interview":

                result = self.save_booking(tool_call["args"])

                booking_result = result

                tool_messages.append(
                    ToolMessage(
                        content=json.dumps(result),
                        tool_call_id=tool_call["id"]
                    )
                )

        return {
            "messages": tool_messages,
            "booking_result": booking_result
        }


    # 6. Decide whether to execute tools or finish
    def route(self, state: AgentState):

        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tools"

        return END


    # 7. Build LangGraph workflow
    def build_graph(self):

        # Define booking tool
        booking_tool = {
                "name": "book_interview",
                "description": "Save an interview booking. Use only after collecting all required details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "date": {
                            "type": "string",
                            "description": "Interview date in YYYY-MM-DD format"
                        },
                        "time": {
                            "type": "string",
                            "description": "Interview time in 24-hour HH:MM format"
                        }
                    },
                    "required": ["name", "email", "date", "time"]
                }
            }

        self.llm_with_tools = self.llm.bind_tools([booking_tool])

        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("agent", self.generate_response)
        workflow.add_node("tools", self.execute_tools)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "agent")

        workflow.add_conditional_edges(
            "agent",
            self.route,
            {
                "tools": "tools",
                END: END
            }
        )

        workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=self.memory)


    # 8. Run the chatbot
    def run(
        self,
        session_id: str,
        query: str,
        db: Session
    ):

        self.db = db

        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "context": "",
                "booking_result": None
            },
            config=config
        )

        answer = result["messages"][-1].content

        booking_details = None

        if result["booking_result"]:

            booking = result["booking_result"]

            booking_details = BookingDetails(
                id=booking["booking_id"],
                name=booking["name"],
                email=booking["email"],
                date=booking["date"],
                time=booking["time"]
            )

        return answer, booking_details


# Create service
graph_rag_service = GraphRAGService()
